import json
import sys

import pytest

from conftest import API_KEY, completion_body  # noqa: F401
from evals import cli, flground_adapter
from evals.baseline import make_prediction
from evals.errors import EXIT_ERROR, EXIT_OK
from evals.flground_adapter import FLGroundPredictor
from flground.contracts import CallRecord, Point
from workspace_fixture import workspace  # noqa: F401

HARNESS = """
from evals.baseline import make_prediction

class Harness:
    def __init__(self, client):
        self.client = client
    def predict(self, image, instruction):
        return make_prediction(point_px=[100.0, 50.0], latency_seconds=1.0)

def factory(client):
    return Harness(client)

def malformed(client):
    class Bad:
        def predict(self, image, instruction):
            return {"point_px": [1.0, 1.0]}
    return Bad()
"""


@pytest.fixture
def harness_module(workspace, tmp_path, monkeypatch):
    path = tmp_path / "my_harness.py"
    path.write_text(HARNESS)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("my_harness", None)
    yield path
    sys.modules.pop("my_harness", None)


def report_of(out):
    return json.loads(out.with_suffix(".report.json").read_text())


def test_a_teammates_harness_gets_the_shared_scorer_and_report(workspace, harness_module):
    out = workspace / "outputs/mine.jsonl"
    assert cli.main(["run", "--predictor", "my_harness:factory", "--limit", "2", "--out", str(out)]) == EXIT_OK
    written = report_of(out)
    assert written["mode"] == "custom:my_harness:factory"
    assert written["overall"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    assert written["paper_metrics"]["num_correct_action"] == 2
    assert written["finalized"] is True


def test_harness_defaults_to_the_dev_split(workspace, harness_module):
    out = workspace / "outputs/mine.jsonl"
    assert cli.main(["run", "--predictor", "my_harness:factory", "--out", str(out)]) == EXIT_OK
    assert report_of(out)["split"] == "dev"
    assert report_of(out)["expected_targets"] == 8


def test_editing_a_harness_blocks_resume(workspace, harness_module, server):
    out = workspace / "outputs/mine.jsonl"
    code = cli.main(["run", "--predictor", "my_harness:factory", "--limit", "1", "--out", str(out)])
    assert code == EXIT_OK
    config = out.with_suffix(".config.json")
    report = out.with_suffix(".report.json")
    # Reopen the finished run, so resume reaches the configuration check.
    config.chmod(0o644)
    report.chmod(0o644)
    report.write_text(json.dumps({"finalized": False}))
    harness_module.write_text(HARNESS + "\n# edited\n")
    assert cli.main(["resume", "--run", str(out)]) == EXIT_ERROR


def test_malformed_harness_result_is_refused_clearly(workspace, harness_module, capsys):
    out = workspace / "outputs/mine.jsonl"
    assert cli.main(["run", "--predictor", "my_harness:malformed", "--out", str(out)]) == EXIT_ERROR
    assert "make_prediction" in capsys.readouterr().err


@pytest.mark.parametrize("spec", ["no_colon", "missing_module_xyz:factory", "my_harness:absent"])
def test_unloadable_predictor_names_the_problem_and_writes_nothing(workspace, harness_module, capsys, spec):
    out = workspace / "outputs/mine.jsonl"
    assert cli.main(["run", "--predictor", spec, "--out", str(out)]) == EXIT_ERROR
    assert "predictor" in capsys.readouterr().err
    assert not out.exists()


def test_limit_must_be_positive(workspace):
    assert cli.main(["run", "--limit", "0"]) == EXIT_ERROR


class FakeModel:
    model = "fake"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def predict(self, image, instruction):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _adapter(monkeypatch, strategy, outcomes, api_key="key-not-real"):
    monkeypatch.setenv("FIREWORKS_API_KEY", api_key)
    monkeypatch.setattr(flground_adapter, "FireworksModel", lambda: FakeModel(outcomes))
    return FLGroundPredictor(strategy)


def _call(x, y, usage=None):
    return CallRecord(point=Point(x=x, y=y), raw_response=f'{{"x": {x}, "y": {y}}}', latency_seconds=0.5,
                      usage=usage or {})


def test_adapter_returns_pixels_and_every_call(monkeypatch, image_path):
    usage = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    predictor = _adapter(monkeypatch, "baseline", [_call(100.0, 50.0, usage)])
    prediction = predictor.predict(image_path, "click")
    assert prediction["point_px"] == [100.0, 50.0]
    assert prediction["point_normalized"] is None
    assert prediction["error"] is None
    assert prediction["calls"][0]["usage"] == usage


def test_adapter_refine_records_two_calls_and_maps_back_to_the_original_image(monkeypatch, image_path):
    predictor = _adapter(monkeypatch, "refine", [_call(100.0, 50.0), _call(100.0, 50.0)])
    prediction = predictor.predict(image_path, "click")
    assert len(prediction["calls"]) == 2
    assert 0 <= prediction["point_px"][0] <= 200
    assert 0 <= prediction["point_px"][1] <= 100


@pytest.mark.parametrize("factory,calls", [(flground_adapter.baseline, 1), (flground_adapter.refine, 2)])
def test_each_factory_runs_its_own_strategy(monkeypatch, image_path, factory, calls):
    monkeypatch.setattr(flground_adapter, "FireworksModel",
                        lambda: FakeModel([_call(100.0, 50.0), _call(100.0, 50.0)]))
    assert len(factory(None).predict(image_path, "click")["calls"]) == calls


def test_adapter_marks_missing_usage_as_unknown(monkeypatch, image_path):
    prediction = _adapter(monkeypatch, "baseline", [_call(1.0, 1.0)]).predict(image_path, "click")
    assert prediction["calls"][0]["usage"] is None


def test_adapter_reports_out_of_bounds_as_invalid_not_error(monkeypatch, image_path):
    prediction = _adapter(monkeypatch, "baseline", [_call(5000.0, 1.0)]).predict(image_path, "click")
    assert prediction["point_px"] is None
    assert prediction["invalid_reason"] == "ValueError"
    assert prediction["error"] is None


def test_adapter_reports_api_failures_in_error_without_the_key(server, monkeypatch, image_path, make_client):
    import openai
    client = make_client()
    server.queue((401, {"error": {"message": f"bad {API_KEY}"}}))
    try:
        client._client.chat.completions.create(model="m", messages=[])
    except openai.APIError as failure:
        error = failure
    prediction = _adapter(monkeypatch, "baseline", [error], api_key=API_KEY).predict(image_path, "click")
    assert prediction["error"]["kind"] == "authentication"
    assert prediction["error"]["systematic"] is True
    assert API_KEY not in prediction["error"]["message"]


def test_adapter_runs_under_the_shared_command(workspace, server, monkeypatch):
    monkeypatch.setattr(flground_adapter, "FireworksModel",
                        lambda: FakeModel([_call(100.0, 50.0)] * 2))
    out = workspace / "outputs/flground.jsonl"
    code = cli.main(["run", "--predictor", "evals.flground_adapter:baseline", "--limit", "2", "--out", str(out)])
    assert code == EXIT_OK
    assert report_of(out)["overall"]["correct"] == 2


def test_make_prediction_defaults_are_a_valid_result():
    assert make_prediction() == {
        "point_px": None, "point_normalized": None, "calls": [], "latency_seconds": 0.0,
        "fallback_used": False, "invalid_reason": None, "error": None}
