"""The commands end to end, offline: real CLI and client, mocked HTTP, temporary data."""

import json

import pytest
from PIL import Image

from conftest import API_KEY, MODEL, completion_body
from evals import cli, dataset, evaluate, report
from workspace_fixture import workspace  # noqa: F401
from evals.errors import EXIT_ERROR, EXIT_OK, EXIT_STOPPED, EXIT_TRUNCATED


def run_command(arguments):
    """The final-set form of `evals run`, which every test below that uses it scores on."""
    prefix = [] if "--resume" in arguments else ["--split", "eval"]
    return cli.main(["run", "--mode", "simple_json", *prefix, *arguments])


def only_results(workspace):
    results = list((workspace / "outputs").glob("*.jsonl"))
    assert len(results) == 1
    return results[0]


def records_of(results):
    return [json.loads(line) for line in results.read_text().splitlines()]


def report_of(results):
    return json.loads(report.report_path(results).read_text())


def test_smoke_uses_one_dev_example_and_finishes_with_report_and_overlays(workspace, server, capsys):
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_OK
    assert len(server.requests) == 1
    results = only_results(workspace)
    assert records_of(results)[0]["example_id"] == "dev_0"
    assert report_of(results)["overall"]["correct"] == 1
    assert report_of(results)["finalized"] is True
    assert list(report.overlays_path(results).glob("*.png"))
    assert API_KEY not in capsys.readouterr().out


def test_dev_covers_the_whole_dev_pool(workspace, server):
    assert cli.main(["run"]) == EXIT_OK
    assert len(records_of(only_results(workspace))) == 8


def test_full_covers_every_eval_target_and_no_dev_examples(workspace, server):
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval"]) == EXIT_OK
    ids = [record["example_id"] for record in records_of(only_results(workspace))]
    assert ids == ["eval_0", "eval_1", "eval_2"]


def test_evaluate_entry_matches_the_agreed_command(workspace, server):
    out = workspace / "outputs" / "baseline.jsonl"
    assert run_command(["--out", str(out)]) == EXIT_OK
    assert [r["example_id"] for r in records_of(out)] == ["eval_0", "eval_1", "eval_2"]
    assert report.report_path(out).name == "baseline.report.json"
    assert report.config_path(out).name == "baseline.config.json"


def test_existing_output_is_never_overwritten(workspace, server):
    out = workspace / "outputs" / "baseline.jsonl"
    assert run_command(["--out", str(out)]) == EXIT_OK
    before = out.read_text()
    assert run_command(["--out", str(out)]) == EXIT_ERROR
    assert out.read_text() == before
    assert len(server.requests) == 3


def test_truncated_smoke_exits_distinctly(workspace, server):
    server.queue((200, completion_body(content='{"x"', finish_reason="length")))
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_TRUNCATED


def test_auth_failure_exits_stopped_and_logs_separately(workspace, server):
    server.queue((401, {"error": {"message": "bad"}}))
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval"]) == EXIT_STOPPED
    results = only_results(workspace)
    assert len(server.requests) == 1
    summary = report_of(results)
    assert summary["api_error_count"] == 1
    assert summary["overall"]["total"] == 0
    assert summary["complete"] is False
    assert summary["finalized"] is False


def test_a_lone_503_is_not_reported_as_success_and_resume_recovers(workspace, server):
    server.queue((503, {"error": {"message": "scaling up"}}))
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1", "--out", str(workspace / "outputs/s.jsonl")]) == EXIT_STOPPED
    assert cli.main(["resume", "--run", "s"]) == EXIT_OK
    assert report_of(only_results(workspace))["complete"] is True


@pytest.mark.parametrize("model", [
    "accounts/acme/deployments/12345678",
    "accounts/fireworks/models/qwen3p8-27b#my-deployment",
])
def test_documented_deployment_forms_are_accepted(workspace, server, monkeypatch, model):
    monkeypatch.setenv("FLGROUND_MODEL", model)
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_OK
    assert server.bodies()[0]["model"] == model


def test_rerunning_the_same_config_is_refused(workspace, server):
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_OK
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_ERROR
    assert len(server.requests) == 1


def test_changing_settings_starts_a_separate_run(workspace, server):
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_OK
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1", "--max-tokens", "8192"]) == EXIT_OK
    assert len(list((workspace / "outputs").glob("*.jsonl"))) == 2


def test_resume_finishes_after_a_stop_without_repeating_work(workspace, server):
    out = str(workspace / "outputs/r.jsonl")
    server.queue((200, completion_body()), (401, {"error": {"message": "bad"}}))
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", out]) == EXIT_STOPPED
    assert run_command(["--out", out, "--resume"]) == EXIT_OK
    summary = report_of(only_results(workspace))
    assert summary["complete"] is True
    assert summary["overall"]["total"] == 3
    assert len(server.requests) == 4


def test_resume_refuses_a_finished_run(workspace, server):
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")]) == EXIT_OK
    requests = len(server.requests)
    assert cli.main(["resume", "--run", "r"]) == EXIT_ERROR
    assert len(server.requests) == requests


def test_resume_refuses_a_changed_model(workspace, server, monkeypatch):
    server.queue((401, {"error": {"message": "bad"}}))
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")]) == EXIT_STOPPED
    monkeypatch.setenv("FLGROUND_MODEL", "accounts/acme/deployments/other")
    requests = len(server.requests)
    assert cli.main(["resume", "--run", "r"]) == EXIT_ERROR
    assert len(server.requests) == requests


def test_resume_allows_a_new_timeout(workspace, server):
    server.queue((401, {"error": {"message": "bad"}}))
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")]) == EXIT_STOPPED
    assert cli.main(["resume", "--run", "r", "--timeout", "600"]) == EXIT_OK


def test_resume_rejects_changed_settings_rather_than_ignoring_them(workspace, server):
    server.queue((401, {"error": {"message": "bad"}}))
    out = str(workspace / "outputs/r.jsonl")
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", out]) == EXIT_STOPPED
    requests = len(server.requests)
    assert run_command(["--out", out, "--resume", "--max-tokens", "9"]) == EXIT_ERROR
    assert len(server.requests) == requests


def test_missing_configuration_names_fields_and_makes_no_calls(workspace, server, monkeypatch, capsys):
    monkeypatch.delenv("FIREWORKS_API_KEY")
    monkeypatch.delenv("FLGROUND_MODEL")
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_ERROR
    error = capsys.readouterr().err
    assert "FIREWORKS_API_KEY" in error and "FLGROUND_MODEL" in error
    assert not server.requests


def test_undeployed_model_id_is_refused(workspace, server, monkeypatch):
    monkeypatch.setenv("FLGROUND_MODEL", "accounts/fireworks/models/qwen3p8-27b")
    assert cli.main(["run", "--mode", "simple_json", "--limit", "1"]) == EXIT_ERROR
    assert not server.requests


def test_check_never_prints_the_key(workspace, capsys):
    cli.main(["check"])
    output = capsys.readouterr().out
    assert API_KEY not in output
    assert "FIREWORKS_API_KEY: set" in output


def test_overlays_refuse_incomplete_runs_and_make_no_calls(workspace, server):
    server.queue((200, completion_body()), (401, {"error": {"message": "bad"}}))
    cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")])
    assert cli.main(["overlays", "--run", "r"]) == EXIT_ERROR
    requests = len(server.requests)
    assert cli.main(["overlays", "--run", "r", "--allow-partial"]) == EXIT_OK
    assert len(server.requests) == requests
    drawn = list(report.overlays_path(only_results(workspace)).glob("*.png"))
    assert [path.name for path in drawn] == ["eval_0.png"]
    with Image.open(drawn[0]) as image:
        assert image.size == (200, 134)
        # The left edge of the gold box, clear of the prediction cross.
        assert image.convert("RGB").getpixel((90, 76)) == (0, 230, 90)


def test_two_finished_runs_compare_through_the_report_entry(workspace, server, capsys):
    a, b = [str(workspace / f"outputs/{name}.jsonl") for name in ("a", "b")]
    assert run_command(["--out", a]) == EXIT_OK
    server.queue(*[(200, completion_body(content='{"x": 0.01, "y": 0.01}'))] * 3)
    assert run_command(["--out", b, "--temperature", "0.5"]) == EXIT_OK
    capsys.readouterr()
    assert cli.main(["report", a, b]) == EXIT_OK
    result = json.loads(capsys.readouterr().out)
    assert result["a_only"] == 3
    assert result["b_only"] == 0


def test_report_entry_verifies_a_single_run(workspace, server, capsys):
    out = str(workspace / "outputs/a.jsonl")
    assert run_command(["--out", out]) == EXIT_OK
    capsys.readouterr()
    assert cli.main(["report", out]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["finalized"] is True


def test_rescore_command_writes_a_separate_file_and_makes_no_calls(workspace, server, capsys):
    out = workspace / "outputs" / "r.jsonl"
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(out)]) == EXIT_OK
    requests = len(server.requests)
    capsys.readouterr()
    assert cli.main(["rescore", "--run", "r"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["num_correct_action"] == 3
    assert out.with_suffix(".paper_score.json").exists()
    assert len(server.requests) == requests


def test_cost_report_separates_lifetime_from_runtime(workspace, server, capsys):
    cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")])
    capsys.readouterr()
    assert cli.main(["cost", "--gpu-hourly-price", "2", "--gpus-per-replica", "2",
                     "--billable-hours", "3", "--run", "r"]) == EXIT_OK
    result = json.loads(capsys.readouterr().out)
    assert result["dedicated_deployment_gpu_estimate"]["estimated_usd"] == 12.0
    assert result["benchmark_runtime"]["active_inference_hours"] < 3
    assert "hypothetical_token_priced_cost" not in result


def test_cost_requires_all_token_rates(workspace, server):
    cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")])
    assert cli.main(["cost", "--gpu-hourly-price", "2", "--gpus-per-replica", "1",
                     "--billable-hours", "1", "--run", "r", "--input-rate", "1"]) == EXIT_ERROR


def test_token_cost_is_labeled_hypothetical(workspace, server, capsys):
    cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", str(workspace / "outputs/r.jsonl")])
    capsys.readouterr()
    cli.main(["cost", "--gpu-hourly-price", "2", "--gpus-per-replica", "1", "--billable-hours", "1",
              "--run", "r", "--input-rate", "1", "--output-rate", "2", "--cached-rate", "0.5"])
    result = json.loads(capsys.readouterr().out)
    assert "NOT this deployment's bill" in result["hypothetical_token_priced_cost"]["note"]
