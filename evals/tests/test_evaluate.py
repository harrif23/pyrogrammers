import json

import pytest

from conftest import API_KEY, completion_body
from evals.errors import RunStateError
from evals.evaluate import evaluate_examples
from evals.report import read_records

GOLD_HIT = '{"x": 0.5, "y": 0.5}'
GOLD_MISS = '{"x": 0.01, "y": 0.01}'


def test_record_contains_every_required_field(server, make_predictor, make_examples, meta, tmp_path):
    results = tmp_path / "results.jsonl"
    evaluate_examples(make_examples("icon"), make_predictor(), results, meta)
    record = read_records(results)[0]
    assert record["example_id"] == "ex_0"
    assert record["instruction"] == "click item 0"
    assert (record["image_width"], record["image_height"]) == (200, 100)
    assert record["mode"] == "baseline"
    assert record["model"] == meta["model"]
    assert record["config"] == meta["inference"]
    assert record["point_normalized"] == [0.5, 0.5]
    assert record["point_px"] == [100.0, 50.0]
    assert record["correct"] is True
    assert record["correctness"] == "correct"
    assert record["ui_type"] == "icon"
    assert record["latency_seconds"] >= 0
    call = record["calls"][0]
    assert call["content"] == GOLD_HIT
    assert call["finish_reason"] == "stop"
    assert call["response_id"] == "resp-1"
    assert call["usage"]["total_tokens"] == 120


def test_gold_data_never_reaches_the_model(server, make_predictor, make_examples, meta, tmp_path):
    evaluate_examples(make_examples("icon"), make_predictor(), tmp_path / "r.jsonl", meta)
    sent = server.requests[0].content.decode()
    assert "bbox" not in sent
    assert "ui_type" not in sent
    assert '"icon"' not in sent
    assert "90, 40, 110, 60" not in sent


def test_results_are_durable_after_each_example(server, make_predictor, make_examples, meta, tmp_path):
    results = tmp_path / "r.jsonl"
    seen = []
    evaluate_examples(make_examples("icon", "text"), make_predictor(), results, meta,
                      on_record=lambda record: seen.append(len(read_records(results))))
    assert seen == [1, 2]


def test_resume_skips_scored_examples_including_wrong_ones(server, make_predictor, make_examples,
                                                           meta, tmp_path):
    results = tmp_path / "r.jsonl"
    examples = make_examples("icon", "text")
    server.queue((200, completion_body(content=GOLD_MISS)),
                 (200, completion_body(content="garbage")))
    evaluate_examples(examples, make_predictor(), results, meta)
    calls = len(server.requests)

    outcome = evaluate_examples(examples, make_predictor(), results, meta)

    assert len(server.requests) == calls
    assert outcome.attempted == 0
    assert outcome.skipped == 2


def test_resume_retries_only_api_failures(server, make_predictor, make_examples, meta, tmp_path):
    results = tmp_path / "r.jsonl"
    examples = make_examples("icon", "text")
    server.queue((500, {"error": {"message": "boom"}}), (200, completion_body()))
    evaluate_examples(examples, make_predictor(), results, meta)
    assert [r["status"] for r in read_records(results)] == ["api_error", "scored"]

    outcome = evaluate_examples(examples, make_predictor(), results, meta)

    assert outcome.attempted == 1
    assert read_records(results)[-1]["example_id"] == "ex_0"


def test_api_failure_is_not_scored(server, make_predictor, make_examples, meta, tmp_path):
    results = tmp_path / "r.jsonl"
    server.queue((500, {"error": {"message": "boom"}}))
    evaluate_examples(make_examples("icon"), make_predictor(), results, meta)
    assert read_records(results)[0]["correct"] is None
    assert read_records(results)[0]["correctness"] is None


@pytest.mark.parametrize("content,correctness,correct", [
    (GOLD_HIT, "correct", True),
    (GOLD_MISS, "wrong", False),
    ('{"x": 183, "y": 40}', "wrong_format", False),
    ("garbage", "wrong_format", False),
])
def test_records_use_the_papers_three_outcomes(server, make_predictor, make_examples, meta, tmp_path,
                                               content, correctness, correct):
    results = tmp_path / "r.jsonl"
    server.queue((200, completion_body(content=content)))
    evaluate_examples(make_examples("icon"), make_predictor(), results, meta)
    record = read_records(results)[0]
    assert record["correctness"] == correctness
    assert record["correct"] is correct


@pytest.mark.parametrize("status", [401, 403, 404])
def test_systematic_failure_stops_immediately(server, make_predictor, make_examples, meta, tmp_path,
                                              status):
    server.queue((status, {"error": {"message": "no"}}))
    outcome = evaluate_examples(make_examples("icon", "text", "icon"), make_predictor(),
                                tmp_path / "r.jsonl", meta)
    assert "systematic" in outcome.stop_reason
    assert len(server.requests) == 1


def test_repeated_failures_stop_the_run(server, make_predictor, make_examples, meta, tmp_path):
    server.queue(*[(503, {"error": {"message": "down"}})] * 5)
    outcome = evaluate_examples(make_examples(*["icon"] * 5), make_predictor(),
                                tmp_path / "r.jsonl", meta)
    assert "3 consecutive" in outcome.stop_reason
    assert len(server.requests) == 3


def test_a_success_resets_the_consecutive_failure_count(server, make_predictor, make_examples, meta,
                                                        tmp_path):
    error = (503, {"error": {"message": "down"}})
    server.queue(error, error, (200, completion_body()), error, error, (200, completion_body()))
    outcome = evaluate_examples(make_examples(*["icon"] * 6), make_predictor(),
                                tmp_path / "r.jsonl", meta)
    assert outcome.stop_reason is None
    assert outcome.attempted == 6


def test_truncation_stops_when_requested_and_is_scored_invalid(server, make_predictor, make_examples,
                                                               meta, tmp_path):
    results = tmp_path / "r.jsonl"
    server.queue((200, completion_body(content='{"x": 0.', finish_reason="length")))
    outcome = evaluate_examples(make_examples("icon", "text"), make_predictor(), results, meta,
                                stop_on_truncation=True)
    assert "truncated" in outcome.stop_reason
    assert outcome.attempted == 1
    assert read_records(results)[0]["correct"] is False


def test_truncation_continues_when_not_requested(server, make_predictor, make_examples, meta, tmp_path):
    server.queue((200, completion_body(content='{"x": 0.', finish_reason="length")))
    outcome = evaluate_examples(make_examples("icon", "text"), make_predictor(),
                                tmp_path / "r.jsonl", meta)
    assert outcome.stop_reason is None
    assert outcome.attempted == 2


def test_torn_final_line_is_rerun(server, make_predictor, make_examples, meta, tmp_path):
    results = tmp_path / "r.jsonl"
    examples = make_examples("icon", "text")
    evaluate_examples(examples[:1], make_predictor(), results, meta)
    with open(results, "a") as handle:
        handle.write('{"example_id": "ex_1", "status": "sco')

    outcome = evaluate_examples(examples, make_predictor(), results, meta)

    assert outcome.attempted == 1
    assert [r["example_id"] for r in read_records(results)] == ["ex_0", "ex_1"]


def test_corruption_mid_log_refuses_to_continue(tmp_path):
    results = tmp_path / "r.jsonl"
    results.write_text('not json\n{"example_id": "a", "status": "scored"}\n')
    with pytest.raises(RunStateError):
        read_records(results)


def test_api_key_never_reaches_the_log(server, make_predictor, make_examples, meta, tmp_path):
    results = tmp_path / "r.jsonl"
    server.queue((400, {"error": {"message": f"bad key {API_KEY}"}}))
    evaluate_examples(make_examples("icon", "text"), make_predictor(), results, meta)
    assert API_KEY not in results.read_text()
    assert json.loads(results.read_text().splitlines()[0])["error"]["status"] == 400


def test_any_predictor_can_use_the_same_loop_and_scorer(make_examples, meta, tmp_path):
    class TwoCallPredictor:
        def predict(self, image, instruction):
            usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
            call = {"finish_reason": "stop", "content": "{}", "usage": usage}
            return {"point_px": [100.0, 50.0], "point_normalized": None, "calls": [call, call],
                    "latency_seconds": 3.0, "fallback_used": True, "invalid_reason": None,
                    "error": None}

    results = tmp_path / "r.jsonl"
    evaluate_examples(make_examples("icon"), TwoCallPredictor(), results, meta)
    record = read_records(results)[0]
    assert record["correct"] is True
    assert len(record["calls"]) == 2
    assert record["fallback_used"] is True
