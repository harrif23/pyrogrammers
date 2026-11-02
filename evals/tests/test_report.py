import json
import os
import stat

import pytest

from evals.errors import RunStateError
from evals.report import (aggregate_tokens, compare_runs, config_path, load_finalized_report,
                             mcnemar_exact_p, percentile, report_path, rescore_with_paper_scorer,
                             summarize, token_counts, write_report)


def call(usage=None, finish_reason="stop"):
    return {"usage": usage, "finish_reason": finish_reason}


def scored(example_id, ui_type, correct, usage=None, latency=1.0, point=(5.0, 5.0),
           finish_reason="stop", calls=None, fallback=False):
    return {"example_id": example_id, "status": "scored", "ui_type": ui_type, "correct": correct,
            "calls": calls if calls is not None else [call(usage, finish_reason)],
            "latency_seconds": latency, "point_px": list(point) if point else None,
            "point_normalized": [point[0] / 10, point[1] / 10] if point else None,
            # Inside this box when correct, so paper_metrics agrees with the stored result.
            "gold_bbox": [0, 0, 10, 10] if correct else [9, 9, 10, 10],
            "image_width": 10, "image_height": 10, "fallback_used": fallback}


def failed(example_id, ui_type="icon", calls=None):
    return {"example_id": example_id, "status": "api_error", "ui_type": ui_type, "correct": None,
            "calls": calls or [], "latency_seconds": 2.0, "point_px": None, "fallback_used": False}


def test_accuracy_splits_by_ui_type():
    records = [scored("a", "icon", True), scored("b", "icon", False), scored("c", "text", True)]
    summary = summarize(records, ["a", "b", "c"])
    assert summary["overall"] == {"correct": 2, "total": 3, "accuracy": pytest.approx(2 / 3)}
    assert summary["icon"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert summary["text"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert summary["complete"] is True


def test_invalid_output_counts_against_accuracy():
    records = [scored("a", "icon", False, point=None), scored("b", "text", True)]
    summary = summarize(records, ["a", "b"])
    assert summary["invalid_output_count"] == 1
    assert summary["overall"]["accuracy"] == 0.5


def test_api_errors_are_separate_and_mark_the_run_incomplete():
    summary = summarize([failed("a"), scored("b", "text", True)], ["a", "b", "c"])
    assert summary["api_error_count"] == 1
    assert summary["overall"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert summary["missing"] == 1
    assert summary["complete"] is False


def test_a_successful_retry_supersedes_the_failed_attempt():
    summary = summarize([failed("a"), scored("a", "icon", True)], ["a"])
    assert summary["api_error_count"] == 0
    assert summary["api_error_attempts"] == 1
    assert summary["complete"] is True


def test_truncation_and_fallback_are_counted():
    summary = summarize([scored("a", "icon", False, point=None, finish_reason="length"),
                         scored("b", "text", True, fallback=True)], ["a", "b"])
    assert summary["truncated_count"] == 1
    assert summary["fallback_used_count"] == 1


def test_missing_usage_is_unknown_not_zero():
    tokens = aggregate_tokens([scored("a", "icon", True, usage=None)])
    assert tokens["total"] == {"sum": None, "known_calls": 0, "unknown_calls": 1}


USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "prompt_tokens_details": {"cached_tokens": 40},
         "completion_tokens_details": {"reasoning_tokens": 30}}


def test_reasoning_and_cached_are_not_added_to_totals():
    tokens = aggregate_tokens([scored("a", "icon", True, usage=USAGE),
                               scored("b", "icon", True, usage=USAGE)])
    assert tokens["completion"]["sum"] == 100
    assert tokens["total"]["sum"] == 300
    assert tokens["reasoning"]["sum"] == 60
    assert tokens["cached"]["sum"] == 80


def test_tokens_sum_across_every_call_of_a_multi_call_prediction():
    record = scored("a", "icon", True, calls=[call(USAGE), call(USAGE), call(USAGE)])
    assert aggregate_tokens([record])["total"]["sum"] == 450


def test_failed_attempts_keep_tokens_from_calls_that_succeeded():
    assert summarize([failed("a", calls=[call(USAGE)])], ["a"])["tokens"]["total"]["sum"] == 150


def test_partial_usage_keeps_known_values_and_counts_unknowns():
    usage = {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": 10}
    tokens = aggregate_tokens([scored("a", "icon", True, usage=usage)])
    assert tokens["prompt"]["sum"] == 10
    assert tokens["completion"] == {"sum": None, "known_calls": 0, "unknown_calls": 1}
    assert tokens["reasoning"]["unknown_calls"] == 1


def test_reasoning_falls_back_to_output_tokens_details():
    assert token_counts({"output_tokens_details": {"reasoning_tokens": 7}})["reasoning"] == 7


def test_zero_tokens_stay_known():
    tokens = aggregate_tokens([scored("a", "icon", True,
                                      usage={"prompt_tokens_details": {"cached_tokens": 0}})])
    assert tokens["cached"] == {"sum": 0, "known_calls": 1, "unknown_calls": 0}


def test_percentile_is_nearest_rank():
    assert percentile(list(range(1, 21)), 0.95) == 19
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([], 0.95) is None


def test_latency_uses_scored_examples_and_active_time_includes_failures():
    summary = summarize([failed("a"), scored("b", "text", True, latency=4.0),
                         scored("c", "text", True, latency=6.0)], ["a", "b", "c"])
    assert summary["latency_median_seconds"] == 5.0
    assert summary["active_inference_seconds"] == 12.0


def write_run(directory, name, records, ids=None, revision="rev", annotation="ann"):
    results = directory / f"{name}.jsonl"
    results.write_text("".join(json.dumps(record) + "\n" for record in records))
    ids = ids if ids is not None else [record["example_id"] for record in records]
    config_path(results).write_text(json.dumps({
        "mode": "baseline", "model": "m", "split": "dev", "dataset_revision": revision,
        "annotation_sha256": annotation, "fingerprint": name, "example_ids": ids}))
    return results


def test_finished_run_is_locked_with_a_hash(tmp_path):
    results = write_run(tmp_path, "a", [scored("x", "icon", True)])
    report = write_report(results)
    assert report["finalized"] is True
    assert len(report["results_sha256"]) == 64
    for path in (results, config_path(results), report_path(results)):
        assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_unfinished_run_stays_writable_and_unlocked(tmp_path):
    results = write_run(tmp_path, "a", [failed("x")])
    report = write_report(results)
    assert report["finalized"] is False
    assert report["results_sha256"] is None
    assert os.stat(results).st_mode & stat.S_IWUSR
    assert load_finalized_report(results) is None


def test_finalized_report_is_not_rewritten(tmp_path):
    results = write_run(tmp_path, "a", [scored("x", "icon", True)])
    first = report_path(results)
    write_report(results)
    before = first.read_text()
    write_report(results)
    assert first.read_text() == before


def test_changed_results_are_detected_after_locking(tmp_path):
    results = write_run(tmp_path, "a", [scored("x", "icon", True)])
    write_report(results)
    os.chmod(results, 0o644)
    results.write_text(json.dumps(scored("x", "icon", False)) + "\n")
    with pytest.raises(RunStateError, match="changed after"):
        write_report(results)


def test_report_includes_the_papers_metrics(tmp_path):
    results = write_run(tmp_path, "a", [scored("x", "icon", True), scored("y", "text", False),
                                        scored("z", "text", False, point=None)])
    paper = write_report(results)["paper_metrics"]
    assert paper["num_correct_action"] == 1
    assert paper["num_total"] == 3
    assert paper["wrong_format_num"] == 1
    assert paper["action_acc"] == pytest.approx(1 / 3)
    assert paper["icon_acc"] == 1.0
    assert paper["text_acc"] == 0
    assert paper["disagreements_with_stored_result"] == 0


def test_rescoring_a_locked_run_leaves_its_files_unchanged(tmp_path):
    results = write_run(tmp_path, "a", [scored("x", "icon", True)])
    write_report(results)
    before = {path: path.read_bytes() for path in
              (results, config_path(results), report_path(results))}
    scores = rescore_with_paper_scorer(results)
    assert scores["num_correct_action"] == 1
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads(results.with_suffix(".paper_score.json").read_text()) == scores


def test_rescoring_flags_a_stored_result_the_paper_scorer_disagrees_with(tmp_path):
    results = write_run(tmp_path, "a", [{**scored("x", "icon", True), "correct": False}])
    assert rescore_with_paper_scorer(results)["disagreements_with_stored_result"] == 1


def test_rescoring_refuses_tampered_results(tmp_path):
    results = write_run(tmp_path, "a", [scored("x", "icon", True)])
    write_report(results)
    os.chmod(results, 0o644)
    results.write_text(json.dumps(scored("x", "icon", False)) + "\n")
    with pytest.raises(RunStateError, match="changed after"):
        rescore_with_paper_scorer(results)


def test_api_errors_stay_out_of_the_papers_metrics(tmp_path):
    results = write_run(tmp_path, "a", [failed("x"), scored("y", "icon", True)])
    assert write_report(results)["paper_metrics"]["num_total"] == 1


@pytest.mark.parametrize("a_only,b_only,p", [
    (0, 0, 1.0),
    (0, 6, 2 * 1 / 64),
    (1, 7, 2 * 9 / 256),
    (5, 5, 1.0),
    (10, 0, 2 / 1024),
])
def test_mcnemar_exact_p(a_only, b_only, p):
    assert mcnemar_exact_p(a_only, b_only) == pytest.approx(p)


def _finished(tmp_path, name, outcomes, **kwargs):
    records = [scored(example_id, ui, correct, usage=USAGE)
               for example_id, ui, correct in outcomes]
    results = write_run(tmp_path, name, records, **kwargs)
    write_report(results)
    return results


def test_paired_comparison_counts_each_cell(tmp_path):
    a = _finished(tmp_path, "a", [("1", "icon", True), ("2", "icon", True),
                                  ("3", "text", False), ("4", "text", False)])
    b = _finished(tmp_path, "b", [("1", "icon", True), ("2", "icon", False),
                                  ("3", "text", True), ("4", "text", False)])
    result = compare_runs(a, b)
    assert (result["both_correct"], result["a_only"], result["b_only"], result["neither"]) == (1, 1, 1, 1)
    assert result["a_only_ids"] == ["2"]
    assert result["b_only_ids"] == ["3"]
    assert result["mcnemar_exact_p"] == 1.0
    assert result["accuracy_delta_b_minus_a"] == {"overall": 0.0, "icon": -0.5, "text": 0.5}
    assert result["a"]["tokens_total"] == 600


def test_comparison_refuses_unfinished_runs(tmp_path):
    a = _finished(tmp_path, "a", [("1", "icon", True)])
    b = write_run(tmp_path, "b", [failed("1")])
    write_report(b)
    with pytest.raises(RunStateError, match="finalized"):
        compare_runs(a, b)


def test_comparison_refuses_different_examples(tmp_path):
    a = _finished(tmp_path, "a", [("1", "icon", True)])
    b = _finished(tmp_path, "b", [("2", "icon", True)])
    with pytest.raises(RunStateError, match="different examples"):
        compare_runs(a, b)


def test_comparison_refuses_different_dataset_versions(tmp_path):
    a = _finished(tmp_path, "a", [("1", "icon", True)])
    b = _finished(tmp_path, "b", [("1", "icon", True)], revision="other")
    with pytest.raises(RunStateError, match="dataset versions"):
        compare_runs(a, b)
