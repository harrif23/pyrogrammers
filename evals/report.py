"""Aggregate one run, lock finished runs, and compare two runs example by example."""

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from .errors import RunStateError
from .paper_scorer import calc_metric_for_result_list, score_prediction

TOKEN_FIELDS = ("prompt", "completion", "total", "cached", "reasoning")
_READ_ONLY = 0o444


def config_path(results):
    return Path(results).with_suffix(".config.json")


def report_path(results):
    return Path(results).with_suffix(".report.json")


def overlays_path(results):
    return Path(results).with_suffix(".overlays")


def read_records(results):
    """Read a results file, dropping a final line torn by a crash.

    why: a torn line was never a completed example, so it is re-run. Corruption anywhere
    else means the log cannot be trusted, and raises RunStateError.
    """
    path = Path(results)
    if not path.exists():
        return []
    lines = path.read_text().split("\n")
    records = []
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if index == len(lines) - 1:
                continue
            raise RunStateError(f"{path} line {index + 1} is corrupt") from None
    return records


def read_config(results):
    path = config_path(results)
    if not path.exists():
        raise RunStateError(f"no run at {results}: {path.name} is missing")
    return json.loads(path.read_text())


def token_counts(usage):
    """Pull token counts from a raw usage object; a missing value is None, never zero."""
    usage = usage or {}
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return {
        "prompt": usage.get("prompt_tokens"),
        "completion": usage.get("completion_tokens"),
        "total": usage.get("total_tokens"),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning": details.get("reasoning_tokens"),
    }


def aggregate_tokens(records):
    """Sum per field over every model call, reporting how many calls could not contribute.

    why: reasoning is a breakdown of completion and cached a breakdown of prompt, so they
    are reported alongside and never added into the totals.
    """
    totals = {field: {"sum": None, "known_calls": 0, "unknown_calls": 0}
              for field in TOKEN_FIELDS}
    for record in records:
        for call in record.get("calls", []):
            counts = token_counts(call.get("usage"))
            for field in TOKEN_FIELDS:
                entry = totals[field]
                if counts[field] is None:
                    entry["unknown_calls"] += 1
                else:
                    entry["sum"] = (entry["sum"] or 0) + counts[field]
                    entry["known_calls"] += 1
    return totals


def percentile(values, fraction):
    """Nearest-rank percentile: always a latency that was actually observed."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(fraction * len(ordered)) - 1, 0)]


def _accuracy(records):
    correct = sum(1 for record in records if record["correct"])
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else None,
    }


def latest_by_example(records):
    latest = {}
    for record in records:
        latest[record["example_id"]] = record
    return latest


def summarize(records, expected_ids):
    """Summarize a run. The latest attempt per example decides its state.

    rules: invalid outputs count as incorrect. API errors stay out of accuracy and are
    reported separately; the run is incomplete until every example is scored. Tokens and
    active time include every attempt, because every attempt was served.
    """
    latest = latest_by_example(records)
    scored = [record for record in latest.values() if record["status"] == "scored"]
    unresolved_errors = [record for record in latest.values() if record["status"] == "api_error"]
    missing = [example_id for example_id in expected_ids if example_id not in latest]
    latencies = [record["latency_seconds"] for record in scored]
    return {
        "complete": not missing and not unresolved_errors,
        "expected_targets": len(expected_ids),
        "scored": len(scored),
        "missing": len(missing),
        "overall": _accuracy(scored),
        "icon": _accuracy([r for r in scored if r["ui_type"] == "icon"]),
        "text": _accuracy([r for r in scored if r["ui_type"] == "text"]),
        "invalid_output_count": sum(1 for r in scored if r["point_px"] is None),
        "truncated_count": sum(1 for r in scored
                               if any(c["finish_reason"] == "length" for c in r["calls"])),
        "fallback_used_count": sum(1 for r in scored if r["fallback_used"]),
        "api_error_count": len(unresolved_errors),
        "api_error_attempts": sum(1 for r in records if r["status"] == "api_error"),
        "tokens": aggregate_tokens(records),
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_p95_seconds": percentile(latencies, 0.95),
        "active_inference_seconds": sum(r["latency_seconds"] for r in records),
    }


def paper_metrics(records):
    """The paper's metrics, recomputed from stored points and gold boxes for each scored example."""
    results = [
        {"correctness": score_prediction(record["point_normalized"], record["point_px"],
                                         record["gold_bbox"], record["image_width"],
                                         record["image_height"]),
         "ui_type": record["ui_type"], "correct": record["correct"]}
        for record in latest_by_example(records).values() if record["status"] == "scored"
    ]
    return {
        **calc_metric_for_result_list(results),
        "disagreements_with_stored_result": sum(
            1 for result in results if (result["correctness"] == "correct") != result["correct"]),
    }


def rescore_with_paper_scorer(results):
    """Write <name>.paper_score.json for an existing run, leaving the run's files untouched.

    why: locked runs from before the paper scorer stay valid and unchanged, so the published
    metrics are added beside them rather than rewritten into them.
    raises: RunStateError if a locked run's results changed after it was finalized.
    """
    load_finalized_report(results)
    scores = paper_metrics(read_records(results))
    Path(results).with_suffix(".paper_score.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_finalized_report(results):
    """The locked report for a run, or None if the run isn't finalized.

    raises: RunStateError if the results no longer match the hash recorded when locked.
    """
    path = report_path(results)
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if not report.get("finalized"):
        return None
    if _sha256(results) != report["results_sha256"]:
        raise RunStateError(f"{results} changed after it was finalized; its report is invalid")
    return report


def write_report(results):
    """Write a run's report, locking the run once every example is scored.

    rules: a finalized run's report is verified and returned, never rewritten.
    side effects: writes <name>.report.json; on completion marks results, config and report
    read-only.
    """
    finalized = load_finalized_report(results)
    if finalized:
        return finalized
    config = read_config(results)
    records = read_records(results)
    summary = summarize(records, config["example_ids"])
    report = {
        "label": config.get("label"),
        "mode": config["mode"],
        "model": config["model"],
        "split": config["split"],
        "dataset_revision": config["dataset_revision"],
        "config_fingerprint": config["fingerprint"],
        **summary,
        "paper_metrics": paper_metrics(records),
        "finalized": summary["complete"],
        "results_sha256": _sha256(results) if summary["complete"] else None,
    }
    report_path(results).write_text(json.dumps(report, indent=2) + "\n")
    if summary["complete"]:
        for path in (results, config_path(results), report_path(results)):
            os.chmod(path, _READ_ONLY)
    return report


def mcnemar_exact_p(a_only, b_only):
    """Two-sided exact McNemar test on the examples where the two runs disagree."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def compare_runs(results_a, results_b):
    """Compare two finalized runs on the same examples.

    raises: RunStateError unless both are finalized on identical examples and data, since
    any other comparison would not be paired.
    """
    report_a = load_finalized_report(results_a)
    report_b = load_finalized_report(results_b)
    if not report_a or not report_b:
        raise RunStateError("both runs must be finished and finalized before comparing")
    config_a, config_b = read_config(results_a), read_config(results_b)
    if config_a["example_ids"] != config_b["example_ids"]:
        raise RunStateError("runs cover different examples, so they cannot be paired")
    if (config_a["dataset_revision"], config_a["annotation_sha256"]) != (
            config_b["dataset_revision"], config_b["annotation_sha256"]):
        raise RunStateError("runs used different dataset versions, so they cannot be paired")

    latest_a = latest_by_example(read_records(results_a))
    latest_b = latest_by_example(read_records(results_b))
    counts = {"both_correct": 0, "a_only": 0, "b_only": 0, "neither": 0}
    a_only_ids, b_only_ids = [], []
    for example_id in config_a["example_ids"]:
        a_correct, b_correct = latest_a[example_id]["correct"], latest_b[example_id]["correct"]
        if a_correct and b_correct:
            counts["both_correct"] += 1
        elif a_correct:
            counts["a_only"] += 1
            a_only_ids.append(example_id)
        elif b_correct:
            counts["b_only"] += 1
            b_only_ids.append(example_id)
        else:
            counts["neither"] += 1

    def delta(group):
        a, b = report_a[group]["accuracy"], report_b[group]["accuracy"]
        return b - a if a is not None and b is not None else None

    def side(report):
        return {
            "mode": report["mode"],
            "overall": report["overall"], "icon": report["icon"], "text": report["text"],
            "tokens_total": report["tokens"]["total"]["sum"],
            "latency_median_seconds": report["latency_median_seconds"],
            "latency_p95_seconds": report["latency_p95_seconds"],
        }

    return {
        "examples": len(config_a["example_ids"]),
        **counts,
        "a_only_ids": a_only_ids,
        "b_only_ids": b_only_ids,
        "mcnemar_exact_p": mcnemar_exact_p(counts["a_only"], counts["b_only"]),
        "accuracy_delta_b_minus_a": {group: delta(group) for group in ("overall", "icon", "text")},
        "a": side(report_a),
        "b": side(report_b),
    }
