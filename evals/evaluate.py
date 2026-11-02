"""Dataset loop, gold-only scoring, and result checkpointing for any predictor."""

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import baseline, reference_qwen
from .baseline import BaselinePredictor
from .client import FireworksClient
from .config import (BASE_URL, EVALS_DIR, InferenceConfig, build_inference_config,
                     load_credentials, load_env_file, run_fingerprint)
from .dataset import load_examples, load_manifest
from .errors import (EXIT_OK, EXIT_STOPPED, EXIT_TRUNCATED, ConfigError, RunStateError,
                     run_guarded)
from .overlay import draw_run_overlays
from .paper_scorer import score_prediction
from .report import (config_path, load_finalized_report, overlays_path, read_config,
                     read_records, report_path, token_counts, write_report)

OUTPUTS_DIR = EVALS_DIR / "outputs"
MODES = ("baseline", "reference")
_CUSTOM_PREFIX = "custom:"
_PREDICTION_FIELDS = {"point_px", "point_normalized", "calls", "latency_seconds", "fallback_used",
                      "invalid_reason", "error"}
PROMPT_SHA256 = {"baseline": baseline.PROMPT_SHA256, "reference": reference_qwen.PROMPT_SHA256}
MAX_CONSECUTIVE_API_ERRORS = 3
INFERENCE_DEFAULTS = {"reasoning_effort": "low", "max_tokens": 4096, "temperature": 0.0,
                      "response_format": "none", "timeout": 180.0}


@dataclass(frozen=True)
class RunOutcome:
    attempted: int
    skipped: int
    stop_reason: str | None


def _load_custom(spec):
    """Import a module:attribute spec. Raises ConfigError with what to fix."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not attribute:
        raise ConfigError(f"--predictor must look like module:factory, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
        return module, getattr(module, attribute)
    except (ImportError, AttributeError) as error:
        raise ConfigError(f"cannot load predictor {spec!r}: {error}") from None


def build_predictor(mode, credentials, inference):
    client = FireworksClient(credentials, inference)
    if mode == "baseline":
        return BaselinePredictor(client)
    if mode == "reference":
        return reference_qwen.ReferenceQwenPredictor(client)
    if mode.startswith(_CUSTOM_PREFIX):
        return _load_custom(mode.removeprefix(_CUSTOM_PREFIX))[1](client)
    raise ConfigError(f"unknown mode {mode!r}")


def prompt_sha256(mode):
    """What identifies a strategy's behavior: its prompt, or a custom harness's module source.

    why: a harness edit must block resume, or one run would mix two versions of the harness.
    It hashes only the named module, so edits in modules it imports aren't detected.
    """
    if mode.startswith(_CUSTOM_PREFIX):
        module, _ = _load_custom(mode.removeprefix(_CUSTOM_PREFIX))
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return PROMPT_SHA256[mode]


def _checked(prediction):
    """Refuse a malformed harness result now, rather than crash later in the report."""
    missing = _PREDICTION_FIELDS - set(prediction)
    if missing:
        raise RunStateError(f"predictor result is missing {sorted(missing)}; use "
                            "evals.baseline.make_prediction")
    return prediction


def completed_ids(records):
    """Scored examples are final, right or wrong; only API failures remain to be retried."""
    return {record["example_id"] for record in records if record["status"] == "scored"}


def _drop_torn_tail(results):
    path = Path(results)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def _append(results, record):
    with open(results, "a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record(example, meta, prediction):
    failed = prediction["error"] is not None
    # Gold labels are first used here, after prediction has returned.
    correctness = None if failed else score_prediction(
        prediction["point_normalized"], prediction["point_px"], example.bbox, example.width,
        example.height)
    return {
        "example_id": example.id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instruction": example.instruction,
        "image_file": example.image_path.name,
        "image_width": example.width,
        "image_height": example.height,
        "ui_type": example.ui_type,
        "gold_bbox": list(example.bbox),
        "mode": meta["mode"],
        "model": meta["model"],
        "config": meta["inference"],
        "config_fingerprint": meta["fingerprint"],
        "status": "api_error" if failed else "scored",
        **prediction,
        "correctness": correctness,
        "correct": None if failed else correctness == "correct",
    }


def evaluate_examples(examples, predictor, results, meta, stop_on_truncation=False,
                      max_consecutive_api_errors=MAX_CONSECUTIVE_API_ERRORS, on_record=None):
    """Predict every example not yet scored, appending one durable record per attempt.

    rules:
      - predictors receive only the image and instruction.
      - invalid output is a scored, incorrect example and is not retried on resume.
      - API failures are logged, not scored, and are retried on resume.
      - stop at the first systematic failure or after repeated consecutive failures.
    why: results are fsynced per example because each call costs GPU time; a crash must
    lose at most the example in flight.
    side effects: appends to results; model calls for each pending example.
    """
    _drop_torn_tail(results)
    done = completed_ids(read_records(results))
    pending = [example for example in examples if example.id not in done]
    consecutive_errors = 0
    stop_reason = None
    attempted = 0

    for example in pending:
        prediction = _checked(predictor.predict(example.image_path, example.instruction))
        error = prediction["error"]
        if error:
            consecutive_errors += 1
            if error["systematic"]:
                stop_reason = f"systematic API failure: {error['kind']}"
            elif consecutive_errors >= max_consecutive_api_errors:
                stop_reason = f"{consecutive_errors} consecutive API failures: {error['kind']}"
        else:
            consecutive_errors = 0
            if stop_on_truncation and any(c["finish_reason"] == "length"
                                          for c in prediction["calls"]):
                stop_reason = "output truncated (finish_reason=length)"
        record = _record(example, meta, prediction)
        _append(results, record)
        attempted += 1
        if on_record:
            on_record(record)
        if stop_reason:
            break

    return RunOutcome(attempted, len(examples) - len(pending), stop_reason)


def _print_record(record):
    if record["status"] == "api_error":
        error = record["error"]
        print(f"{record['example_id']}: API ERROR {error['kind']} status={error['status']} "
              f"{error['message']}")
        return
    # A harness may answer without recording a call, so there may be nothing further to show.
    call = record["calls"][-1] if record["calls"] else {}
    counts = token_counts(call.get("usage"))
    print(f"{record['example_id']} [{record['ui_type']}] correct={record['correct']} "
          f"finish_reason={call.get('finish_reason')} latency={record['latency_seconds']:.2f}s "
          f"prompt={counts['prompt']} completion={counts['completion']} "
          f"reasoning={counts['reasoning']} invalid={record['invalid_reason']}")
    if call:
        print(f"  answer: {str(call.get('content'))[:300]!r}")


def _execute(results, stored, examples, credentials, inference, stop_on_truncation):
    predictor = build_predictor(stored["mode"], credentials, inference)
    meta = {"mode": stored["mode"], "model": credentials.model, "inference": stored["inference"],
            "fingerprint": stored["fingerprint"]}
    outcome = evaluate_examples(examples, predictor, results, meta,
                                stop_on_truncation=stop_on_truncation, on_record=_print_record)
    report = write_report(results)
    print(json.dumps(report, indent=2))
    if report["complete"]:
        draw_run_overlays(read_records(results), examples, overlays_path(results))
    print(f"Results: {results}")
    print(f"Report: {report_path(results)}")
    if outcome.stop_reason:
        print(f"STOPPED: {outcome.stop_reason}", file=sys.stderr)
        code = EXIT_TRUNCATED if "truncated" in outcome.stop_reason else EXIT_STOPPED
        return report, code
    if not report["complete"]:
        # why: a lone 503 while the deployment scales up stops nothing, but must not read as success.
        print(f"INCOMPLETE: {report['api_error_count']} examples failed at the API; "
              "continue with --resume", file=sys.stderr)
        return report, EXIT_STOPPED
    return report, EXIT_OK


def start_run(mode, split, limit, out, inference, stop_on_truncation, label=None):
    """Start a new run; existing results are never overwritten.

    raises: RunStateError if the output already exists; ConfigError for missing credentials
    or a predictor that can't be loaded.
    """
    load_env_file()
    credentials = load_credentials()
    build_predictor(mode, credentials, inference)
    manifest = load_manifest()
    examples = load_examples(manifest, split)[:limit]
    entry = manifest["splits"][split]
    ids = [example.id for example in examples]
    prompt_hash = prompt_sha256(mode)
    fingerprint = run_fingerprint(mode, credentials.model, inference, prompt_hash,
                                  manifest["revision"], entry["annotation_sha256"], split, ids)
    file_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", mode)
    results = Path(out) if out else OUTPUTS_DIR / f"{file_stem}-{split}-{fingerprint[:8]}.jsonl"
    if results.exists() or config_path(results).exists():
        raise RunStateError(f"{results} already exists; continue it with --resume, or choose "
                            "another output")
    results.parent.mkdir(parents=True, exist_ok=True)
    stored = {
        "label": label,
        "mode": mode,
        "split": split,
        "stop_on_truncation": stop_on_truncation,
        "fingerprint": fingerprint,
        "base_url": BASE_URL,
        "model": credentials.model,
        "inference": inference.scored_fields(),
        "timeout_s": inference.timeout_s,
        "prompt_sha256": prompt_hash,
        "dataset_revision": manifest["revision"],
        "annotation_sha256": entry["annotation_sha256"],
        "example_ids": ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path(results).write_text(json.dumps(stored, indent=2) + "\n")
    results.touch()
    return _execute(results, stored, examples, credentials, inference, stop_on_truncation)


def resume_run(results, timeout=None):
    """Continue a run after proving nothing that affects answers has changed.

    raises: RunStateError if the run is finalized or its configuration changed.
    """
    load_env_file()
    credentials = load_credentials()
    stored = read_config(results)
    if load_finalized_report(results):
        raise RunStateError(f"{results} is finished and locked; start a new run instead")
    manifest = load_manifest()
    inference = InferenceConfig(**stored["inference"], timeout_s=timeout or stored["timeout_s"])
    entry = manifest["splits"][stored["split"]]
    fingerprint = run_fingerprint(stored["mode"], credentials.model, inference,
                                  prompt_sha256(stored["mode"]),
                                  manifest["revision"], entry["annotation_sha256"],
                                  stored["split"], stored["example_ids"])
    if fingerprint != stored["fingerprint"]:
        raise RunStateError("configuration changed since this run started (model, prompt, harness "
                            "or dataset); start a new run instead of mixing results")
    by_id = {example.id: example for example in load_examples(manifest, stored["split"])}
    examples = [by_id[example_id] for example_id in stored["example_ids"]]
    return _execute(results, stored, examples, credentials, inference,
                    stored["stop_on_truncation"])


def add_inference_args(parser):
    parser.add_argument("--reasoning-effort",
                        help="none|low|medium|high|xhigh|max, a positive integer cap, or omit "
                             f"(default {INFERENCE_DEFAULTS['reasoning_effort']})")
    parser.add_argument("--max-tokens", type=int,
                        help=f"default {INFERENCE_DEFAULTS['max_tokens']}")
    parser.add_argument("--temperature", type=float,
                        help=f"default {INFERENCE_DEFAULTS['temperature']}")
    parser.add_argument("--response-format", choices=["none", "json_object"],
                        help=f"default {INFERENCE_DEFAULTS['response_format']}")
    parser.add_argument("--timeout", type=float,
                        help=f"seconds per request; default {INFERENCE_DEFAULTS['timeout']}")


def _setting(args, name):
    value = getattr(args, name)
    return INFERENCE_DEFAULTS[name] if value is None else value


def inference_from_args(args):
    return build_inference_config(*(_setting(args, name) for name in (
        "reasoning_effort", "max_tokens", "temperature", "response_format", "timeout")))


def add_run_args(parser):
    parser.add_argument("--mode", choices=MODES, default="baseline",
                        help="built-in strategy; ignored with --predictor")
    parser.add_argument("--predictor", metavar="MODULE:FACTORY",
                        help="your harness: FACTORY(client) returns an object with "
                             "predict(image_path, instruction)")
    parser.add_argument("--out", help="results file; defaults to a new file under evals/outputs")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev",
                        help="dev is for building and tuning; eval is the final FL Studio set")
    parser.add_argument("--limit", type=int, help="run only the first N examples")
    parser.add_argument("--resume", action="store_true", help="continue an unfinished --out")
    parser.add_argument("--label", help="how this result must be described; kept in its report")
    add_inference_args(parser)


def run_from_args(args):
    if args.resume:
        if not args.out:
            raise RunStateError("--resume needs --out, the run to continue")
        locked = [name for name in ("reasoning_effort", "max_tokens", "temperature",
                                    "response_format", "predictor", "limit", "label")
                  if getattr(args, name) is not None]
        if locked:
            raise RunStateError(f"--resume keeps the run's settings; remove {', '.join(locked)}")
        return resume_run(Path(args.out), args.timeout)[1]
    if args.limit is not None and args.limit <= 0:
        raise ConfigError("--limit must be positive")
    mode = f"{_CUSTOM_PREFIX}{args.predictor}" if args.predictor else args.mode
    return start_run(mode, args.split, args.limit, args.out, inference_from_args(args),
                     args.split == "dev", label=args.label)[1]
