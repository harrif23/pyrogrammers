"""The evals command: setup checks, standard runs, reports, overlays, and cost."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import evaluate
from .baseline import PROMPT_SHA256
from .config import BASE_URL, InferenceConfig, load_credentials, load_env_file
from .cost import benchmark_share, gpu_estimate, parse_segments, token_priced_estimate
from .dataset import DATA_DIR, load_examples, load_manifest, prepare
from .errors import EXIT_ERROR, EXIT_OK, ConfigError, evalsError, RunStateError, run_guarded
from .overlay import draw_run_overlays
from .report import (compare_runs, overlays_path, read_config, read_records,
                     rescore_with_paper_scorer, summarize, write_report)


def _run_path(value):
    """Accept a results path, or the name of a run under outputs/."""
    path = Path(value)
    return path if path.suffix == ".jsonl" else evaluate.OUTPUTS_DIR / f"{value}.jsonl"


def cmd_check(args):
    load_env_file()
    print(f"Base URL: {BASE_URL}")
    print(f".env file: {'present' if Path('.env').exists() else 'absent'}")
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], check=False).returncode == 0
    print(f".env ignored by git: {ignored}")
    code = EXIT_OK if ignored else EXIT_ERROR
    try:
        credentials = load_credentials()
        print("FIREWORKS_API_KEY: set")
        print(f"FLGROUND_MODEL: set, deployment-qualified ({credentials.model})")
    except ConfigError as error:
        print(f"Configuration incomplete: {error}")
        code = EXIT_ERROR
    try:
        manifest = load_manifest()
        for split in manifest["splits"]:
            count = len(load_examples(manifest, split))
            print(f"Dataset split {split}: {count} examples verified at {manifest['revision']}")
    except evalsError as error:
        print(f"Dataset not ready: {error}")
        code = EXIT_ERROR
    print(f"Baseline prompt SHA256: {PROMPT_SHA256}")
    print(f"Default inference settings: {InferenceConfig()}")
    return code


def cmd_prepare(args):
    manifest = prepare(revision=args.revision)
    evaluation = manifest["splits"]["eval"]
    print(f"Pinned {manifest['dataset']} at {manifest['revision']}")
    print(f"Eval targets: {evaluation['target_counts']}; "
          f"unique screenshots: {evaluation['unique_screenshot_count']}; "
          f"matches expected: {manifest['eval_matches_expected']}")
    print(f"Dev pool: {manifest['splits']['dev']['target_counts']} "
          f"from {manifest['splits']['dev']['annotation_path']}")
    print(f"Annotation SHA256: {evaluation['annotation_sha256']}")
    print(f"Manifest: {DATA_DIR / 'manifest.json'}")
    return EXIT_OK


def cmd_resume(args):
    return evaluate.resume_run(_run_path(args.run), args.timeout)[1]


def cmd_report(args):
    """One run writes or verifies its report; two runs are compared example by example."""
    runs = [_run_path(run) for run in args.runs]
    print(json.dumps(compare_runs(*runs) if len(runs) == 2 else write_report(runs[0]), indent=2))
    return EXIT_OK


def cmd_rescore(args):
    print(json.dumps(rescore_with_paper_scorer(_run_path(args.run)), indent=2))
    return EXIT_OK


def cmd_overlays(args):
    results = _run_path(args.run)
    stored = read_config(results)
    records = read_records(results)
    if not summarize(records, stored["example_ids"])["complete"] and not args.allow_partial:
        raise RunStateError("run is incomplete; finish it first or pass --allow-partial")
    examples = load_examples(load_manifest(), stored["split"])
    written = draw_run_overlays(records, examples, overlays_path(results))
    print(f"Wrote {len(written)} overlays to {overlays_path(results)}")
    return EXIT_OK


def cmd_cost(args):
    if (args.billable_hours is not None) == bool(args.replica_segments):
        raise ConfigError("pass exactly one of --billable-hours or --replica-segments")
    segments = (parse_segments(args.replica_segments) if args.replica_segments
                else [(args.billable_hours, args.replicas)])
    gpu = gpu_estimate(args.gpu_hourly_price, args.gpus_per_replica, segments)
    report = {
        "dedicated_deployment_gpu_estimate": {
            **gpu,
            "note": "Covers the whole billable lifetime you supplied, including idle time. "
                    "Your Fireworks invoice is authoritative.",
        },
    }
    rates = [args.input_rate, args.output_rate, args.cached_rate]
    if any(rate is not None for rate in rates) and any(rate is None for rate in rates):
        raise ConfigError("token pricing needs all of --input-rate, --output-rate, --cached-rate")
    if args.run:
        results = _run_path(args.run)
        summary = summarize(read_records(results), read_config(results)["example_ids"])
        report["benchmark_runtime"] = {
            **benchmark_share(gpu["billable_hours"], summary["active_inference_seconds"]),
            "note": "Active time is request latency for this run only, not the billed duration.",
        }
        if rates[0] is not None:
            report["hypothetical_token_priced_cost"] = {
                **token_priced_estimate(summary["tokens"], *rates),
                "note": "What-if at the rates you supplied. NOT this deployment's bill.",
            }
    elif rates[0] is not None:
        raise ConfigError("token pricing needs --run for its token counts")
    print(json.dumps(report, indent=2))
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", help="validate configuration; no network, no secrets printed"
                        ).set_defaults(func=cmd_check)

    prep = commands.add_parser("prepare", help="download and verify data; no paid inference")
    prep.add_argument("--revision", help="dataset commit; defaults to the manifest's pin")
    prep.set_defaults(func=cmd_prepare)

    run = commands.add_parser("run", help="score a strategy; each example is one or more model calls")
    evaluate.add_run_args(run)
    run.set_defaults(func=evaluate.run_from_args)

    resume = commands.add_parser("resume", help="continue a run; scored examples are never redone")
    resume.add_argument("--run", required=True, help="results path, or a run name under outputs/")
    resume.add_argument("--timeout", type=float, help="the one setting a resume may change")
    resume.set_defaults(func=cmd_resume)

    report = commands.add_parser("report", help="report one run, or compare two; no network")
    report.add_argument("runs", nargs="+", metavar="RUN", help="results path, or a name under evals/outputs")
    report.set_defaults(func=cmd_report)

    rescore = commands.add_parser("rescore", help="apply the paper's scorer to a run; no network")
    rescore.add_argument("--run", required=True)
    rescore.set_defaults(func=cmd_rescore)

    overlays = commands.add_parser("overlays", help="redraw predictions over gold boxes; no network")
    overlays.add_argument("--run", required=True)
    overlays.add_argument("--allow-partial", action="store_true")
    overlays.set_defaults(func=cmd_overlays)

    cost = commands.add_parser("cost", help="dedicated GPU estimate; optional token what-if")
    cost.add_argument("--gpu-hourly-price", type=float, required=True, help="USD per GPU-hour")
    cost.add_argument("--gpus-per-replica", type=int, required=True)
    cost.add_argument("--billable-hours", type=float, help="total deployment lifetime billed")
    cost.add_argument("--replicas", type=int, default=1, help="with --billable-hours")
    cost.add_argument("--replica-segments", help="hours:replicas,... when the count changed")
    cost.add_argument("--run", help="run to compare against")
    cost.add_argument("--input-rate", type=float, help="USD per million input tokens")
    cost.add_argument("--output-rate", type=float, help="USD per million output tokens")
    cost.add_argument("--cached-rate", type=float, help="USD per million cached input tokens")
    cost.set_defaults(func=cmd_cost)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "report" and len(args.runs) > 2:
        parser.error("pass one run to report, or two to compare")
    return run_guarded(args.func, args)


if __name__ == "__main__":
    sys.exit(main())
