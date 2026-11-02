FL Studio benchmark runbook — 2026-11-02

Primary experiment (user-confirmed): Qwen3.8-27B alone versus the SAME Qwen3.8-27B
with our panel harness. Both use FLGROUND_MODEL from .env (exported variables still
win). The suite does not select another model or run Holo2. Leaderboard scores below
are external context, not the baseline for measuring our contribution.

Official repository review: run_ss_pro.sh selects English, positive targets and the
original instruction style. eval_screenspot_pro.py scores inclusive pixel-xyxy target
boxes after normalization and reports text/icon slices. Our prepared tasks and scorer
follow those choices. The repository's Qwen adapters use a computer_use tool-call
prompt and local Transformers preprocessing; our Fireworks JSON prompts and hosted
preprocessing differ. Therefore this is a matched-backbone before/after experiment
on the official dataset, not an exact reproduction of an official Qwen adapter run.
Sources: [evaluation script](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/eval_screenspot_pro.py),
[launch configuration](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/run_ss_pro.sh),
and [Qwen adapter](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/models/qwen3_5.py).

The local dataset is prepared under artifacts/benchmark-data/prepared. The pinned
ScreenSpot-Pro revision is 210e78d3844251110bff86c95835ebd37a6930fa: 57 examples,
57 screenshot files, 31 icon and 26 text targets. The auto-generated exact-image
split contains 34 dev and 23 held-out examples. Near-duplicate/session grouping
still needs review before using these splits for tuning.

The official leaderboard JSON snapshot inspected on 2026-11-02 has Holo2-235B-A22B
(Agentic) at 38/57 = 66.6667% on fruitloops. Exceeding that reference requires at
least 39/57 = 68.4211%. This is not the overall ScreenSpot-Pro leaderboard score.
Sources: [leaderboard](https://gui-agent.github.io/grounding-leaderboard/) and
[score data](https://gui-agent.github.io/grounding-leaderboard/results/screenspot_pro.json).
The local raw snapshot is artifacts/leaderboard-20261102.json. Recheck before a
public claim; a higher local score does not automatically become an accepted leaderboard submission.

First experiment: freeze baseline versus panel on all 57

Use the full original screenshot and English instruction. Do not inspect benchmark
failure overlays or labels until both runs finish. The primary candidate is panel,
chosen before seeing these results. Baseline and panel use the same Qwen deployment,
normalized_1000 coordinates and explicit temperature=0 (greedy decoding). Other
request settings stay unchanged. The runner records settings and exact source
snapshots. Deployment identity/preprocessing and teammate comparability still need
confirmation; normalized_1000 has one successful real desktop example, not a full
calibration result.

From the repository root:

```bash
uv sync --extra desktop
# Already completed in this checkout; only needed on a fresh checkout:
uv run python scripts/benchmark.py prepare

# Free dry run: prints examples/settings/max calls, creates no run directory.
uv run python scripts/benchmark.py run --output artifacts/fl-first --split full

# Billed execution: 57 baseline + up to 114 panel calls = at most 171 calls.
uv run python scripts/benchmark.py run --output artifacts/fl-first --split full --execute
```

Skip prepare when the prepared folder already exists. Output directories must be
new; the runner refuses to overwrite previous runs. Change fl-first for a new run.
Credentials load from the existing .env without printing values. The suite explicitly
sets coordinate units and temperature in its process; it does not edit .env.

The command runs baseline, then panel, prints per-example progress and final counts,
and writes:

- plan.json: predeclared split, strategies and call ceiling.
- baseline/ and panel/: manifest, source snapshot, JSONL outcomes and per-example reports.
- summary.json: accuracy counts, errors, abstentions, slices, latency, usage, fixes/regressions.

Errors are failed predictions, not removed from the denominator. No automatic API
retries occur. If interrupted, the manifest exposes missing examples and scoring
refuses a partial denominator. Resume is not implemented: preserve partial artifacts
and choose a new output directory if restarting. The script does not currently stop
early on systemic API errors; stop it manually if repeated failures appear.

Calibration before spending on 171 calls

To independently check mapping across resolutions without exposing benchmark targets:

```bash
uv run flground calibrate artifacts/fl-calibration
FLGROUND_COORDINATE_SPACE=normalized_1000 FLGROUND_TEMPERATURE=0 uv run flground evaluate artifacts/fl-calibration/tasks.jsonl artifacts/fl-calibration-run --strategy baseline
uv run flground score artifacts/fl-calibration-run/predictions.jsonl artifacts/fl-calibration/labels.jsonl
```

Fixture generation is offline. Evaluation is nine billed requests. Inspect these
synthetic results; resolve geometry/format issues before freezing and starting the
full baseline. Colored-square success does not demonstrate FL Studio understanding.

Interpretation and next experiments

Compare panel against baseline first. At least 39 correct out of the full 57 exceeds
the recorded 38/57 reference numerically. To claim a credible harness improvement,
report paired fixes and regressions, model/settings, all failures and additional
cost, not just a higher final percentage. A one-example win is weak evidence given
sampling and dataset size. No new gain is known before the experiment runs.

If full-set failures guide further changes, subsequent full-set runs are exploratory;
the auto-generated 23-case heldout is no longer untouched after inspecting the full
57. Obtain separate unseen examples for a final generalization claim. Alternatively,
choose the dev/heldout route below INSTEAD OF the full first experiment, keeping
held-out failures sealed; its heldout percentage cannot be compared directly to a
57-example leaderboard percentage.

```bash
# Alternative development route: at most 34*(1+1+2+2+2+3) = 374 calls.
uv run python scripts/benchmark.py run --split dev --output artifacts/fl-dev --strategies baseline knowledge refine panel_plain panel panel_verify
# Add --execute only when ready to run this declared sweep.

# Freeze one winner using development evidence; this example assumes panel won.
# At most 23*(1+2) = 69 calls. Do not change the winner after inspecting this result.
uv run python scripts/benchmark.py run --split heldout --output artifacts/fl-heldout --strategies baseline panel --execute
```

The default first comparison intentionally omits verification to limit cost and avoid
choosing a winner on the test set. A three-call verification variant needs a separate
predeclared experiment. The runner always requires baseline first.

Cost and time

The 171-call ceiling is an API-request bound, not a dollar quote. The deployment is
on-demand; use actual GPU-hour billing and allocation/idle-time assumptions. The
runner leaves estimated_cost null rather than inventing prices. At an assumed
10–20 seconds per call, 171 sequential calls would take approximately 29–57 minutes;
this is planning arithmetic, not measured benchmark latency. High-resolution inputs,
reasoning length and failures can change this substantially.

No paid benchmark or calibration inference was launched while preparing this runbook.
The downloader and dry-run workflow were exercised, and 68 mocked/offline tests passed.
