# Benchmark optimization, 2026-11-02

## Evidence and limitations

Read-only snapshot of artifacts/qwen-before-after at 53 completed panel examples:
matched baseline 23/53, initial panel point 23/53, final panel point 23/53;
4 fixes and 4 regressions versus baseline. The complete baseline is 26/57.
This is provisional, not a completed panel result. Initial-point scoring is a
post-hoc ablation, not evidence that the new prompt will reproduce that score.
At the earlier 51-example snapshot, mean latency was 6.31s baseline versus 24.52s
panel; reasoning tokens were 58,968 versus 265,315. Local overhead was only
12.06s versus 14.27s in total. Multi-call reasoning dominates latency.

Three observed errors: null control rejected for an absent target; two coordinate
contract violations (pixel-looking values above 1000). Six abstentions: two at
planning, four after cropping. Do not reinterpret invalid coordinates or replace
abstentions with oracle-selected baseline predictions.

Development failures include wrong slot/control selection and ambiguous instructions;
some correct gains already occur in the first panel call. More cropping is not yet
supported as the remedy. Full-set aggregate results have been inspected; subsequent
full-set tuning/results are exploratory. No leaderboard victory is established.

## Implemented candidates

- `knowledge`: existing one-call grounding plus manual-derived domain guidance.
- `panel_fast`: one structured panel/anchor/point call, validated in original-image
  coordinates. Skips crop localization. Requests brief evidence and reiterates units.
- Absent plans may return null control; visible plans still require a control.
  This changes an error into an abstention, not a correct prediction.
- `--max-tokens` explicitly sets the same output ceiling for baseline and candidates.
  Try 4096 initially; if truncation is frequent, compare 8192. This caps output,
  including reasoning, and may reduce accuracy. It is not a guaranteed reasoning
  budget or a validated improvement. Length-terminated responses remain failures.

## Parallel evaluation

`--concurrency` accepts 1–20 (default 1). Each example has an isolated model client
and call log. Stages within an example remain sequential. The coordinator alone
writes/flushed JSONL records, in completion order; scoring joins by stable ID.
Artifact directories retain original task indices. The manifest records concurrency
and elapsed suite-arm wall time. Exceptions count as failed examples, including
client initialization errors. Retries remain disabled; 429s count as failures.
Concurrency reduces suite wall time, not tokens, and can increase per-call latency
or rate-limit errors. Actual deployment capacity for 15–20 calls is unverified.

After the existing run finishes, preview the next experiment:

```bash
uv run python scripts/benchmark.py run \
  --output artifacts/qwen-efficient-dev-4096 \
  --split dev --strategies baseline knowledge panel_fast \
  --max-tokens 4096 --concurrency 15
```

Append `--execute` to launch billed inference. Use a new output directory for each
experiment; existing runs are never overwritten. Set concurrency 20 only if the
15-worker run avoids rate limits. Compare all candidates to the newly run baseline
with identical budget/concurrency, not just the older 32768-token baseline.

Selection criteria: correct count first, fixes AND regressions, error/abstention and
length-termination counts, then completion/reasoning tokens, wall time and per-example
latency. Do not choose a strategy by valid-response rate. Pricing remains unknown;
report token usage without inventing a dollar estimate. Keep the old panel strategy
available as the two-call ablation. No paid follow-up experiment was launched here.

## Reuse an existing baseline without API calls

```bash
uv run python scripts/benchmark.py run \
  --output artifacts/qwen-harness-only-dev-4096 \
  --split dev --strategies knowledge panel_fast \
  --reuse-baseline artifacts/qwen-before-after/baseline/predictions.jsonl \
  --max-tokens 4096 --concurrency 15 --execute
```

Only candidate strategies run. The saved full baseline is validated and copied as a
subset matching dev IDs, instructions and image hashes. Original artifacts remain
unchanged. Different output budgets are explicitly flagged as an unmatched-budget
comparison; other model/settings differences are rejected before candidate calls.
The subset manifest retains the original model settings and source-file hash.
Use a fresh output path; this does not resume a partially completed candidate run.

## Next ablations after the 30/57 full-set result

The saved `qwen-panel-fast-full-8192` run scored 30/57 versus reused baseline
26/57: 9 fixes, 5 regressions. Text 22/26 versus 16/26; icons 8/31 versus 10/31.
All three errors were length termination at 8192 tokens. Full arm wall time was
75.24 seconds at concurrency 20. Output tokens increased from 63,619 to 130,983;
this is an accuracy gain, not demonstrated token efficiency. Settings differ.

Keep panel_fast unchanged. Three separate experimental strategies are now available:

- panel_recover: same panel_fast first call; one plain point-only recovery call only
  after explicit token exhaustion. Includes the failed call in usage and latency.
  No recovery for API errors, coordinate errors, or abstentions. Maximum two calls.
- panel_compact: replaces the larger plan schema with visible/panel/entity/point/
  ambiguous. No bounding boxes or explanatory evidence. One call.
- panel_selective: same compact first call, with an additional full-screen plus
  768-pixel context crop verification only when the model says ambiguous. The second
  call outputs original full-image coordinates. Maximum two calls. The ambiguity
  flag is an uncalibrated hypothesis, not a confidence guarantee.

All strategies retain invalid-output rejection and never consult benchmark labels,
example IDs, or saved baseline predictions during inference. No new live accuracy
results have been obtained for these strategies. Full-set comparisons are exploratory.

Run all 57 examples with the existing baseline (maximum 285 new calls):

```bash
uv run python scripts/benchmark.py run \
  --output artifacts/qwen-next-full-8192 \
  --split full --strategies panel_recover panel_compact panel_selective \
  --reuse-baseline artifacts/qwen-before-after/baseline/predictions.jsonl \
  --max-tokens 8192 --concurrency 20 --execute
```

The three strategies execute as separate arms. Compare their summaries with the
saved 30/57 panel_fast result as well as the reused 26/57 baseline. Inspect fixes,
regressions, truncations, calls and token usage, not just the best aggregate score.
