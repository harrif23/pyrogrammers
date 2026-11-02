Experimental FL Studio harness

The strategies below are implemented but have not been validated on live inference or benchmark accuracy. Baseline and refine retain their one/two-call behavior. All strategies use the same configured deployment. API calls are billed; preparation, fixture generation and scoring are offline.

| Strategy | Calls | Behavior |
| --- | --- | --- |
| baseline | 1 | Existing full screenshot prediction |
| knowledge | 1 | Full screenshot plus concise manual-derived FL Studio guidance |
| refine | 2 | Existing fixed crop around initial point |
| panel_plain | 1–2 | Select panel/anchor, then localize in context-preserving crop, without manual rules |
| panel | 1–2 | Panel/anchor selection with manual rules, then localize |
| panel_verify | 1–3 | Panel pipeline plus comparison of initial/refined candidates |

Panel strategies stop early when the target is not identified. The demo then saves a report without displaying a marker or clicking. This is model-reported absence, not proof the control is absent. On positive benchmark examples it counts as a failed prediction. Invalid structured responses fail the run rather than silently guessing or falling back; this deliberately conservative implementation differs from the research plan's optional fallback. No hidden retry calls occur.

The region preserves both panel and visible label/header, with 5% padding (minimum 8 pixels). Crops are enlarged up to 2x with a 2048-pixel longest-side enlargement ceiling; native crops already larger than that are not shrunk. Verification gets full context and two separately marked copies of the crop. These constants are initial hypotheses, not tuned improvements. No self-reported confidence score is treated as a probability. The knowledge rules are versioned in knowledge.py and derived from Image-Line's Channel Rack and Mixer manuals; check applicability to benchmark version 20.8.3.

Try the new demo with an explicitly visible target:

```bash
uv run flground demo --strategy panel --instruction "Find the channel button labeled Drums" --delay 8 --overlay-seconds 20
```

Use `--strategy panel_verify` to add verification. Successful reports include raw responses, prompts, image dimensions/hashes, crop transforms, candidates, and usage. Failure records preserve completed and attempted model calls, raw received outputs and error categories. SDK exception strings are not copied into artifacts. Artifacts can contain sensitive screenshots/model responses and remain Git-ignored.

Coordinate calibration

The default remains `FLGROUND_COORDINATE_SPACE=pixels`. Supported alternatives are `normalized_1000` and `normalized_1`. This config controls both the requested units and explicit conversion back to pixels, including panel/anchor boxes. Units are NEVER guessed from response magnitudes. Do not switch just because a point happens to be below 1000. Confirm deployment documentation and independently test the convention first. No coordinate mode has been established as correct for this deployment by the new tests alone.

Create nine synthetic tasks across three image resolutions, independently of the benchmark:

```bash
uv run flground calibrate artifacts/calibration
# Billed: run only when ready to test the configured deployment.
uv run flground evaluate artifacts/calibration/tasks.jsonl artifacts/calibration-run --strategy baseline
uv run flground score artifacts/calibration-run/predictions.jsonl artifacts/calibration/labels.jsonl
```

For a separate explicitly configured experiment (not an automatic correction):

```bash
FLGROUND_COORDINATE_SPACE=normalized_1000 uv run flground evaluate artifacts/calibration/tasks.jsonl artifacts/calibration-1000 --strategy baseline
```

Review overlays and raw outputs as well as aggregate scores. These large colored targets check geometry/formatting, not FL Studio semantics. Freeze the convention before the benchmark and rerun the baseline after any adapter correction.

Benchmark preparation and split

Use the exact fruitloops_windows annotation file and matching original screenshots from the pinned dataset revision. The importer expects official fields id, img_filename, img_size, instruction, bbox (pixel xyxy), and ui_type. It validates image dimensions and boxes. It does not download or examine held-out failure results.

```bash
uv run flground prepare-benchmark /path/to/fruitloops_windows.json /path/to/images artifacts/fl-dataset --revision DATASET_COMMIT
```

This creates tasks.jsonl, labels.jsonl, dev.jsonl, heldout.jsonl, and dataset.json. Exact duplicate screenshot bytes stay in the same group; approximately 60% of groups go into dev, with seed 42 by default. Near-duplicate screenshots and shared sessions need manual grouping review before freezing. Stratification is not implemented: inspect the text/icon counts without examining target errors. The output records actual IDs; a split of 57 targets need not produce 34/23 examples.

Prefer external development tasks and all 57 official examples as the final test when available. If official failures have already influenced tuning, call that evaluation exploratory. Do not compare an internal subset percentage directly with the full leaderboard.

Inference input JSONL contains only id, image, instruction and group. Unknown fields (including bbox/correctness) are rejected. Image paths can be absolute or relative to the task file. Labels are loaded only by the separate offline scorer. Keep the label file out of any model context.

Run development comparisons:

```bash
uv run flground evaluate artifacts/fl-dataset/dev.jsonl artifacts/dev-baseline --strategy baseline
uv run flground evaluate artifacts/fl-dataset/dev.jsonl artifacts/dev-panel --strategy panel
uv run flground score artifacts/dev-panel/predictions.jsonl artifacts/fl-dataset/labels.jsonl --baseline artifacts/dev-baseline/predictions.jsonl
```

Use a new output directory for each run; existing runs are never overwritten. Each example is flushed to predictions.jsonl, including failures. Interrupted runs with missing IDs are rejected by the scorer when their manifest is present. Resume is not implemented. Run manifests save task/image/source hashes, source snapshots, model settings, code revision/dirty state, and the prepared dataset manifest when present. Prompts and image transforms are stored with individual calls. Server-side image preprocessing and the exact deployment revision still require external confirmation.

The scorer uses inclusive target-box boundaries while enforcing valid image coordinates, counts errors and abstentions as misses, and returns text/icon counts, mean/p95 latency, call counts, and usage. Paired comparison requires matching example IDs, instructions and image hashes; when both manifests exist, model/settings must also match. Missing manifests explicitly leave compatibility unverified. Cost remains null until a deployment billing basis is supplied; Fireworks on-demand GPU-time charges must not be replaced with invented per-token pricing. Confidence intervals and an external teammate-schema importer are not yet implemented.

Evaluate knowledge, refine and panel_plain as ablations on development data. Add panel_verify only if its development fixes exceed its regressions and justify extra calls. Freeze strategy/prompts/settings, then run baseline and chosen strategy on heldout.jsonl once. Do not tune on that result. The teammates' frontier baseline still needs its own matched manifest and outputs; this CLI does not claim leaderboard comparability automatically.
