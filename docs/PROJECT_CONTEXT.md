# FLGround project context and handoff

## Objective

For the Fireworks × Microsoft × NVIDIA hackathon, demonstrate that an inference-time
harness makes the same open vision model more reliable at locating FL Studio controls
without fine-tuning. Input: screenshot plus natural-language target instruction.
Output: click coordinates in the original screenshot, with diagnostic metadata.
This is grounding, not a full autonomous music-production agent.

Professional tools have dense controls, repeated visual patterns, and domain-specific
semantics. FL Studio is the first application; CAD, video editing, engineering, and
scientific software are potential future applications, not current deliverables.

## Ownership and decisions

- Teammates run the ScreenSpot-Pro FL Studio baseline. This repository builds the
  harness, local macOS demo, and eventual comparison tooling.
- Run locally on Mac with hosted Fireworks inference; no Docker or local GPU required.
- User-selected deployment: accounts/yuxinz721-ofi0hn0kj3/deployments/h0xbbw9t.
  Its underlying model identity and access must be confirmed with the team/account.
- API base: https://api.fireworks.ai/inference/v1. SDK calls chat/completions.
- Supplied settings: max_tokens 32768, top_k 40, presence_penalty 0,
  frequency_penalty 0; temperature omitted. Token ceiling is configurable.
- Credentials belong in local .env, loaded automatically; shell variables take priority.
- Prefer one evidence-supported improvement over a complicated pipeline.

## Benchmark and research references

- https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
- https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
- https://gui-agent.github.io/grounding-leaderboard/
- https://huggingface.co/Hcompany/Holo2-235B-A22B
- https://fireworks.ai/
- https://github.com/TopherMayor/open-computer-use

The brief reports approximately 57 fruitloops_windows examples, Holo2 single-step
56.1% (about 32/57), and agentic 66.7% (about 38/57). These are unverified references,
not results from this project. Verify current sources, subset membership, and scoring
before using them. Aggregate scores do not reveal individual Holo2 failures.
Point-in-target-box is the intended metric; match official boundary conventions.
The primary goal is improvement over our own matched-model baseline, not an unsupported
claim of beating Holo2. Windows benchmark performance and Mac live-demo behavior are
separate evidence.

## Implemented state

- FireworksModel encodes screenshots as PNG data URLs and requests strict JSON x/y.
- GroundingModel protocol allows replacement adapters.
- Pydantic validates finite coordinates; geometry rejects out-of-bounds predictions.
- Default baseline: one call, original screenshot, no preprocessing resize.
- Opt-in refine: initial prediction, clipped 512-pixel crop, 2x enlargement, second
  prediction, explicit mapping back. No confidence router or semantic verification.
- CLI commands: predict and demo. Successful runs save result.json, prediction.png,
  report.html, and refinement.png when applicable, under unique artifacts directories.
- Model-call responses, latency, and usage are recorded for successful calls.
- macOS capture uses mss; PyAutoGUI clicks map Retina pixels to desktop coordinates.
  Single display only; explicit click flag, confirmation, and exact stale-image check.
- Python dependencies are locked with uv.lock. Tests use fake/mock models.

Validation at this handoff: 12 unit tests passed in Linux and CLI help was exercised.
No live Fireworks prediction, real Mac capture/click, official benchmark run, or accuracy
improvement has been verified. No API credential was supplied to the development session.

## Known limitations

Malformed JSON, API errors, and invalid coordinates abort the CLI; failed calls are not
yet persisted in batch records. JSON fences are not stripped. A reasoning-heavy model
may require adapter-specific response handling after observing actual responses.
The 32768 token ceiling and 180-second timeout are configuration choices, not measured
latency/cost guarantees. Confidence and cost estimation are not implemented.
Exact screenshot equality before clicking can reject animation, focus changes, or
opening the report; there remains a capture-to-click race. Do not weaken this casually.
The saved baseline point is overlaid in cyan, final in red; on baseline runs red covers
cyan. Desktop reports do not establish that the application control actually responded.

## Baseline handoff contract (proposed, not an implemented importer)

Request a JSONL record per example with:

- example_id, instruction, image_path, image_width, image_height
- ground_truth_bbox plus its coordinate/boundary convention
- predicted_x, predicted_y, correct, status/error
- raw_response, latency_seconds, token_usage, estimated_cost if available
- target_type (text/icon), split, and development-only failure_category

Also request a run manifest with dataset revision, model/deployment identity, full
prompt, sampling settings, image preprocessing, code revision, split IDs, and cost
pricing/date. Do not place ground-truth boxes in inference inputs. Preserve the
teammates' original artifacts and adapt schemas explicitly rather than guessing.

## Evaluation plan and remaining priorities

1. Validate real deployment access and JSON behavior on five development screenshots.
2. Import teammates' baseline and generate visual development failure analysis.
3. Freeze a grouped development/held-out split before inspecting errors; group repeated
   screenshots and balance text/icon targets where feasible. Suggested ratio: 60/40.
   If all examples were already used for tuning, label results exploratory.
4. Classify errors: wrong panel, wrong track/channel, wrong control type, similar icon,
   coordinate imprecision, application semantics, or parsing/scaling infrastructure.
5. Choose the first optimization from the dominant development error: workspace
   selection, track-aware verification, crop refinement, or candidate comparison.
   Infrastructure fixes require rerunning the baseline, not claiming harness gains.
6. Add batch evaluation with failure persistence, point-in-box scoring, visual reports,
   accuracy counts, text/icon slices, fixes/regressions, latency, calls, usage and cost.
7. Freeze prompts and strategy; run held-out matched-model comparison and ablation.
   Show small-sample uncertainty. Do not tune on held-out results.
8. Validate Mac capture/Retina mapping and a snare-volume demo; optionally a Mixer or
   plugin task. Finish reproducible README commands and a concise measured-results pitch.

Judging weights supplied in the brief: benchmark 40%, business value 25%, technical
execution 20%, demo 10%, documentation 5%. Preserve evaluation time over demo polish.
Required end deliverables include a reproducible baseline, visual failure report,
specialized harness, ablation, desktop demo, and documented results. Several remain
outstanding; current foundation is not the completed benchmark project.
