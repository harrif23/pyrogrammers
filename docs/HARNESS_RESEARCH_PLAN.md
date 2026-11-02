FLGround research and experiment plan — 2026-11-02

Recommendation: test FL Studio-specific semantic guidance plus panel-first localization, then add candidate verification only if it improves development results. Keep Qwen3.8-27B fixed across comparisons. This is a proposed experiment, not an implemented or validated gain.

The user identified the deployment backbone as Qwen3.8-27B. [Fireworks lists that model](https://fireworks.ai/models/fireworks/qwen3p8-27b) with image input and on-demand deployment. Record the actual deployed revision, quantization, image processing, and settings before treating another run as comparable. Deadline, budget, and teammate baseline availability remain unknown. No billed experiment was run for this research.

1. Establish what must be beaten

The [dataset card](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) lists FL Studio 20.8.3 on Windows, with 31 icon and 26 text targets (57 total). The Mac demo is a separate usability test. The linked leaderboard's query parameters do not establish a dataset split or inference configuration.

The [official evaluator](https://raw.githubusercontent.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/main/eval_screenspot_pro.py) uses x1/y1/x2/y2 boxes and inclusive point-in-box boundaries. Match its instruction language/style and positive-target protocol explicitly. Preserve every example, counting API, parsing, invalid-coordinate errors and positive-target abstentions as failures. Model inputs receive only image and instruction; the scorer alone reads labels.

Use three comparisons: the teammates' frontier baseline required by the supplied hackathon slides; correctly configured Qwen3.8-27B single pass; and the same Qwen model with our harness. A published Holo2 number is context, not a substitute for a matched run. Obtain the team's model, full prompts, preprocessing, decoding settings, dataset revision, IDs, and per-example outputs. Show both improvement over our backbone and the gap to the frontier baseline. Do not call a gain over a broken adapter a semantic improvement.

2. Research that informs the design

| Source | Evidence and implication |
| --- | --- |
| [ScreenSpot-Pro / ScreenSeekeR, 2025](https://arxiv.org/abs/2504.07981) | Training-free planner-guided visual search motivates selecting a region before fine localization. |
| [H Company Holo2 blog, Feb 2026](https://hcompany.ai/newsroom/holo2-235b-a22b-preview) | Reports 70.6% single-pass versus 78.5% within three steps on the full benchmark. Evidence that inference procedures can matter; not an FL-specific gain or a guarantee for Qwen. |
| [DiMo-GUI, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1334.pdf) | Separates text/icon handling and searches focal regions. Supports testing label-assisted localization separately from icon localization. |
| [UI-Zoomer, Apr 2026](https://zjureal.com/UI-Zoomer/) | Uses eight sampled boxes plus token confidence to gate and size crops. Its full method is not simply one extra call; probability access and cost must be checked. |
| [MHVF, Oct 13, 2026 preprint](https://arxiv.org/abs/2610.17792) | Searches multiple hypotheses and verifies marked local views. Supports retaining alternatives rather than trusting one initial point. |
| [AdaGround, Oct 27, 2026 preprint](https://arxiv.org/html/2610.36874v1) | Tests adaptive zoom and verification. On Qwen3-VL-8B ScreenSpot-Pro, its table reports 51.3% single pass, 63.5% fixed crop, 66.2% full pipeline. Full execution averages 12.5 calls; adaptive zoom alone trails fixed crop. This argues for a cheap crop control and component ablations, not copying the whole pipeline. |

These are authors' reported results on other models/protocols. None establishes a gain for our Qwen3.8-27B deployment on fruitloops_windows. Recent preprints are useful hypotheses, not independent replication. This is a targeted literature review, not an exhaustive survey.

3. First gate: adapter calibration

The current parser accepts point_2d and x/y but assumes original-image pixels. Parsing a response does not establish its coordinate convention. Our prior Mac examples cannot distinguish a semantic mistake from a model/adapter convention mismatch conclusively.

Confirm the model's recommended image format, coordinate units, preprocessing limits and decoding with its documentation and the deployment owner. Do not infer units merely because values are below 1000. Use 5–10 synthetic or separately collected calibration images with large obvious targets at diverse positions and multiple resolutions; never choose a per-example transform using benchmark ground truth. Log the exact bytes/dimensions sent, raw outputs, finish reason, model identity, and normalized-to-pixel/crop transformations. Freeze one explicit output convention. Test original image to crop to resized crop to original image round trips. Correct infrastructure and rerun the single-pass baseline before measuring semantic strategies.

4. Proposed FL-specific value

Use a small versioned collection of facts from Image-Line documentation, retrieved by task terms. Start with a deterministic lookup, not an embedding service. Keep text short and include no benchmark answers, image IDs, coordinates, screenshots, or per-example labels.

The [Channel Rack manual](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/channelrack.htm) distinguishes channel pan, channel volume, and the channel button. The [Mixer manual](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/mixer.htm) distinguishes mixer level faders and routing. Use these distinctions to prevent confusing channel volume with mixer/master volume. Current manuals may differ from version 20.8.3: verify applicability, using the [FL Studio 20 guide](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/boxinstaller/WebContent/FLStudio20_GettingStartedManual.pdf) where appropriate.

Represent the request as panel, named entity, control function, and visible evidence. For example, “snare channel volume” becomes Channel Rack → row visibly labeled Snare → volume control in that row. “Drums” is not automatically a synonym for “Snare.” Do not invent an entity that is not visible.

Proposed two-call strategy:

- Call 1 receives the full screenshot, original instruction and relevant manual facts. It returns a likely panel box, visible anchor label/box, target control type, and a provisional point. If context is unclear, preserve a broad region instead of cropping tightly. Routing must cover toolbars, menus, plugin windows and dialogs as well as Channel Rack/Mixer.
- Construct a crop using predicted panel/anchor geometry, retaining row labels or track headers and neighboring controls. Use native pixels and bounded enlargement; cropping increases effective detail but upscaling does not create missing information. Do not center every crop on the initial point. Reject invalid crop geometry without clamping predicted points.
- Call 2 receives the crop, the original instruction, and the semantic constraints, with an explicit crop-local coordinate contract. It returns one target; map it back using recorded transforms. Preserve the provisional point as a candidate if the second output is invalid. Record fallback and failure events instead of hiding them.

Optional third call, only after development evidence justifies it: compare the provisional and refined candidates using the full context plus labeled closeups. Ask which candidate matches panel, entity and control type, with a neither option. Marker placement must leave the control visible. Measure verifier-induced fixes AND regressions; an affirmative answer is not calibrated confidence. Candidate comparison is an adaptation, not an exact reproduction of the cited methods. No oracle selection using labels.

On the positive benchmark, neither/abstention is an incorrect prediction. For the desktop demo, it can instead suppress a misleading overlay. Keep these outcomes and metrics separate.

5. Freeze evaluation before tuning

Preferred: preserve all 57 official examples for the final test and develop on separately collected FL Studio 20 screenshots/tasks. If unavailable, split the official subset by screenshot/session before looking at failures, approximately 60/40 if group counts allow. All instructions sharing a screenshot or near-duplicate layout stay in one group. Record the real resulting counts, seed and manifest; do not assume 57 independent images. Balance text/icon coverage when possible.

If teammates have already used all examples for tuning, disclose that the full-subset score is exploratory and obtain a fresh holdout. A score on a smaller internal holdout is not directly comparable to a full-subset leaderboard score. No repeated final-test inspection to choose prompts, crops, or strategies.

Inspect development failures only. Count coordinate/format errors, wrong panel, wrong entity, wrong control type, small-target imprecision and occlusion. Choose the next intervention from the largest recoverable category. If failures are primarily geometric, fix the adapter; if labels are unreadable, try an OCR-assisted crop; if the right control is narrowly missed, test simple zoom before semantic routing. Add OCR only if it fixes demonstrated errors, and include its latency/dependencies in reports.

6. Run a small ablation matrix

| Experiment | Calls per example | What it isolates |
| --- | --- | --- |
| B0: corrected single pass | 1 | Real backbone baseline |
| B1: same prompt plus concise FL manual facts | 1 | Domain semantics without more calls |
| B2: existing fixed crop refinement | 2 | Generic zoom control |
| H1: panel/anchor crop without manual facts | 2 | Region selection |
| H2: panel/anchor crop with manual facts | 2 | Proposed specialized harness |
| H3: H2 plus candidate verification | 3 | Whether verification helps |

Reuse only exact matching cached requests, with cache keys including model revision, prompt/settings, image hash and transforms. Do not reuse B0 as H2's first call: their prompts differ. Keep a fresh cold-run latency measurement separate from cached experiment accounting. Add a matched-call generic alternative if H3 wins so extra compute is not mistaken for domain value.

Keep default baseline and existing refine behavior stable. Add named experimental strategies rather than silently replacing them. Generalize request/response types behind GroundingModel so strategies can request regions or verification without embedding Fireworks logic. Add batch records with status/error, final and intermediate coordinates, call counts, finish reasons, latency, usage and transforms. Store evaluation annotations in separate files inaccessible to inference.

Report paired correct/total, text and icon slices, fixes/regressions, errors, mean/p95 latency, calls, token usage, and estimated cost with dated pricing. Since Fireworks lists this model as on-demand, obtain the deployment billing basis; do not invent per-token prices. Report allocated GPU-time cost if applicable, including idle-time assumptions. Use screenshot-group bootstrap uncertainty if enough groups exist; show counts prominently when there are few groups.

One target is about 1.75 percentage points on 57. Set a development ambition of at least five net fixes (about 8.8 points on a full 57-case run), not a promised outcome or a significance threshold. Freeze the simplest strategy with a positive paired development gain and acceptable cost, then run the untouched final set once. Keep a component only if its measured benefit exceeds its regressions/cost; report a negative result honestly.

7. Suggested execution budget and deliverables

Provisional one-day allocation, to adjust when deadline/budget are known: 15% calibration and baseline contract; 20% dataset/split/scorer and development failure report; 35% implement/test the smallest supported strategy; 20% frozen evaluation and comparisons; 10% demo and presentation. If time is short, prioritize B0/B1/B2/H2 and drop verification, fine-tuning, vector retrieval, and UI polish.

An illustrative 34-development-example sweep costs 34 × (1+1+2+2+2+3) = 374 maximum inference calls without caching. A 23-example final test with B0 and a three-call harness costs at most 92 more, excluding calibration, frontier baseline and repeats. Counts are illustrative: actual grouped split, API latency, concurrency, and budget determine feasibility. No live calls are authorized by this document alone.

Submission evidence: pinned run manifests; one paired before/after table including the frontier baseline; per-example predictions and failed requests; development-only failure gallery; ablations; fixes and regressions; cost/latency; and a reproducible command. Present a baseline miss that the frozen harness fixes, plus a remaining failure. The product hypothesis is reliable control discovery in dense professional interfaces; demonstrate it with measured grounding accuracy and a cautious Mac overlay, without claiming autonomous workflow completion or productivity gains not measured.
