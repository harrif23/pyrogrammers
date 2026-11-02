# Working on FLGround

Read README.md, docs/PROJECT_CONTEXT.md, and docs/LOCAL_SETUP.md before making changes.

## Scope and design

- Build a small screenshot-to-coordinate harness, not an autonomous agent.
- Python 3.11+, uv, Fireworks OpenAI-compatible SDK, Pydantic, Pillow, Typer.
- Local desktop target is macOS, initially one display; cloud/Linux runs support
  screenshot inference and tests, not proof of macOS desktop compatibility.
- Keep provider logic behind GroundingModel and strategies separate from the adapter.
- Preserve original-image pixel coordinates. Explicitly map crop/resize transforms;
  reject invalid outputs instead of silently clamping them.
- The default path uses one model call. Experimental refinement uses two.
- Do not feed benchmark annotations or correctness labels into inference.
- Choose semantic optimizations from development failure evidence. Refinement is
  experimental infrastructure, not a validated improvement.

## Workflow

Use the existing checkout. Cloud tasks are already isolated; do not create worktrees
unless requested. Check git status before editing and preserve teammate changes.
Use uv sync (or uv sync --extra desktop on Mac), then uv run pytest -q.
Commit uv.lock with dependency changes. Run git diff --check before committing.
Use focused tests for transforms, parsing, configuration, and model-call behavior.
Mock hosted inference for unit tests; live API calls incur cost and need credentials.
Never claim a live API or desktop test passed based on mocked tests.

## Configuration and security

.env loads from the current working directory; exported variables win.
.env.example contains only placeholders and non-secret settings. Never print, commit,
or copy actual API keys into reports or documentation. Do not inspect .env values
when checking setup; inspect presence only. Keep run artifacts out of source control.
Desktop capture is the whole display and is sent to Fireworks. Clicking remains opt-in
and confirmed; keep the fail-safe and stale-screen checks. Document their limitations.

## Evaluation and handoff

Score strategies through `evals/` (see evals/README.md) so every result shares one scorer
and dataset revision. Tune on the dev split; `--split eval` is for final scores only.
For results made elsewhere, obtain the model, prompt, preprocessing, inference settings,
dataset revision, split manifest, and per-example outputs before claiming comparability.
Keep held-out failures out of tuning. Count API/parsing errors
as failed predictions in future batch evaluation; do not silently discard examples.
Report fixes AND regressions, counts, latency, usage, and estimated costs with sources.
Update docs when implementation or validation status changes. Publish no accuracy
claims unsupported by saved reproducible experiments. Follow the user's requested
commit/push scope; never force-push over teammate work.
