# FLGround

A Python screenshot-to-coordinate harness for FL Studio. Runs locally on macOS;
vision inference uses Fireworks. No local GPU required.

## Setup

Install Python 3.11+ and uv, then run from this repository:

```bash
uv sync --extra desktop
test -f .env || cp .env.example .env
nano .env
# Paste your API key, save with Ctrl-O then Enter, exit with Ctrl-X.
uv run flground --help
uv run pytest
```

Use `uv sync` without the desktop extra for offline screenshot workflows.
The adapter automatically loads `.env` from the current working directory.
Run commands from the repository root to use its `.env`. Existing exported shell
variables take precedence over `.env` values. `.env` is excluded from Git.
The example selects `accounts/yuxinz721-ofi0hn0kj3/deployments/h0xbbw9t`.
Your API key must have access to that deployment. The SDK appends
`/chat/completions` to `FLGROUND_BASE_URL`; do not include that suffix in the variable.
Requests use `top_k=40`, zero presence/frequency penalties, and a configurable
`FLGROUND_MAX_TOKENS=32768` ceiling. Temperature is left at the endpoint default,
matching the supplied deployment example. Calls time out after 180 seconds.
Never commit credentials. Live deployment access has not been validated here.
Images are sent to the configured inference endpoint.

## Ground a screenshot

```bash
uv run flground predict --image screenshot.png --instruction "Find the snare channel volume control"
uv run flground predict --image screenshot.png --instruction "Find the snare channel volume control" --strategy refine
```

Every successful run saves JSON, an annotated PNG, and an HTML report under a unique
`artifacts/` directory. Cyan marks the first prediction; red marks the final one.
Baseline uses one call. Experimental refinement uses two: an initial prediction,
then a 2x enlarged 512-pixel crop centered near it, with explicit coordinate mapping.
Refinement is opt-in and is not claimed to improve accuracy. It can lose context or
reinforce an incorrect initial prediction. Select an optimization using development
failures, not held-out labels. Raw responses and token usage are saved for successful
calls; invalid JSON or out-of-bounds predictions currently stop the command.

## macOS demo

Grant Screen Recording and Accessibility permissions to the terminal/application
running Python, then restart it if required. Use a single display. FL Studio must
already be open; this tool does not launch or navigate it.

```bash
uv run flground demo --instruction "Find the snare channel volume control"
uv run flground demo --instruction "Find the snare channel volume control" --execute-click
```

The default only captures and predicts. Clicking requires the explicit flag and an
interactive confirmation after inspecting the overlay. The adapter scales screenshot
pixels to desktop coordinates for Retina displays, retains PyAutoGUI's corner
fail-safe, and rejects a changed screenshot before clicking. This conservative check
can reject animated controls or changes caused by switching windows; use overlay-only
mode in that case. Multi-monitor setups are deliberately rejected. Desktop support
must be validated on a real Mac before relying on clicks.

## Teammate integration

`GroundingModel` is a protocol separating model access from the harness. Match your
team's model, prompt, image preprocessing, and inference settings before comparing
results: the included generic prompt is not presumed identical to their baseline.
Request dataset revision, split IDs, per-example predictions, raw responses, latency,
usage, and development failure labels. Keep held-out failures sealed until tuning ends.

Current scope: adapter, baseline/refinement harness, geometry, CLI, reports, and macOS
capture/click adapter. Benchmark ingestion, batch failure recording, comparison metrics,
and evidence-selected semantic strategies remain follow-up work. There are no measured
accuracy improvements yet. No fine-tuning or autonomous navigation is included.
