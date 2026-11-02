# Run locally on macOS

## Install and update

With Homebrew available:

```bash
brew install uv git
```

First checkout:

```bash
git clone https://github.com/harrif23/pyrogrammers.git
cd pyrogrammers
uv sync --extra desktop
```

Existing checkout (preserve or commit your local changes before pulling):

```bash
cd pyrogrammers
git pull --ff-only
uv sync --extra desktop
```

Python 3.11+ is required; uv can provision Python when necessary. For screenshot-only
inference use `uv sync`; desktop dependencies are optional. Run all commands below
from the repository root.

## Configure .env

```bash
test -f .env || cp .env.example .env
nano .env
```

Set:

```dotenv
FIREWORKS_API_KEY=your_actual_api_key
FLGROUND_MODEL=accounts/yuxinz721-ofi0hn0kj3/deployments/h0xbbw9t
FLGROUND_BASE_URL=https://api.fireworks.ai/inference/v1
FLGROUND_MAX_TOKENS=32768
```

Save with Control-O, Enter, and exit with Control-X. Do not append /chat/completions
to the base URL. .env is ignored by Git and loaded automatically from your current
working directory. Existing environment variables take priority. To clear old exports:

```bash
unset FIREWORKS_API_KEY FLGROUND_MODEL FLGROUND_BASE_URL FLGROUND_MAX_TOKENS
```

Your key must have access to the deployment. Never paste the key into an issue, report,
or chat. The app sends the selected screenshot to Fireworks for inference.

## Test and predict

```bash
uv run pytest -q
uv run flground --help
```

Current suite: 65 tests. Unit tests use mocks and do not incur API charges.
Take an FL Studio screenshot with Shift-Command-4, then use its actual filename:

```bash
uv run flground predict --image "$HOME/Desktop/screenshot.png" \
  --instruction "Find the snare channel volume control"
```

Open the exact report path printed by the command:

```bash
open artifacts/REPLACE_WITH_RUN_ID/report.html
```

Inspect the crosshair against the target. This is a real, billed API call, not an
accuracy benchmark. To try two-call experimental refinement:

```bash
uv run flground predict --image "$HOME/Desktop/screenshot.png" \
  --instruction "Find the snare channel volume control" --strategy refine
```

## Desktop demo

Grant your terminal app Screen Recording permission under System Settings → Privacy
& Security; restart it if requested. Keep FL Studio visible on one display:

```bash
uv run flground demo --instruction "Find the snare channel volume control"
```

Run the command, then switch to FL Studio within five seconds. Keep the same window
and controls visible while inference runs. A cyan crosshair appears over the predicted
location for ten seconds, then disappears automatically. The transparent marker
passes mouse clicks through and does not activate Python or move your pointer.

Adjust the timing with `--delay 8 --overlay-seconds 20`; use `--no-overlay` to save
only the report. The marker is based on the captured image and does not follow a
moving window or control. A saved report is still produced for every successful run.

Default behavior does not click. For clicks, also grant Accessibility
permission, verify the overlay, and opt in:

```bash
uv run flground demo --instruction "Find the snare channel volume control" --execute-click
```

An interactive confirmation is required. Exact screenshot changes cause refusal;
animations or switching windows to view the report can trigger this. Use capture-only
mode if that happens. PyAutoGUI's corner fail-safe remains enabled. Real Mac capture,
overlay rendering/alignment, Retina mapping, and clicking still require local validation.
The overlay is dismissed before the existing click confirmation and stale-screen check.

## Troubleshooting

- Missing configuration: check that .env exists in the current directory and contains
  nonempty key/model entries. Clear stale exports as above.
- 401: verify the key; 403 or deployment/model access errors: verify account permissions
  and the exact deployment ID. Do not share the key with diagnostic output.
- 429: inspect quota/rate limits; retries are disabled to keep calls explicit.
- Timeout: confirm deployment readiness; current timeout is 180 seconds.
- JSON validation error: the model returned something other than the required numeric
  x/y JSON, or coordinates were invalid. Preserve a sanitized error for adapter work;
  do not treat this as a successful prediction.
- Missing image: use the real screenshot filename and quote paths containing spaces.
- Black/failed capture: check Screen Recording permission and restart your terminal.
- Desktop failure on Linux: expected; only macOS desktop operation is supported.

When reporting an issue, include OS version, command with secrets removed, error text,
model ID, and whether it occurred in predict or demo. Do not include sensitive screenshots.

Experimental panel strategies, calibration, and batch comparisons are documented in
[EXPERIMENTS.md](EXPERIMENTS.md). Use `--strategy panel` for two-stage localization
or `--strategy panel_verify` for optional verification. These have mocked test
coverage but no measured benchmark gain yet.
