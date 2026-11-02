# evals

Scores any grounding strategy on the ScreenSpot-Pro benchmark with one shared scorer, so results
from different harnesses can be compared. Run everything from the repository root.

## Setup

```bash
uv sync
test -f .env || cp .env.example .env   # add your own FIREWORKS_API_KEY
uv run evals check                     # no network; never prints the key
uv run evals prepare                   # pinned dataset commit, about 46 MB; no model calls
uv run pytest -q
```

Your key needs access to the deployment in `FLGROUND_MODEL`. Deployments bill GPU time while
replicas are active, so a shared one is a shared bill.

## Get a score

```bash
# Try it on the development examples first
uv run evals run --mode reference --reasoning-effort xhigh --limit 1

# Qwen 3.8 on the 57 FL Studio targets
uv run evals run --mode reference --reasoning-effort xhigh --split eval --out evals/outputs/reference.jsonl \
  --label "Qwen3.8-27B on the FL subset using the Qwen3.5 adapter protocol."

# See or verify a run; compare two, example by example
uv run evals report evals/outputs/reference.jsonl
uv run evals report evals/outputs/reference.jsonl evals/outputs/mine.jsonl
```

Each finished run leaves results, a config, a report, and overlay images under `evals/outputs/`.

| Command | Purpose |
|---|---|
| `evals run` | Score a strategy. `--resume --out FILE` continues an unfinished run |
| `evals report RUN [RUN]` | Report one run, or compare two with an exact McNemar test |
| `evals rescore --run RUN` | Add the paper's metrics to a run made before they were recorded |
| `evals overlays --run RUN` | Redraw predictions over gold boxes |
| `evals cost --gpu-hourly-price P --gpus-per-replica G --billable-hours H --run RUN` | GPU cost estimate |

Settings: `--reasoning-effort`, `--max-tokens`, `--temperature`, `--response-format`, `--timeout`.
Defaults are `low`, `4096`, `0.0`, `none`, `180`.

## Bring your own harness

Write a module anywhere on the Python path:

```python
from evals.baseline import make_prediction

class MyHarness:
    def __init__(self, client):
        self.client = client   # a FireworksClient; ignore it if you use your own

    def predict(self, image, instruction):
        # image is the screenshot's path. Return a point in original-image pixels.
        return make_prediction(point_px=[x, y], calls=[...], latency_seconds=seconds)

def factory(client):
    return MyHarness(client)
```

```bash
uv run evals run --predictor my_module:factory                    # development examples
uv run evals run --predictor my_module:factory --split eval --out evals/outputs/mine.jsonl
```

- For the harness in `src/flground`: `--predictor evals.flground_adapter:baseline` or `:refine`.
- Return `invalid_reason` for an unusable answer. It scores `wrong_format`.
- Return `error` for an API failure. It stays out of accuracy and can be retried with `--resume`.
- Record each model call in `calls` with its `usage`, or its tokens count as unknown.
- A run is tied to the harness module's source hash, so editing that module blocks resume.
  Edits in modules it imports aren't detected.

## Keep the final set out of tuning

| Split | Source | Use |
|---|---|---|
| `dev` (default) | 8 Premiere examples | Building and tuning |
| `eval` | All 57 FL Studio targets | The final score only |

`prepare` refuses any overlap between them. Dev scores don't estimate FL Studio accuracy. Don't
change a prompt or setting because of `eval` results.

## How scoring works

- **Scorer:** the paper's `eval_sample_positive_gt` and metrics (`paper_scorer.py`, pinned commit).
  Inside the gold box, boundaries included, is `correct`. Outside is `wrong`. No point is `wrong_format`.
- **Parsing belongs to the strategy.** `--mode baseline` asks for strict 0-1 JSON and repairs nothing.
  `--mode reference` uses the authors' Qwen 3.5 prompt, 0-1000 coordinates, and tool-call parsing
  from a pinned commit, so a prose answer with no tool call scores `wrong_format`. See
  [the protocol record](QWEN35_ADAPTER_PROTOCOL.md) for its differences and required label.
- **API failures** are logged separately and keep the run unfinished until `--resume` retries them,
  so no example is silently dropped.
- **Tokens** are summed over every call. Reasoning is inside completion and cached inside prompt, so
  neither is added again. Missing usage is unknown, never zero.
- **Finished runs are locked:** files become read-only and the report records the results' hash.
  An existing output is never overwritten. To change anything, start a new run.
- **Data** is pinned to one dataset commit, and hashes are rechecked on every load.

## Comparing with published numbers

FL Studio is 57 of the benchmark's 1,581 targets and is harder than average. Published overall
scores aren't comparable with it; the leaderboard's per-application FL Studio numbers are closer.
Results stay out of git, so a score needs its saved run to back it up.

## Billing

The GPU estimate is `hourly price x GPUs per replica x replicas x billable hours`. Benchmark runtime
is only request latency; the billable lifetime includes idle time, so take it from your Fireworks
usage page. `--input-rate`, `--output-rate`, and `--cached-rate` add a hypothetical token-priced
figure that isn't this deployment's bill.

## Troubleshooting

| Symptom | Cause and action |
|---|---|
| `Configuration incomplete` | Fill the named fields in `.env`. |
| 401 / 403 | Bad key or no access. The run stops at once; fix it, then `--resume`. |
| 404 | Wrong `FLGROUND_MODEL`, or the deployment is gone. |
| 503 | Deployment scaling up. SDK retries are off; wait, then `--resume`. |
| `finish_reason=length` (exit 3 on dev) | Budget too small. Raise `--max-tokens` in a new run. |
| `already exists` | Add `--resume`, or choose another `--out`. |
| `configuration changed` on resume | Model, prompt, harness, or dataset differs. Start a new run. |
| `changed after it was finalized` | A locked results file was edited, so its report is invalid. |

Exit codes: `0` complete, `1` configuration or state error, `2` stopped or API failures remain,
`3` truncated output.
