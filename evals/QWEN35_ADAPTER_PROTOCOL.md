# Qwen3.5 adapter protocol

This is the default baseline. `--mode reference` runs another model through the benchmark authors' Qwen 3.5 adapter. Label results
from our deployment exactly:

> Qwen3.8-27B on the FL subset using the Qwen3.5 adapter protocol.

**This is not a reproduction of Qwen3.5's published score.** The model and serving backend differ.
Reproducing that score means running Qwen3.5 with its originally reported configuration.

```bash
uv run evals run --split eval \
  --label "Qwen3.8-27B on the FL subset using the Qwen3.5 adapter protocol." \
  --out evals/outputs/qwen38_on_qwen35_protocol.jsonl
```

## Reference

- **Repository:** `likaixin2000/ScreenSpot-Pro-GUI-Grounding`, `main` at
  `dbe00114bc53a32c61c1a267786da85967710da8`. `models/qwen3_5.py` was last changed in
  `83f3041fdb76848e29a6c76094495b2489a3b817`, and the two versions of that file are identical.
- **Path:** the non-prefilled Transformers path, `Qwen3_5Model`, which `model_factory.py` selects for
  `qwen3_5`. The same file's vLLM path, `Qwen3_5_VLLM_Model`, is not the reference. It prefills the
  tool call and uses `temperature=0.7`, `max_tokens=100`, and a lower pixel limit.
- **Scorer:** `eval_screenspot_pro.py`, `eval_sample_positive_gt`, at commit `6ca3cf90`.

## What matches

| Item | Treatment |
|---|---|
| Messages | `get_qwen3_5_prompt_msg`: same system prompt, image before instruction, bare instruction. A check compares the text against the pinned source |
| Tools | Described only in the system prompt. No API `tools` and no structured output |
| Parsing | XML-wrapped `computer_use` call, using the last `<tool_call>` block |
| Coordinates | 0-1000, divided by 1000; two values are a point, four are a box whose center is used; no range check |
| Preprocessing | RGB, then `smart_resize` (factor 8, 1,024 to 8,294,400 pixels), then PNG. Every image in both splits is 2560x1440, which this leaves unchanged |
| Output limit | 4096 tokens |
| Decoding | `temperature=0`, the closest API setting to `do_sample=False` |

## Known differences

| Item | Reference | Ours | Effect |
|---|---|---|---|
| Model | Qwen3.5 | Qwen3.8-27B, as named in every response | Scores aren't comparable as the same model |
| Backend | Local Transformers | Fireworks chat completions | |
| Chat template | `apply_chat_template` from the checkpoint | Applied by Fireworks; not visible to us | **Unknown.** We can't confirm the final prompt text |
| Image processing | Their `AutoProcessor` receives the resized image | Applied by Fireworks; not visible to us | **Unknown.** About 4,005 prompt tokens is consistent with no downscale (3,600 at 32 px per token), but doesn't prove it |
| Reasoning | No setting; the checkpoint template's default | `reasoning_effort` is set per run | Deliberate; `xhigh` so far |
| Reasoning text | Part of the decoded response | Returned separately in `reasoning_content` | Parsing sees only the final answer |
| Sampling | Greedy decoding | Other server defaults aren't visible | Greedy decoding isn't confirmed |
| Context overflow | Not applicable | `context_length_exceeded_behavior=error` | Keeps the output limit from being lowered silently |
| Non-finite values | Would score as `wrong`, or crash on huge integers | `wrong_format` | Incorrect in both cases |

Answers that are prose with no tool call score `wrong_format`, as in the reference parser. Whether
Qwen3.5 under the reference path answers in prose as often isn't known without running it.
