# Pyrogrammers

**An open-model agent harness that operates professional creative software.**

Demo use case: music production in FL Studio.

Fireworks Deploy Buildathon · Microsoft × NVIDIA · San Francisco · November 2, 2026

| +10.5 points | 32 / 57 | 0 |
|:---:|:---:|:---:|
| From 45.6% to 56.1% on the same model deployment | FL Studio controls found by our best run, up from 26 / 57 | Fine-tuning. All of the gain comes from the harness around the model |

> **How to read this result:** it's exploratory. We studied these 57 cases while building, and the gain isn't statistically significant on its own (exact McNemar p = 0.21).

---

## Meet Cameron: our user, on our team

**Wants to be Fred again.. yesterday. Has produced for 30 minutes.**

Self-proclaimed playlist savant, guitar player, novice DJ. Cameron has the taste. He needs an agent to drive FL Studio so his ideas come out sounding right.

| 6 months | 30 minutes | 0 |
|:---:|:---:|:---:|
| DJing | Producing, ever | Tracks released |

---

## The problem

**Professional creative software is powerful, and punishing to operate.**

- **Dense screens:** hundreds of tiny knobs, icons and panels on one high-resolution screen.
- **Repetitive chores:** routing, mixing setup and exporting take many clicks, every session.
- **Steep learning curve:** new users spend their time hunting for controls instead of making music.

Today's AI agents struggle here too. The **ScreenSpot-Pro** benchmark exists to measure how well models find controls in professional apps like these.

---

## The business goal

**Help people use the professional creative tools they already have.**

We're building computer use for creative professionals, starting with the step every action depends on: finding the right control.

| | |
|---|---|
| **Who it's for** | First-time producers first, then creative professionals |
| **What it does today** | Ask for a control out loud and see where it is on screen |
| **Where it's going** | An assistant that operates the software with you |
| **Why this approach** | It improves an open model through a harness, with no fine-tuning |

FL Studio is our first measured application. Other professional tools, such as CAD and video editing, are possible next steps that we haven't built or tested.

---

## The approach: a harness around Qwen 3.8

**Pyro adds FL Studio context and structured localization.**

1. **Ask:** screenshot plus the user's request. The baseline asks only for a click point.
2. **Add context:** FL Studio rules distinguish panels, named items and control functions.
3. **Identify:** name the panel and exact item, and return a detailed or compact plan.
4. **Check:** validate the point. Recover from token limits, or look again if ambiguous.
5. **Show:** map the point to screen pixels. The demo marks the target for the user.

**How it's scored:** a hit means the point lands inside the target box, across all 57 FL Studio cases in ScreenSpot-Pro. Errors and missing points count as misses.

**Platforms:** Fireworks serves Qwen3.8-27B on one dedicated deployment, running on NVIDIA B200 GPUs. A local Gemma model transcribes speech in the demo.

---

## The harness: four configurations

**One model. Different ways of asking.**

| Configuration | Model calls | What it does |
|---|:---:|---|
| **Baseline** | 1 | Asks only for the click point. This is what the harness has to beat |
| **Panel recover** | 1–2 | A detailed plan with panel, item, control, evidence and boxes. Retries simply if it runs out of tokens |
| **⭐ Panel compact (recommended)** | 1 | A short plan with panel, item and point. Less to write means less time and fewer tokens |
| **Panel selective** | 1–2 | Same as compact. If the model says it's ambiguous, it takes a second look at a zoomed crop |

The harness adds short rules from Image-Line's manuals, such as "Channel Rack volume, Mixer track level, and main monitor volume are different controls." They contain no benchmark answers.

---

## Results: ScreenSpot-Pro, FL Studio, 57 cases

**Six more correct predictions, with no fine-tuning.**

| Configuration | Correct | Accuracy | Icons / 31 | Text / 26 | Fixes | Regressions | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 26 / 57 | 45.6% | 8 | 18 | n/a | n/a | n/a |
| Panel recover | 30 / 57 | 52.6% | 9 | 21 | 7 | 3 | 0.34 |
| **⭐ Panel compact (recommended)** | **31 / 57** | **54.4%** | **11** | **20** | **10** | **5** | **0.30** |
| Panel selective (top score) | 32 / 57 | 56.1% | 11 | 21 | 11 | 5 | 0.21 |

> **⭐ Our recommendation: panel compact.** Five more correct than baseline, one model call per case, about 8% more tokens, and the fastest in this run. Panel selective scored one more, 32 / 57, but that's within run-to-run noise and it used 17% more tokens than compact. Compact's weakness: 8 of 57 cases ended with no usable point.

This is a matched rerun: the same deployment, an 8,192-token limit for every configuration, and temperature 0. The three harness configurations are effectively tied; an earlier run put panel recover first at 32 / 57.

---

## Latency and cost

**Our recommended configuration, panel compact, is also the fastest in this run.**

Seconds per case, including all of a case's model calls:

| Configuration | Median | Mean | p95 | Calls for 57 | Total tokens | Tokens vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.7 s | 8.4 s | 30.3 s | 57 | 263,515 | n/a |
| Panel recover | 8.0 s | 16.0 s | 54.2 s | 63 | 403,532 | +53% |
| **⭐ Panel compact (recommended)** | **3.7 s** | **7.0 s** | **39.8 s** | **57** | **285,485** | **+8%** |
| Panel selective | 5.9 s | 10.7 s | 44.4 s | 59 | 335,180 | +27% |

- **Conditions:** one run, at up to 15 cases at a time.
- **The slowest cases:** most answers take seconds, but the slowest 5% took 30–54 s or more. Latency also varies between runs: compact and selective swapped order in our earlier run.
- **Voice demo:** saved tempo-control requests took about 3.7 and 4.8 s in the harness with panel recover. Speech transcription and the demo's extra localization call add time on top.
- **Cost:** the deployment bills GPU time, so token counts aren't a dollar cost. We don't have a measured cost per request.

---

## Demo: pointing assistant

**Ask for a control. See where it is.**

> Cameron asks: "Where do I add reverb to the snare?"

1. **Speak:** hold Option+Z. Local Gemma transcribes the recorded request, and the audio stays on your Mac.
2. **Locate:** Fireworks receives the transcript and screenshot. The harness finds a point.
3. **Show:** a cyan marker indicates the predicted control. The user stays in control.

The voice demo points without clicking. It makes one extra localization call that isn't part of the benchmark score, and live success across requests still needs verification.

---

*Results: matched rerun on dataset revision `210e78d3`, ScreenSpot-Pro `fruitloops_windows`, using the paper's point-in-box rule. Significance: exact McNemar test against baseline.*
