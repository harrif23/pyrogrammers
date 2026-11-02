FLGround hold-to-talk demo

One-command launch from the repository root:

```bash
python3 demo_app/launch.py
```

Default UX is now a background accessory with no foreground window or Dock icon:

1. Launch once, then stay in FL Studio.
2. Hold Option+Z. A compact HUD appears centered just below the physical notch
   (top-center on a display without a notch). Wait for Listening before speaking.
3. Speak your request; the waveform responds to microphone amplitude.
4. Release Z or Option. The transcript is submitted automatically to the local harness.
5. A cyan marker points to the predicted control for ten seconds. Escape cancels.

The HUD disappears before screenshot capture so it is not included in the model input.
The screenshot is captured after transcription completes; keep the view unchanged.
A first-run permission prompt can delay recording. Grant permissions, then retry the
shortcut. A very short tap may capture no speech. Recording is capped at 20 seconds.

Grant Accessibility and, if macOS requests it, Input Monitoring to the terminal/app
launching Python. These allow the global shortcut to be captured and suppressed in
FL Studio. Microphone, Speech Recognition, and Screen Recording permissions are still
needed. Restart after permission changes. The FL menu-bar item provides Quit and Cancel.
Only the Option+Z chord and Escape during an active request/marker are consumed.
Speech recognition remains local; no audio is sent to the grounding model.

16 mocked/offline tests passed; native event-tap API imports and Swift compilation
passed. Actual global key interception, permission prompts, waveform appearance and
voice-to-pointer execution still require validation on your desktop.

This uses ~/.local/bin/llama serve with your Gemma E2B Q8 model/projector and
-ngl 99 -c 65536 -b 4096 -ub 2048 on localhost:8080. It builds the voice helper
if needed, waits for /health and verifies the served model before opening the app.
Quit using the FL menu or Ctrl-C in the launcher terminal. It stops only a server
it started; an existing matching server is reused and left running. An occupied
port with a different model is rejected. --port, --binary, --model, --mmproj and
--timeout allow overrides. --dry-run validates paths and prints commands without
starting anything. If a voice-helper build fails, typed input remains available.
The launcher's paths/command and lifecycle helpers were tested; a full GUI/model
startup has not been exercised during implementation.

This is a standalone frontend on main. It adds files only under demo_app and build/run
artifacts under artifacts/. It does not edit harness code, .env, dependencies, uv.lock,
benchmark inputs, or the running benchmark. No package sync/install is needed.

Default route: local Apple on-device speech → editable transcript → local Gemma 4 E2B
via llama-server → existing FLGround panel harness → native on-screen marker.
This is Gemma, not Gemini. The Qwen Fireworks benchmark is a separate experiment.
The demo never falls back to Fireworks. Its local model adapter discovers the single
loaded model from /v1/models and accepts only loopback HTTP endpoints.

Start your existing llama-server with the model and --mmproj arguments you supplied,
on port 8080. It was not running when this frontend was checked. Running Gemma consumes
local CPU/GPU/RAM, though it does not use the benchmark's hosted Qwen deployment.

Legacy window mode (optional, not used by the one-command launcher):

```bash
# Already built in this checkout; repeat on a fresh checkout or after Speech.swift edits.
.venv/bin/python demo_app/build.py

# Launch the native window; no paid API calls and no dependency changes.
.venv/bin/python demo_app/app.py
```

Use it:

1. Open FL Studio on one display, with the desired controls visible.
2. In FLGround, click Speak. Grant Microphone and Speech Recognition permission if asked.
3. Ask “Where is the Drums channel button?” Click Stop, or recording stops at 20 seconds.
4. Review/edit the transcript. Typing is always available.
5. Click Show me; the window hides. Switch to FL Studio within five seconds and keep its
   layout/page unchanged until the marker appears. Requests only proceed if FL Studio is frontmost.
6. The cyan marker lasts ten seconds and passes mouse input through. No clicks or cursor
   movements are performed. Use the FL ◎ menu-bar item to ask again, cancel, or quit.
7. Open last report shows the captured screenshot, predicted point and model traces.

Speech is requested with requiresOnDeviceRecognition=true and only starts when
supportsOnDeviceRecognition is available. If local recognition is unavailable for
English, the app reports the issue and leaves typed input available. Audio is not saved
or sent to llama-server/Fireworks. Microphone buffers are transient. Transcripts and
screenshots appear in local report artifacts after you submit a request. macOS may
need an English dictation asset downloaded in advance; airplane-mode transcription
and actual permission prompts still require a real user test.

Screen Recording permission belongs to the launching application/Python context.
The helper bundle is named FLGround Voice. If permissions fail, inspect Privacy &
Security settings and restart the launching app/helper. Window mode uses Speak/Stop buttons. The default launcher instead uses the global
Option+Z hold-to-talk shortcut described above.

The UI and backend run separately. The Cocoa UI stays on the main thread; inference
runs in a child process. Cancel terminates that child and ignores late events; a request
already accepted by llama-server may continue server-side. Speech errors and absent
model targets return to the request field. No overlapping frontend jobs are allowed.
The frontend checks foreground application and display geometry before showing the
marker. It does not track controls, detect every in-window navigation/layout change,
or promise stale-screen safety. Keep the view static. Existing click-mode safeguards
in the harness are not changed or used by this pointing-only app.

The default demo coordinate contract is normalized_1000, explicitly requested from
the model. It worked on Qwen but has NOT been calibrated for local Gemma. To test an
alternative explicitly, use --coordinates pixels (or normalized_1). Do not infer units
from raw magnitudes or apply per-example corrections using benchmark labels. Local
Gemma may also fail the panel JSON schema; reports preserve errors without guessing.
Choose Baseline in the dropdown for a one-call control, or Panel + verification for
up to three calls. Accuracy is unmeasured for all local-model routes.

Validation performed: native Python imports/class registration; Swift helper compilation
and ad-hoc signing; mocked local endpoint routing, coordinate mapping and failure checks;
checksums confirming every pre-existing tracked/unignored project file is unchanged.
No actual microphone recording, desktop screenshot, live model inference or visual
alignment was exercised by these checks. The Swift audio tap API produces a macOS 27
deprecation warning but compiled successfully. UI appearance still needs visual review.

Independent tests (no uv sync and no benchmark requests):

```bash
.venv/bin/python -m pytest demo_app/test_demo.py -q -p no:cacheprovider
```

References:
[Apple on-device recognition](https://developer.apple.com/documentation/speech/sfspeechrecognitionrequest/requiresondevicerecognition),
[llama-server API](https://github.com/ggml-org/llama.cpp/tree/master/tools/server),
[Google Gemma llama.cpp deployment](https://github.com/google-gemma/docs/blob/main/docs/deployment/llamacpp.md).

### Voice helper privacy crash

If the HUD says the helper stopped unexpectedly and a FLGroundVoice crash report
reports TCC / missing NSSpeechRecognitionUsageDescription, rebuild and restart:
`python3 demo_app/build.py`, then `python3 demo_app/launch.py`.
The builder now embeds both privacy descriptions in the executable as well as the
bundle Info.plist, because the helper is launched directly with subprocess pipes.
`artifacts/demo-build/FLGround Voice.app/Contents/MacOS/FLGroundVoice --check-metadata`
(with the executable path quoted) validates metadata without requesting microphone
access. Actual speech capture still requires a live permission/recording check.
