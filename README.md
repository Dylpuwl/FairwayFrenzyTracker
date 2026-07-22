# Golf Shot Tracer

A Streamlit web app that draws a smooth, professional tracer line over a
normal-speed (30/60 fps) iPhone golf swing clip, with two fail-safes:

1. **Automatic tracking**: classical CV detection + a constant-acceleration
   Kalman filter, with a physics-shaped "ghost path" for brief mid-flight
   detection gaps (clouds, compression artifacts, motion blur).
2. **Manual fallback**: tap launch / apex / landing and the app draws a
   corrected quadratic Bezier curve that passes exactly through all three
   points.

Original audio is always preserved and re-synced via `ffmpeg`. Final output
is re-encoded to H.264 / yuv420p specifically so it plays back correctly in
iOS Safari.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — upload, tap-to-select, progress, download |
| `ball_tracker.py` | Ball detector (classical CV, or optional YOLOv8-nano) + Kalman filter |
| `physics_model.py` | Gap-fill physics model + corrected Bezier fallback |
| `video_processor.py` | Frame I/O, tracer rendering, ffmpeg audio mux |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages needed on Streamlit Community Cloud (ffmpeg, GL libs) |

## Test it on your iPhone right now (fastest path — no deployment)

1. On your computer: `pip install -r requirements.txt` (add `--break-system-packages`
   on some Linux setups), then:
   ```
   streamlit run app.py --server.address 0.0.0.0
   ```
2. Find your computer's LAN IP (macOS: `ipconfig getifaddr en0`; Windows: `ipconfig`).
3. On your iPhone, connect to the **same Wi-Fi network**, then open
   `http://<that-IP>:8501` in Safari.
4. Upload a clip straight from your camera roll and go.

This is the fastest loop while you're iterating — no public URL, no
deployment step, and you're testing on the actual device from minute one.

## Deploy it properly (shareable URL, HTTPS)

**Streamlit Community Cloud** (free):
1. Push this folder to a GitHub repo (include `requirements.txt` and
   `packages.txt` — the latter is what installs `ffmpeg` on the server).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   point it at `app.py`.
3. You'll get a `https://<something>.streamlit.app` URL — open that
   directly in mobile Safari on your iPhone. Tap the Share icon → **Add to
   Home Screen** for an app-like icon and full-screen launch.

**Alternative**: run it on your own machine/server as above and put it
behind a tunnel (e.g. `ngrok`, Cloudflare Tunnel) for a temporary public
HTTPS URL without a full deployment.

## Tuning knobs worth knowing about

- `video_processor.MAX_COAST_FRAMES` (default 45): how many frames the
  auto-tracker will "coast" on physics predictions before giving up and
  asking whether to fall back to manual mode.
- `video_processor.MIN_LOCK_FRAMES_BEFORE_COAST` (default 5): matches the
  spec's "successfully tracked for 5-10 frames" — a loss before this many
  confirmed detections skips straight to the fallback question rather than
  trusting a shaky, barely-established trajectory.
- `TracerRenderer(..., alpha=0.65)` and line thickness (auto-scaled to
  frame width in `render_path_over_video`) — cosmetic, adjust to taste.
- `mux_audio(..., crf=16)` — lower = higher quality / larger file. 16 is
  close to visually lossless; 18–23 is a more typical "high quality" range
  if file size matters more.

## Known limitations (read before relying on this for anything important)

- **Tripod recommended.** The default detector uses background subtraction,
  which assumes a mostly static camera. Handheld footage will hurt
  detection accuracy; see `ball_tracker.stabilize_frame` for an optional
  (off by default, CPU-costly) ECC-based stabilization step you can wire in.
- **No true 3D physics.** A single iPhone camera has no depth channel, so
  the gap-fill/aerodynamics is a *shape model* calibrated from the pixels
  you actually have, not a from-scratch simulation from real launch speed
  and spin. See `physics_model.py`'s module docstring for the reasoning,
  and `Projectile3D` for where a real 3D upgrade would plug in if you add
  camera calibration later.
- **Stock object detectors don't know "golf ball."** `YoloBallDetector` in
  `ball_tracker.py` is a ready-to-use drop-in, but it needs weights
  fine-tuned on your own annotated golf-ball frames — COCO's generic
  "sports ball" class isn't reliable on a small, fast, blurred golf ball.
- **Processing isn't instant.** A few seconds of 1080p60 footage takes
  meaningfully longer than its own runtime to process on typical hardware
  (background subtraction + per-frame drawing + a full re-encode). The
  progress bar is there to set expectations, not because something's stuck.
