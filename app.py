"""
app.py
-------
Streamlit UI tying together ball_tracker / physics_model / video_processor.

Flow
----
  1. Upload a clip (from the iPhone camera roll via mobile Safari's file picker)
  2. Tap the ball at the impact frame -> automatic tracking runs
     (Fail-Safe 1 handles brief mid-flight loss on its own)
  3. If tracking stops early, YOU decide: was that the natural end of the
     shot (ball landed / left frame), or did it lose the ball? CV alone
     can't reliably tell those apart, so we ask rather than guess.
  4. If it lost the ball (or you skip straight there), tap launch / apex /
     landing for the Bezier fallback (Fail-Safe 2)
  5. Preview + download the final tracer video, original audio intact
"""

import os
import tempfile
from typing import Tuple

import cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from video_processor import (
    get_video_meta,
    process_video_auto,
    process_video_manual,
    render_path_over_video,
    freeze_path_from,
)

st.set_page_config(page_title="Golf Shot Tracer", page_icon="🏌️", layout="centered")

PREVIEW_MAX_WIDTH = 480

# Grounded in the subject rather than a generic dark-mode default: a real
# fairway green, not a generic near-black + neon accent.
st.markdown("""
<style>
    .stApp { background-color: #10221A; }
    h1, h2, h3, p, label, .stMarkdown, .stCaption { color: #EDF3EE !important; }
    .stButton>button {
        border-radius: 8px; font-weight: 600; border: 1px solid #3E6A50;
        background-color: #2D5A3D; color: #EDF3EE;
    }
    .stButton>button:hover { background-color: #3E6A50; border-color: #4C8563; }
    div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #3E6A50; }
</style>
""", unsafe_allow_html=True)


def extract_preview_frames(video_path: str):
    """Decodes every frame once, downscaled to PREVIEW_MAX_WIDTH, so the
    frame slider is instant (no re-decoding per interaction) and a
    several-second 4K clip doesn't blow up memory. Returns (frames_rgb,
    scale) -- `scale` maps a tap on the preview image back to
    source-resolution coordinates for the actual tracker."""
    cap = cv2.VideoCapture(video_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = PREVIEW_MAX_WIDTH / src_w if src_w > PREVIEW_MAX_WIDTH else 1.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, scale


def to_source_xy(preview_xy: Tuple[float, float], scale: float) -> Tuple[float, float]:
    return preview_xy[0] / scale, preview_xy[1] / scale


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def reset_state():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


st.title("🏌️ Golf Shot Tracer")
st.caption("Upload a normal-speed iPhone clip (30/60 fps) — no slow-mo needed. "
           "Best results with the camera on a tripod; handheld footage may reduce tracking accuracy.")

uploaded = st.file_uploader("Upload swing video", type=["mov", "mp4", "m4v"])

if uploaded is None:
    st.info("Upload a video to get started.")
    st.stop()

if st.session_state.get("uploaded_name") != uploaded.name:
    reset_state()
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state.video_path = video_path
    st.session_state.uploaded_name = uploaded.name
    with st.spinner("Decoding preview…"):
        frames, scale = extract_preview_frames(video_path)
    st.session_state.preview_frames = frames
    st.session_state.preview_scale = scale
    st.session_state.mode = "auto_pick_impact"

frames = st.session_state.preview_frames
scale = st.session_state.preview_scale
n_frames = len(frames)

color_hex = st.color_picker("Tracer color", "#FF3B30")
color_bgr = hex_to_bgr(color_hex)

st.divider()
mode = st.session_state.mode

# ---------------------------------------------------------------- STEP 1
if mode == "auto_pick_impact":
    st.subheader("Step 1 — Tap the ball at the moment of impact")
    idx = st.slider("Scrub to the impact frame", 0, n_frames - 1, n_frames // 4, key="impact_slider")

    # Key includes the frame index on purpose: streamlit_image_coordinates
    # keeps returning its LAST value across reruns, so without this, moving
    # the slider to a new frame (without clicking) would silently keep the
    # old click's (x, y) but pair it with the new frame's index.
    coords = streamlit_image_coordinates(frames[idx], key=f"impact_click_{idx}")
    if coords is not None:
        st.session_state.impact_frame = idx
        st.session_state.impact_xy = to_source_xy((coords["x"], coords["y"]), scale)

    if "impact_xy" in st.session_state:
        ix, iy = st.session_state.impact_xy
        st.success(f"Impact set: frame {st.session_state.impact_frame} at ({ix:.0f}, {iy:.0f})")

    c1, c2 = st.columns(2)
    with c1:
        run_auto = st.button("▶ Track shot automatically", type="primary",
                              disabled="impact_xy" not in st.session_state)
    with c2:
        skip_manual = st.button("Skip to manual mode")

    if run_auto:
        progress = st.progress(0.0, text="Tracking ball…")
        out_dir = tempfile.mkdtemp()
        result = process_video_auto(
            st.session_state.video_path, os.path.join(out_dir, "tracer_output.mp4"),
            st.session_state.impact_frame, st.session_state.impact_xy, color_bgr,
            progress_callback=lambda p: progress.progress(min(p, 1.0), text="Tracking ball…"),
        )
        if result["tracking_stopped"]:
            st.session_state.path_so_far = result["path_so_far"]
            st.session_state.stopped_at_frame = result["stopped_at_frame"]
            st.session_state.mode = "confirm_stop"
        else:
            st.session_state.output_path = result["output_path"]
            st.session_state.mode = "done"
        st.rerun()

    if skip_manual:
        st.session_state.mode = "manual_pick_points"
        st.rerun()

# ---------------------------------------------------------- CONFIRM STOP
elif mode == "confirm_stop":
    stopped_at = st.session_state.stopped_at_frame
    st.warning(
        f"Automatic tracking stopped at frame {stopped_at}. This can mean the ball simply "
        f"left the frame or landed (normal end of shot), or that tracking lost it "
        f"(clouds, motion blur, low contrast, etc)."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ That's the end of the shot — finish"):
            progress = st.progress(0.0, text="Rendering…")
            out_dir = tempfile.mkdtemp()
            out_path = os.path.join(out_dir, "tracer_output.mp4")
            frozen = freeze_path_from(st.session_state.path_so_far, stopped_at)
            render_path_over_video(
                st.session_state.video_path, out_path, frozen, color_bgr,
                progress_callback=lambda p: progress.progress(min(p, 1.0), text="Rendering…"),
            )
            st.session_state.output_path = out_path
            st.session_state.mode = "done"
            st.rerun()
    with c2:
        if st.button("❌ It lost the ball — use manual mode"):
            st.session_state.mode = "manual_pick_points"
            st.rerun()

# --------------------------------------------------------- MANUAL MODE
elif mode == "manual_pick_points":
    st.subheader("Tap launch, apex, and landing")
    st.caption("Scrub to the right frame for each point, then tap the ball's position on the image.")
    point_key = st.radio("Which point are you setting?", ["launch", "apex", "landing"],
                          horizontal=True, format_func=str.title)
    default_idx = st.session_state.get(f"{point_key}_frame", n_frames // 4)
    idx = st.slider("Scrub to the right frame", 0, n_frames - 1, default_idx, key=f"slider_{point_key}")

    coords = streamlit_image_coordinates(frames[idx], key=f"click_{point_key}_{idx}")
    if coords is not None:
        st.session_state[f"{point_key}_frame"] = idx
        st.session_state[f"{point_key}_xy"] = to_source_xy((coords["x"], coords["y"]), scale)

    for p in ["launch", "apex", "landing"]:
        if f"{p}_xy" in st.session_state:
            px, py = st.session_state[f"{p}_xy"]
            st.caption(f"{p.title()}: frame {st.session_state[f'{p}_frame']}, ({px:.0f}, {py:.0f})")

    ready = all(f"{p}_xy" in st.session_state for p in ["launch", "apex", "landing"])
    if ready and st.button("▶ Generate tracer", type="primary"):
        progress = st.progress(0.0, text="Rendering tracer…")
        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "tracer_output.mp4")
        process_video_manual(
            st.session_state.video_path, out_path,
            launch=(st.session_state["launch_frame"], st.session_state["launch_xy"]),
            apex=(st.session_state["apex_frame"], st.session_state["apex_xy"]),
            landing=(st.session_state["landing_frame"], st.session_state["landing_xy"]),
            color_bgr=color_bgr,
            progress_callback=lambda p: progress.progress(min(p, 1.0), text="Rendering tracer…"),
        )
        st.session_state.output_path = out_path
        st.session_state.mode = "done"
        st.rerun()

# --------------------------------------------------------------- DONE
elif mode == "done":
    st.subheader("Done! 🎉")
    st.video(st.session_state.output_path)
    with open(st.session_state.output_path, "rb") as f:
        st.download_button("⬇ Download tracer video", f, file_name="golf_tracer.mp4", mime="video/mp4")
    if st.button("Start over"):
        reset_state()
        st.rerun()
