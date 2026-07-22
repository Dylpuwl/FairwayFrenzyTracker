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
    extrapolate_lost_flight,
)
from impact_detection import detect_impact_frame

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
    .stButton>button:disabled {
        background-color: #1A2E22; color: #6F8A7A; border-color: #23402F; opacity: 0.6;
    }
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


@st.fragment
def impact_picker_fragment(video_path, frames, scale, n_frames, color_bgr):
    """
    Everything a person touches repeatedly while hunting for the exact
    impact frame lives in here. Wrapped in @st.fragment so that dragging
    the slider, tapping +/-, or tapping the ball only reruns THIS function
    -- not the whole page -- which is what was causing the page to jump
    back to the top on every single interaction. The two exceptions are
    the action buttons at the bottom: those still call st.rerun() (full
    scope, the default) because switching to a different step genuinely
    does need the rest of the page to re-render.
    """
    with st.expander("▶ Watch the clip first (real playback, drag to scrub)"):
        st.video(video_path)
        st.caption("This player supports real scrubbing and sound — use it to get oriented "
                   "(listen for the click of contact), then pinpoint the exact frame below.")

    if "impact_slider" not in st.session_state:
        suggested = st.session_state.get("suggested_impact_frame")
        st.session_state.impact_slider = suggested if suggested is not None else n_frames // 4

    if st.session_state.get("suggested_impact_frame") is not None:
        st.caption("🔊 Jumped to the loudest, sharpest sound in the clip — usually the strike. "
                   "Confirm it's the ball, and nudge with ◀ ▶ if it's off by a frame or two.")
    else:
        st.caption("No audio track found to estimate from — scrub manually below.")

    def _step_impact(delta):
        st.session_state.impact_slider = max(0, min(n_frames - 1, st.session_state.impact_slider + delta))

    col_prev, col_slider, col_next = st.columns([1, 8, 1])
    with col_prev:
        st.button("◀", key="impact_prev", on_click=_step_impact, args=(-1,), width="stretch")
    with col_slider:
        idx = st.slider("Scrub to the impact frame", 0, n_frames - 1, key="impact_slider",
                         label_visibility="collapsed")
    with col_next:
        st.button("▶", key="impact_next", on_click=_step_impact, args=(1,), width="stretch")

    # Draw a marker on a copy of the preview at the currently-set position
    # (if it's on this frame) so a misclick is immediately visible and you
    # can just tap again to move it. Tapping always overwrites, so
    # correcting a misclick is simply "tap the right spot."
    display_frame = frames[idx]
    if st.session_state.get("impact_frame") == idx and "impact_xy" in st.session_state:
        import numpy as _np
        display_frame = frames[idx].copy()
        mx = int(st.session_state.impact_xy[0] * scale)
        my = int(st.session_state.impact_xy[1] * scale)
        # simple crosshair marker (no cv2 needed on the preview array)
        h_p, w_p = display_frame.shape[:2]
        for dx in range(-8, 9):
            xx = mx + dx
            if 0 <= xx < w_p and 0 <= my < h_p:
                display_frame[my, xx] = [255, 60, 60]
        for dy in range(-8, 9):
            yy = my + dy
            if 0 <= mx < w_p and 0 <= yy < h_p:
                display_frame[yy, mx] = [255, 60, 60]

    st.caption("Tap the ball on the image. **Tapped the wrong spot? Just tap again** — the newest tap wins.")
    # Key includes the frame index on purpose: streamlit_image_coordinates
    # keeps returning its LAST value across reruns, so without this, moving
    # the slider to a new frame (without clicking) would silently keep the
    # old click's (x, y) but pair it with the new frame's index.
    coords = streamlit_image_coordinates(display_frame, key=f"impact_click_{idx}")
    if coords is not None:
        st.session_state.impact_frame = idx
        st.session_state.impact_xy = to_source_xy((coords["x"], coords["y"]), scale)
        st.rerun()  # redraw immediately so the marker moves to the new tap

    if "impact_xy" in st.session_state:
        ix, iy = st.session_state.impact_xy
        col_ok, col_clear = st.columns([3, 1])
        with col_ok:
            st.success(f"Impact set: frame {st.session_state.impact_frame} at ({ix:.0f}, {iy:.0f})")
        with col_clear:
            if st.button("↺ Clear", key="clear_impact", width="stretch"):
                del st.session_state["impact_xy"]
                if "impact_frame" in st.session_state:
                    del st.session_state["impact_frame"]
                st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        run_auto = st.button("▶ Track shot automatically", type="primary")
    with c2:
        skip_manual = st.button("Skip to manual mode")

    if run_auto:
        # No disabled= here on purpose -- a disabled button gives no
        # feedback about WHY it won't respond, and can be easy to miss
        # entirely depending on styling. An explicit message can't be
        # ambiguous either way.
        if "impact_xy" not in st.session_state:
            st.warning("Tap the ball's position on the image above first.")
        else:
            progress = st.progress(0.0, text="Tracking ball…")
            out_dir = tempfile.mkdtemp()
            result = process_video_auto(
                video_path, os.path.join(out_dir, "tracer_output.mp4"),
                st.session_state.impact_frame, st.session_state.impact_xy, color_bgr,
                progress_callback=lambda p: progress.progress(min(p, 1.0), text="Tracking ball…"),
            )
            if result["tracking_stopped"]:
                st.session_state.path_so_far = result["path_so_far"]
                st.session_state.stopped_at_frame = result["stopped_at_frame"]
                st.session_state.last_state = result.get("last_state")
                st.session_state.mode = "confirm_stop"
            else:
                st.session_state.output_path = result["output_path"]
                st.session_state.mode = "done"
            st.rerun()

    if skip_manual:
        st.session_state.mode = "manual_pick_points"
        st.rerun()


@st.fragment
def manual_picker_fragment(frames, scale, n_frames, color_bgr, video_path):
    """
    Rewritten to a clear one-point-at-a-time flow. The previous version
    used a radio to switch between launch/apex/landing, but tapping only
    recorded the CURRENTLY selected point and the Generate button just
    silently didn't appear until all three happened to be set -- easy to
    get stuck in, with no feedback about what was missing. Now each point
    is set explicitly with its own button, a running checklist shows what's
    done, and Generate is always visible (disabled with a reason until all
    three are in).
    """
    POINTS = ["launch", "apex", "landing"]
    HELP = {
        "launch": "the ball at the moment it leaves the club (impact)",
        "apex": "the highest point of the ball's flight",
        "landing": "where the ball lands or leaves the frame",
    }

    if "manual_active_point" not in st.session_state:
        st.session_state.manual_active_point = "launch"
    active = st.session_state.manual_active_point

    # Status checklist -- always visible so it's never unclear what's left.
    cols = st.columns(3)
    for col, p in zip(cols, POINTS):
        with col:
            if f"{p}_xy" in st.session_state:
                col.success(f"✓ {p.title()}\nframe {st.session_state[f'{p}_frame']}")
            elif p == active:
                col.info(f"● {p.title()}\n(setting now)")
            else:
                col.caption(f"○ {p.title()}\n(not set)")

    st.caption(f"**Setting {active.title()}** — {HELP[active]}. "
               f"Scrub to the right frame, then tap the ball. **Misclick? Just tap again.**")

    slider_key = f"slider_{active}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state.get(f"{active}_frame", n_frames // 4)

    def _step_point(delta):
        st.session_state[slider_key] = max(0, min(n_frames - 1, st.session_state[slider_key] + delta))

    col_prev, col_slider, col_next = st.columns([1, 8, 1])
    with col_prev:
        st.button("◀", key=f"prev_{active}", on_click=_step_point, args=(-1,), width="stretch")
    with col_slider:
        idx = st.slider("Scrub", 0, n_frames - 1, key=slider_key, label_visibility="collapsed")
    with col_next:
        st.button("▶", key=f"next_{active}", on_click=_step_point, args=(1,), width="stretch")

    # Show a marker where THIS point is currently set (if on this frame),
    # so a misclick is visible and correcting it is just another tap.
    disp = frames[idx]
    if st.session_state.get(f"{active}_frame") == idx and f"{active}_xy" in st.session_state:
        disp = frames[idx].copy()
        mx = int(st.session_state[f"{active}_xy"][0] * scale)
        my = int(st.session_state[f"{active}_xy"][1] * scale)
        h_p, w_p = disp.shape[:2]
        for dx in range(-8, 9):
            xx = mx + dx
            if 0 <= xx < w_p and 0 <= my < h_p:
                disp[my, xx] = [255, 60, 60]
        for dy in range(-8, 9):
            yy = my + dy
            if 0 <= mx < w_p and 0 <= yy < h_p:
                disp[yy, mx] = [255, 60, 60]

    coords = streamlit_image_coordinates(disp, key=f"click_{active}_{idx}")
    if coords is not None:
        st.session_state[f"{active}_frame"] = idx
        st.session_state[f"{active}_xy"] = to_source_xy((coords["x"], coords["y"]), scale)
        # Auto-advance to the next unset point so the flow moves forward
        # on its own after each tap, instead of leaving the person to
        # figure out they need to switch a control.
        for p in POINTS:
            if f"{p}_xy" not in st.session_state:
                st.session_state.manual_active_point = p
                break
        st.rerun()

    # Let the person jump back to re-set any already-placed point.
    set_points = [p for p in POINTS if f"{p}_xy" in st.session_state]
    if set_points:
        st.caption("Re-set a point:")
        recols = st.columns(len(set_points))
        for col, p in zip(recols, set_points):
            with col:
                if st.button(f"Edit {p.title()}", key=f"edit_{p}", width="stretch"):
                    st.session_state.manual_active_point = p
                    st.rerun()

    st.divider()
    ready = all(f"{p}_xy" in st.session_state for p in POINTS)
    if not ready:
        missing = [p.title() for p in POINTS if f"{p}_xy" not in st.session_state]
        st.info(f"Still need: {', '.join(missing)}. Tap each on the image to set it.")
    if st.button("▶ Generate tracer", type="primary", disabled=not ready):
        progress = st.progress(0.0, text="Rendering tracer…")
        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "tracer_output.mp4")
        process_video_manual(
            video_path, out_path,
            launch=(st.session_state["launch_frame"], st.session_state["launch_xy"]),
            apex=(st.session_state["apex_frame"], st.session_state["apex_xy"]),
            landing=(st.session_state["landing_frame"], st.session_state["landing_xy"]),
            color_bgr=color_bgr,
            progress_callback=lambda p: progress.progress(min(p, 1.0), text="Rendering tracer…"),
        )
        st.session_state.output_path = out_path
        st.session_state.mode = "done"
        st.rerun()  # full-scope: leaving manual mode needs the whole page


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
        meta = get_video_meta(video_path)
        suggested = detect_impact_frame(video_path, meta["fps"], len(frames))
    st.session_state.preview_frames = frames
    st.session_state.preview_scale = scale
    st.session_state.suggested_impact_frame = suggested
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
    impact_picker_fragment(st.session_state.video_path, frames, scale, n_frames, color_bgr)

# ---------------------------------------------------------- CONFIRM STOP
elif mode == "confirm_stop":
    stopped_at = st.session_state.stopped_at_frame
    meta = get_video_meta(st.session_state.video_path)
    fps = meta["fps"] or 30.0
    timestamp = stopped_at / fps
    st.warning(
        f"Automatic tracking stopped at frame {stopped_at} (about {timestamp:.1f} seconds into the clip). "
        f"This can mean the ball simply left the frame or landed (normal end of shot), "
        f"or that tracking lost it (clouds, motion blur, low contrast, etc)."
    )
    # Show the actual frame so "frame 657" is something you can SEE. Clamp
    # the index: stopped_at counts full-video frames, and the preview list
    # can differ in length, so an unclamped lookup here can crash.
    if frames:
        preview_idx = max(0, min(stopped_at, len(frames) - 1))
        st.image(frames[preview_idx],
                 caption=f"This is frame {stopped_at} — where tracking stopped.",
                 width="stretch")

    have_state = st.session_state.get("last_state") is not None

    st.markdown("**What would you like to do?**")
    if st.button("🎯 Estimate the rest of the flight (recommended)", type="primary",
                 disabled=not have_state):
        try:
            progress = st.progress(0.0, text="Estimating flight path…")
            out_dir = tempfile.mkdtemp()
            out_path = os.path.join(out_dir, "tracer_output.mp4")
            extrapolate_lost_flight(
                st.session_state.video_path, out_path,
                st.session_state.path_so_far, st.session_state.last_state, color_bgr,
                progress_callback=lambda p: progress.progress(max(0.0, min(p, 1.0)),
                                                              text="Estimating flight path…"),
            )
            st.session_state.output_path = out_path
            st.session_state.mode = "done"
            st.rerun()
        except Exception as e:
            # Surface the real reason in the UI instead of crashing to the
            # opaque "Oh no" screen -- and offer the other paths so the run
            # isn't a dead end.
            st.error(f"Couldn't finish the estimate: {e}")
            st.caption("You can still finish with one of the options below.")
    if not have_state:
        st.caption("Estimate unavailable — the ball was never tracked long enough to project a path. "
                   "Use manual mode below.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Stop the tracer here (ball landed / left frame)"):
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
        if st.button("✏️ Draw it myself (manual mode)"):
            st.session_state.mode = "manual_pick_points"
            st.rerun()

# --------------------------------------------------------- MANUAL MODE
elif mode == "manual_pick_points":
    st.subheader("Tap launch, apex, and landing")
    manual_picker_fragment(frames, scale, n_frames, color_bgr, st.session_state.video_path)

# --------------------------------------------------------------- DONE
elif mode == "done":
    st.subheader("Done! 🎉")
    st.video(st.session_state.output_path)
    with open(st.session_state.output_path, "rb") as f:
        st.download_button("⬇ Download tracer video", f, file_name="golf_tracer.mp4", mime="video/mp4")
    if st.button("Start over"):
        reset_state()
        st.rerun()
