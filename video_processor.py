"""
video_processor.py
-------------------
Frame-accurate video I/O, tracer rendering, and audio-preserving mux.

Two-pass design (deliberate)
-----------------------------
PASS 1 (`_build_ball_path_auto`, or the manual-mode point sampling in
`process_video_manual`) figures out the ball's pixel position for every
frame first, with NO rendering. Only once the *complete* path is known --
including which gaps need physics fill, and whether tracking made it to a
clean stopping point -- do we open a writer and render (PASS 2,
`render_path_over_video`). This is what makes clean gap-filling possible:
filling a gap correctly needs to know the re-acquisition point, which a
single streaming pass hasn't seen yet at the moment it first notices the
ball is missing. The cost is decoding the clip twice; for an offline
"record then process" tool (as opposed to a live camera feed) that's a
good trade for correctness.

Audio preservation
-------------------
OpenCV's VideoWriter re-encodes video frame-by-frame and has no audio
support at all, so we always render to a silent intermediate file first,
then shell out to ffmpeg ONCE to mux the ORIGINAL audio track back in.

The final encode explicitly targets libx264 + yuv420p: OpenCV's writer
(via the "mp4v" fourcc used for the intermediate file) produces MPEG-4 Part
2 video, which iOS Safari's <video> element will generally NOT play at all
-- Safari needs H.264 or HEVC. So this last ffmpeg pass isn't just a
quality nicety, it's what makes the output actually work on the target
device.
"""

import os
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ball_tracker import BallDetector, KalmanBallTracker
from physics_model import Point2D, ProjectileGapFiller, ManualBezierFallback

MAX_COAST_FRAMES = 45              # ~0.75-1.5s depending on fps before we give up on a gap


def get_video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


class TracerRenderer:
    """Draws the growing tracer onto a persistent semi-transparent overlay
    and alpha-blends it onto each frame (cv2.LINE_AA + cv2.addWeighted, per
    spec). Only pixels the overlay has actually touched get blended, so the
    rest of the frame stays exactly as sharp as the source."""

    def __init__(self, frame_shape: Tuple[int, int, int], color_bgr: Tuple[int, int, int],
                 thickness: int = 5, alpha: float = 0.65):
        self.overlay = np.zeros(frame_shape, dtype=np.uint8)
        self.color = color_bgr
        self.thickness = thickness
        self.alpha = alpha
        self._last_point: Optional[Tuple[int, int]] = None

    def add_point(self, x: float, y: float):
        pt = (int(round(x)), int(round(y)))
        if self._last_point is not None:
            # Only ever draw line SEGMENTS between successive points -- no
            # standalone dot on the first point. A dot drawn at the very
            # first point sits on the ball while it's still essentially at
            # rest (the address/impact position), which reads as a blob
            # stuck to the tee before the shot. Waiting for a second point
            # means the tracer visibly begins only once the ball has
            # actually moved -- i.e. as it starts flying.
            cv2.line(self.overlay, self._last_point, pt, self.color, self.thickness, lineType=cv2.LINE_AA)
        self._last_point = pt

    def render(self, frame: np.ndarray) -> np.ndarray:
        blended = cv2.addWeighted(self.overlay, self.alpha, frame, 1.0 - self.alpha, 0)
        mask = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2GRAY) > 0
        out = frame.copy()
        out[mask] = blended[mask]
        return out


def mux_audio(silent_video_path: str, source_with_audio_path: str, output_path: str, crf: int = 16) -> None:
    """Stream-copies nothing on purpose: re-encodes video to libx264/yuv420p
    (Safari-safe) while pulling audio from the ORIGINAL source clip, trimmed
    to the shorter of the two streams so nothing drifts out of sync."""
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", source_with_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[-2000:]}")


def _open_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # intermediate only -- re-encoded to H.264 in mux_audio
    return cv2.VideoWriter(path, fourcc, fps, size)


def _build_ball_path_auto(source_path: str, impact_frame_idx: int, impact_xy: Tuple[float, float],
                           progress_callback=None) -> Tuple[List[Optional[Point2D]], Optional[int]]:
    """
    PASS 1. Returns (path, stopped_at_frame).

    stopped_at_frame is None if tracking made it cleanly to the end of the
    clip. Otherwise it's the frame index where tracking gave up -- which
    means Fail-Safe 2 should be OFFERED to the user. Note this could mean
    either a genuine tracking failure, OR simply the ball having left frame
    or landed. CV alone can't reliably tell those apart, so we surface the
    choice in the UI instead of guessing (see app.py's "confirm_stop" mode).
    """
    meta = get_video_meta(source_path)
    total_frames = meta["frame_count"]

    cap = cv2.VideoCapture(source_path)
    detector = BallDetector()
    tracker = KalmanBallTracker(dt=1.0)
    gap_filler = ProjectileGapFiller()

    path: List[Optional[Point2D]] = [None] * total_frames
    frame_idx = 0
    gap_start: Optional[int] = None
    last_good_point: Optional[Point2D] = None
    last_good_velocity: Optional[Tuple[float, float]] = None
    stopped_at_frame: Optional[int] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < impact_frame_idx:
            frame_idx += 1
            continue

        if frame_idx == impact_frame_idx:
            tracker.initialize(*impact_xy)
            path[frame_idx] = Point2D(*impact_xy)
            last_good_point = path[frame_idx]
            frame_idx += 1
            if progress_callback:
                progress_callback(0.5 * frame_idx / total_frames)
            continue

        predicted_xy = tracker.predict()
        if last_good_velocity is None:
            # No real detection since impact yet, so there's no velocity
            # estimate to size the search window from -- and the true
            # moment of contact virtually never lines up exactly with a
            # frame boundary. If impact actually happened a fraction of a
            # frame before this one, the ball may already have covered a
            # large distance by now (worse at 30fps than 60fps), and a
            # search window sized for "presumably still near the impact
            # point" would wrongly reject it. Search generously for this
            # first attempt specifically; every attempt after this one has
            # a real velocity estimate to size the window from instead.
            h, w = frame.shape[:2]
            search_radius = 0.5 * max(w, h)
        else:
            speed = float(np.hypot(*tracker.current_velocity))
            search_radius = max(160.0, speed * 1.5)  # widen the window for genuinely fast shots
        measured_xy = detector.detect(frame, predicted_xy=predicted_xy, search_radius=search_radius)

        if measured_xy is not None:
            corrected_xy = tracker.correct(*measured_xy)
            path[frame_idx] = Point2D(*corrected_xy)

            if gap_start is not None:
                # Fill the gap regardless of how "established" the track
                # was beforehand. Even with a bare (or zero) velocity
                # estimate, the ease-in correction in fill_gap guarantees
                # the filled path lands exactly on this real re-acquisition
                # point -- so a shakier initial guess just means the SHAPE
                # of the fill is less refined, not that it ends up wrong.
                # This matters a lot right after impact specifically: the
                # clubhead/follow-through commonly occludes the ball for a
                # handful of frames before a real lock is established, and
                # treating that as an instant failure (the old behavior)
                # was worse than a slightly-rougher gap-fill.
                gravity_px = max(tracker.current_acceleration[1], 4.0)
                gap_len = frame_idx - gap_start
                filled = gap_filler.fill_gap(
                    p_start=last_good_point, v_start=last_good_velocity or (0.0, 0.0),
                    n_frames=gap_len, p_end=path[frame_idx], gravity_px=gravity_px,
                )
                for i, pt in enumerate(filled):
                    path[gap_start + i] = pt
                gap_start = None

            last_good_point = path[frame_idx]
            last_good_velocity = tracker.current_velocity
        else:
            tracker.register_miss()
            if gap_start is None:
                gap_start = frame_idx

            gap_len = frame_idx - gap_start + 1
            if gap_len > MAX_COAST_FRAMES:
                stopped_at_frame = gap_start
                break

        frame_idx += 1
        if progress_callback:
            progress_callback(0.5 * frame_idx / total_frames)

    cap.release()
    return path, stopped_at_frame


def freeze_path_from(path: List[Optional[Point2D]], freeze_at_frame: int) -> List[Optional[Point2D]]:
    """Use when the user confirms the shot legitimately ended where
    auto-tracking stopped (ball landed / left frame): holds the tracer at
    its last known point for the remainder of the clip instead of leaving
    it undrawn."""
    last_known = None
    for p in path[:freeze_at_frame]:
        if p is not None:
            last_known = p
    frozen = list(path)
    for i in range(freeze_at_frame, len(frozen)):
        if frozen[i] is None:
            frozen[i] = last_known
    return frozen


def render_path_over_video(source_path: str, output_path: str, path: List[Optional[Point2D]],
                            color_bgr: Tuple[int, int, int], progress_callback=None) -> None:
    """PASS 2: walks the already-complete path, draws it frame by frame,
    then muxes the original audio back in."""
    meta = get_video_meta(source_path)
    fps, w, h = meta["fps"], meta["width"], meta["height"]
    total_frames = meta["frame_count"]
    thickness = max(4, w // 250)  # scale line weight to resolution instead of a fixed pixel width

    cap = cv2.VideoCapture(source_path)
    tmp_silent_path = output_path + ".silent.mp4"
    writer = _open_writer(tmp_silent_path, fps, (w, h))
    renderer = TracerRenderer((h, w, 3), color_bgr, thickness=thickness)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < len(path) and path[frame_idx] is not None:
            renderer.add_point(path[frame_idx].x, path[frame_idx].y)
        writer.write(renderer.render(frame))
        frame_idx += 1
        if progress_callback:
            progress_callback(0.5 + 0.5 * frame_idx / total_frames)

    cap.release()
    writer.release()
    mux_audio(tmp_silent_path, source_path, output_path)
    os.remove(tmp_silent_path)


def process_video_auto(source_path: str, output_path: str, impact_frame_idx: int,
                        impact_xy: Tuple[float, float], color_bgr: Tuple[int, int, int],
                        progress_callback=None) -> dict:
    """Fail-Safe 1 pipeline: live CV + Kalman tracking with physics-based
    gap fill. Returns either a finished output_path, or a request for the
    caller to resolve an early stop (see `freeze_path_from` / Fail-Safe 2)."""
    path, stopped_at_frame = _build_ball_path_auto(source_path, impact_frame_idx, impact_xy, progress_callback)

    if stopped_at_frame is not None:
        return {"tracking_stopped": True, "stopped_at_frame": stopped_at_frame, "path_so_far": path}

    render_path_over_video(source_path, output_path, path, color_bgr, progress_callback)
    return {"tracking_stopped": False, "output_path": output_path}


def process_video_manual(source_path: str, output_path: str,
                          launch: Tuple[int, Tuple[float, float]], apex: Tuple[int, Tuple[float, float]],
                          landing: Tuple[int, Tuple[float, float]], color_bgr: Tuple[int, int, int],
                          progress_callback=None) -> None:
    """Fail-Safe 2 pipeline: corrected-control-point quadratic Bezier
    across [launch_frame, landing_frame], frame-accurate."""
    meta = get_video_meta(source_path)
    total_frames = meta["frame_count"]
    launch_frame, launch_xy = launch
    _apex_frame, apex_xy = apex
    landing_frame, landing_xy = landing

    bez = ManualBezierFallback(Point2D(*launch_xy), Point2D(*apex_xy), Point2D(*landing_xy))
    n_flight_frames = max(landing_frame - launch_frame + 1, 2)
    flight_path = bez.sample(n_flight_frames)

    path: List[Optional[Point2D]] = [None] * total_frames
    for i, pt in enumerate(flight_path):
        idx = launch_frame + i
        if idx < total_frames:
            path[idx] = pt

    render_path_over_video(source_path, output_path, path, color_bgr, progress_callback)
