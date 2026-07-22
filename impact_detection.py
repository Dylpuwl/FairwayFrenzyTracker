"""
impact_detection.py
---------------------
Estimates the impact frame from the AUDIO track rather than asking the
user to hunt for it visually. The club-ball contact is typically the
loudest, sharpest sound in a swing recording -- much easier and more
reliable to pinpoint than trying to spot contact in a single video frame,
especially at 30/60 fps where the ball may still be partly obscured by the
club or hands right at the moment of impact.

This is a SUGGESTION used to pre-position the frame picker, not a silent
decision -- the UI always shows it as an estimate and lets the person
confirm or nudge it, because ambient noise, wind, or talking can
occasionally produce a sharper transient than the actual strike.

KNOWN SYSTEMATIC BIAS: sound travels far slower than light. At a typical
filming distance (a few meters), the audio transient arrives some
milliseconds after the true visual moment of contact -- often under one
frame at 30/60fps, but it depends on how far back the camera was set up,
which we don't know. We don't attempt to correct for this since the
correction size is unknown without a real distance estimate; it's part of
why this is offered as an adjustable suggestion rather than an automatic
final answer.
"""

import subprocess
from typing import Optional

import numpy as np

AUDIO_SAMPLE_RATE = 22050
ONSET_WINDOW_SECONDS = 0.015  # ~15ms -- comparable to one video frame at 60fps
SEARCH_MARGIN_FRACTION = 0.05  # ignore the first/last 5% of the clip (handling noise, camera start/stop)


def _extract_mono_audio(video_path: str) -> Optional[np.ndarray]:
    """Returns a mono float32 waveform at AUDIO_SAMPLE_RATE, or None if the
    clip has no audio track or ffmpeg fails for any other reason."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-f", "s16le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)


def detect_impact_frame(video_path: str, fps: float, total_frames: int) -> Optional[int]:
    """
    Returns a best-guess impact frame index, or None if detection wasn't
    possible (no audio track, audio too short, ffmpeg unavailable/failed).
    """
    audio = _extract_mono_audio(video_path)
    if audio is None or len(audio) < AUDIO_SAMPLE_RATE // 10:
        return None

    # Crude high-pass filter: a first difference emphasizes sudden
    # transients (a sharp click) over slower-varying sound (wind rumble,
    # a steady tone, talking) -- deliberately simple, no scipy dependency.
    high_passed = np.diff(audio)

    window = max(int(AUDIO_SAMPLE_RATE * ONSET_WINDOW_SECONDS), 1)
    n_windows = len(high_passed) // window
    if n_windows < 4:
        return None

    trimmed = high_passed[: n_windows * window].reshape(n_windows, window)
    energy = np.sum(trimmed ** 2, axis=1)

    # Onset strength: how sharply energy JUMPS between consecutive windows,
    # not just which window is loudest -- a sudden spike reads as a strike
    # more reliably than raw loudness would (a shout or wind gust can be
    # loud without being sudden).
    onset = np.diff(energy, prepend=energy[0])

    margin = max(int(n_windows * SEARCH_MARGIN_FRACTION), 1)
    search_slice = onset[margin: n_windows - margin]
    if len(search_slice) == 0:
        return None

    peak_window = margin + int(np.argmax(search_slice))
    peak_time_sec = peak_window * window / AUDIO_SAMPLE_RATE
    frame_idx = int(round(peak_time_sec * fps))
    return max(0, min(frame_idx, total_frames - 1))
