"""
physics_model.py
-----------------
Fail-Safe 1 (mid-flight gap fill) and Fail-Safe 2 (manual 3-tap fallback).

READ THIS FIRST: the monocular-camera limitation
--------------------------------------------------
A single iPhone camera has no depth channel. Without a second camera angle
or a calibrated focal length / camera height / tilt, there is no way to
recover true launch speed, spin rate, spin axis, or real-world distance
from 2D pixel data alone. Everything below is therefore a *shape model*:
physically-motivated curvature (gravity, drag, a simplified lift term)
fit to the pixel-space points you actually have -- not a first-principles
ballistic simulation from raw launch conditions. That's the same practical
shortcut broadcast golf tracer graphics use, and it's the right level of
sophistication for a phone-camera prototype. `Projectile3D` at the bottom
shows where a true 3D simulation would plug in if you add real camera
calibration later.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Point2D:
    x: float
    y: float


class ProjectileGapFiller:
    """
    Forward-simulates a physics-shaped path from (p_start, v_start), then --
    if the ball was re-acquired at p_end -- smoothly corrects onto p_end
    with an ease-in curve. This keeps the gap-fill looking like a real golf
    shot throughout, while still landing exactly on the true re-acquisition
    point with no visible "pop" when detection resumes.
    """

    def __init__(self, drag_coeff: float = 0.4, lift_coeff: float = 0.15):
        self.drag_coeff = drag_coeff  # % of horizontal velocity shed per frame (air resistance)
        self.lift_coeff = lift_coeff  # strength of the perpendicular "hold" term (simplified Magnus effect)

    def fill_gap(self, p_start: Point2D, v_start: Tuple[float, float], n_frames: int,
                 p_end: Optional[Point2D] = None, gravity_px: float = 6.0) -> List[Point2D]:
        """
        Returns exactly n_frames points to fill between (exclusive of)
        p_start and, if given, landing on p_end.

        gravity_px should ideally come from the Kalman filter's own `ay`
        state once it has locked on (see video_processor.py) -- that is
        gravity as this specific clip's camera actually observes it
        (distance-to-subject and lens FOV both scale how many pixels of
        "fall" gravity produces per frame), which is far more accurate
        than any fixed constant.

        DELIBERATE TRADE-OFF: the correction "ease" below reaches 1.0
        exactly at the last filled frame, so that frame lands exactly on
        p_end -- which, by construction, is also the position of the very
        next (real) frame. That means the ball appears to pause for one
        extra frame right at reacquisition. We chose this over easing to
        <1.0 and leaving a residual gap to close on the real detection:
        a 1-frame pause reads as a tiny hitch, whereas a residual gap can
        be tens of pixels if the raw physics guess and the true endpoint
        disagree a lot -- which shows up as a visible "snap" in the tracer
        line. A snap is worse than a hitch.
        """
        if n_frames <= 0:
            return []

        forward = self._simulate(p_start, v_start, n_frames, gravity_px)
        if p_end is None:
            return forward

        error_x = p_end.x - forward[-1].x
        error_y = p_end.y - forward[-1].y
        corrected = []
        for i, pt in enumerate(forward):
            t = (i + 1) / n_frames
            ease = t * t  # negligible correction right after p_start, full by p_end
            corrected.append(Point2D(pt.x + error_x * ease, pt.y + error_y * ease))
        return corrected

    def _simulate(self, p0: Point2D, v0: Tuple[float, float], n_frames: int, gravity_px: float) -> List[Point2D]:
        pts = []
        x, y = p0.x, p0.y
        vx, vy = v0
        for _ in range(n_frames):
            vx *= (1.0 - self.drag_coeff / 100.0)
            vy += gravity_px
            lift = -self.lift_coeff * vx * 0.05
            x += vx
            y += vy + lift
            pts.append(Point2D(x, y))
        return pts


class ManualBezierFallback:
    """
    Fail-Safe 2. Builds a curve through THREE user-tapped points (launch,
    apex, landing).

    IMPORTANT: a naive quadratic Bezier B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    does NOT pass through P1 -- P1 only pulls the curve toward it. Plugging
    the tapped apex straight in as P1 makes the rendered curve pass visibly
    *below* the point the user actually tapped. We instead solve for the
    control point that makes the curve pass exactly through the tapped apex
    at t=0.5:

        B(0.5) = 0.25 P0 + 0.5 P1 + 0.25 P2 = apex
        =>  P1  = 2*apex - 0.5*(P0 + P2)

    (A 3-point quadratic polynomial fit, y = ax^2+bx+c solved for all three
    taps directly, is a mathematically equivalent alternative to this
    corrected Bezier -- either passes through all three points exactly.)
    """

    def __init__(self, launch: Point2D, apex: Point2D, landing: Point2D):
        self.p0 = launch
        self.p2 = landing
        self.p1 = Point2D(
            2 * apex.x - 0.5 * (launch.x + landing.x),
            2 * apex.y - 0.5 * (launch.y + landing.y),
        )

    def sample(self, n_frames: int) -> List[Point2D]:
        pts = []
        n = max(n_frames, 2)
        for i in range(n_frames):
            t = i / (n - 1)
            x = (1 - t) ** 2 * self.p0.x + 2 * (1 - t) * t * self.p1.x + t ** 2 * self.p2.x
            y = (1 - t) ** 2 * self.p0.y + 2 * (1 - t) * t * self.p1.y + t ** 2 * self.p2.y
            pts.append(Point2D(x, y))
        return pts


class Projectile3D:
    """
    Upgrade path, intentionally left unimplemented: a real 3D ballistic
    simulation (gravity + drag + Magnus lift from backspin) re-projected
    into the image plane via a pinhole camera model, IF you supply real
    calibration -- camera height/tilt, distance to golfer, focal length
    (from EXIF or a known FOV for the lens used). Without those inputs the
    extra "realism" would be unmoored from the actual footage, so it's
    flagged here as the extension point rather than faked.
    """
    pass
