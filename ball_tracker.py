"""
ball_tracker.py
----------------
Computer-vision tracking layer for the Golf Shot Tracer.

Responsibilities
----------------
1. Detect the golf ball in a single frame (classical CV pipeline by default;
   a YOLOv8-nano detector is available as a drop-in replacement IF you have
   a model fine-tuned on golf balls -- stock COCO weights are not reliable
   here, see YoloBallDetector's docstring).
2. Maintain a Kalman Filter (constant-ACCELERATION model, not the more
   common constant-velocity model) that:
     - smooths noisy detections,
     - predicts the ball's position while detection is briefly lost, and
     - naturally learns gravity + drag from real detections once it has
       locked onto a genuine trajectory, because acceleration is part of
       the state vector.

Why constant-acceleration, not constant-velocity
-------------------------------------------------
At 30/60 fps a golf ball can cover roughly 0.5-2m *between consecutive
frames*, with heavy motion blur. A constant-velocity Kalman filter has no
way to represent gravity, so every predicted frame during a gap will
systematically undershoot the ball's actual (curving) position. Putting
acceleration in the state lets the filter pick up the downward pull (ay)
and aerodynamic deceleration (a slow decay in vx) directly from the first
5-10 real detections after impact -- which is exactly the "successfully
tracked for 5-10 frames" precondition the spec calls out for Fail-Safe 1.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple


class TrackStatus(Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"


class KalmanBallTracker:
    """State vector: [x, y, vx, vy, ax, ay]. Units are pixels / frame (not
    seconds) -- dt=1.0 means "one frame per step", which keeps every
    downstream constant (gravity_px, drag_coeff, search radius) in units
    you can directly eyeball against your footage (pixels per frame)."""

    def __init__(self, dt: float = 1.0, process_noise: float = 5.0, measurement_noise: float = 4.0):
        self.kf = cv2.KalmanFilter(6, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self._initialized = False
        self.consecutive_detections = 0

    def initialize(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self._initialized = True
        self.consecutive_detections = 1

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def predict(self) -> Tuple[float, float]:
        # NOTE: pred[0]/pred[1] are shape-(1,) arrays, not 0-d scalars --
        # float(...) on those raises under numpy>=2.0, hence .item().
        pred = self.kf.predict()
        return pred[0].item(), pred[1].item()

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kf.correct(measurement)
        self.consecutive_detections += 1
        return corrected[0].item(), corrected[1].item()

    def register_miss(self) -> int:
        """Returns the lock streak that existed *before* this miss, so the
        caller can decide whether a physics-based gap-fill is trustworthy
        (spec: only after 5-10 confirmed frames) before it gets reset."""
        streak = self.consecutive_detections
        self.consecutive_detections = 0
        return streak

    @property
    def current_velocity(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return s[2].item(), s[3].item()

    @property
    def current_acceleration(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return s[4].item(), s[5].item()


class BallDetector:
    """
    Classical CV detector: background subtraction (MOG2) -> contour
    extraction -> circularity / size / brightness filtering -> pick the
    candidate closest to the Kalman filter's predicted position.

    ASSUMES A MOSTLY STATIC CAMERA (tripod). Background subtraction breaks
    down on handheld footage because the whole frame "moves" every frame,
    which MOG2 reads as foreground everywhere. If you must shoot handheld,
    run `stabilize_frame` (below) on each frame before detection -- off by
    default because ECC alignment costs real per-frame CPU time.
    """

    def __init__(self, min_radius: float = 2.0, max_radius: float = 20.0, min_circularity: float = 0.6):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity

    def warm_up(self, frame: np.ndarray, exclude_point: Optional[Tuple[float, float]] = None,
                exclude_radius: int = 25):
        """Feed a frame to the background model without attempting a
        detection. Call this on the pre-impact frames (address, waggle) so
        MOG2 has already converged by the time real tracking starts.

        If `exclude_point` is given (the user-tapped impact position), a
        small disk around it is inpainted out before learning. Without
        this, a ball sitting motionless at address for the pre-impact
        frames gets absorbed into the learned background like any other
        static object -- and the very first frame after impact then shows
        a smeared, non-circular blob (the vacated spot plus the moving
        ball) instead of a clean circle, right when detection matters most.
        """
        if exclude_point is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (int(exclude_point[0]), int(exclude_point[1])), exclude_radius, 255, -1)
            frame = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
        self.bg_subtractor.apply(frame, learningRate=0.01)

    def detect(self, frame: np.ndarray, predicted_xy: Optional[Tuple[float, float]] = None,
               search_radius: float = 160.0) -> Optional[Tuple[float, float]]:
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.01)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if not (self.min_radius <= radius <= self.max_radius):
                continue
            if circularity < self.min_circularity:
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), max(int(radius), 1), 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            if mean_val < 140:  # golf balls are bright / near-white
                continue

            candidates.append((cx, cy, circularity))

        if not candidates:
            return None

        if predicted_xy is not None:
            px, py = predicted_xy
            candidates = [c for c in candidates if np.hypot(c[0] - px, c[1] - py) <= search_radius]
            if not candidates:
                return None
            candidates.sort(key=lambda c: np.hypot(c[0] - px, c[1] - py))
        else:
            candidates.sort(key=lambda c: -c[2])

        cx, cy, _ = candidates[0]
        return float(cx), float(cy)


class YoloBallDetector:
    """
    Optional drop-in replacement for BallDetector. Requires a YOLOv8-nano
    model FINE-TUNED on golf balls -- stock COCO weights only know the
    generic "sports ball" class and will not reliably find a small, fast,
    motion-blurred golf ball. To use:

        pip install ultralytics
        detector = YoloBallDetector("golf_ball_yolov8n.pt")

    Same .detect(frame, predicted_xy, search_radius) interface as
    BallDetector, so video_processor.py doesn't care which one is plugged in.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        from ultralytics import YOLO  # heavy optional dependency, imported lazily
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray, predicted_xy: Optional[Tuple[float, float]] = None,
               search_radius: float = 160.0) -> Optional[Tuple[float, float]]:
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        if len(results.boxes) == 0:
            return None
        boxes = results.boxes.xywh.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        if predicted_xy is not None:
            px, py = predicted_xy
            dists = np.hypot(boxes[:, 0] - px, boxes[:, 1] - py)
            valid = np.where(dists <= search_radius)[0]
            if len(valid) == 0:
                return None
            idx = valid[np.argmin(dists[valid])]
        else:
            idx = int(np.argmax(confs))

        return float(boxes[idx, 0]), float(boxes[idx, 1])


def stabilize_frame(reference_gray: np.ndarray, frame_gray: np.ndarray) -> np.ndarray:
    """
    OPTIONAL: estimates the affine transform aligning frame_gray to
    reference_gray (ECC algorithm). Wire this into your loop for handheld
    footage by warping each frame toward the reference before background
    subtraction. Skipped by default -- tripod footage doesn't need it, and
    ECC alignment is not free (a few ms per frame at typical resolutions).
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(reference_gray, frame_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        pass
    return warp_matrix
