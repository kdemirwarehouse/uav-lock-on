"""Lock manager: state machine that decides when a target is locked."""

from dataclasses import dataclass
from enum import Enum


class LockState(Enum):
    SEARCHING = "ARANIYOR"
    TARGET = "HEDEF VAR"
    LOCKING = "KILITLENIYOR"
    LOCKED = "KILITLENDI"


@dataclass
class LockEvent:
    start_s: float      # video time when locking started (seconds)
    end_s: float        # video time when lock was confirmed
    cls_name: str
    box: tuple


class LockManager:
    def __init__(self, frame_w, frame_h, fps, target_classes, region,
                 min_box_ratio=0.05, lock_duration_s=4.0, max_lost_frames=5):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.target_classes = set(target_classes)
        # Region from config is in fractions -> convert to pixels once
        self.region = (
            int(region["x_min"] * frame_w), int(region["y_min"] * frame_h),
            int(region["x_max"] * frame_w), int(region["y_max"] * frame_h),
        )
        self.min_box_ratio = min_box_ratio
        self.required_frames = int(round(lock_duration_s * fps))   # 4 s x 30 FPS = 120
        self.max_lost_frames = max_lost_frames

        self.state = LockState.SEARCHING
        self.frame_idx = 0
        self.lock_frames = 0          # frames that met all conditions
        self.lost_frames = 0          # frames since conditions last held
        self.lock_start_frame = None

    def _conditions_met(self, result):
        # 1) Class: voted class must be a lock target
        if result.voted_class not in self.target_classes:
            return False
        # 2) Region: whole box inside the lock rectangle
        x1, y1, x2, y2 = result.box
        rx1, ry1, rx2, ry2 = self.region
        if not (x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2):
            return False
        # 3) Size: box width or height covers enough of the frame
        w_ratio = (x2 - x1) / self.frame_w
        h_ratio = (y2 - y1) / self.frame_h
        return max(w_ratio, h_ratio) >= self.min_box_ratio

    def update(self, result):
        """Feed one TrackResult (or None) per frame.
        Returns a LockEvent the moment a lock is confirmed, otherwise None."""
        self.frame_idx += 1
        event = None

        if result is not None and self._conditions_met(result):
            if self.lock_frames == 0:
                self.lock_start_frame = self.frame_idx
            self.lock_frames += 1
            self.lost_frames = 0

            if self.lock_frames >= self.required_frames:
                if self.state != LockState.LOCKED:      # fire event only once
                    event = LockEvent(
                        start_s=self.lock_start_frame / self.fps,
                        end_s=self.frame_idx / self.fps,
                        cls_name=result.voted_class,
                        box=result.box,
                    )
                self.state = LockState.LOCKED
            else:
                self.state = LockState.LOCKING
        else:
            self.lost_frames += 1
            # Reset only if there was no progress, or tolerance is exceeded
            if self.lock_frames == 0 or self.lost_frames > self.max_lost_frames:
                self.lock_frames = 0
                self.state = LockState.TARGET if result is not None else LockState.SEARCHING
            # Within tolerance: keep state and counter, just don't count this frame

        return event

    @property
    def progress(self):
        """Lock progress between 0.0 and 1.0 (for drawing a progress bar)."""
        return min(self.lock_frames / self.required_frames, 1.0)


# Logic test WITHOUT any video: python src/lock_manager.py
if __name__ == "__main__":
    from tracker import TrackResult

    lm = LockManager(
        frame_w=640, frame_h=480, fps=30,
        target_classes=["drone", "aircraft"],
        region={"x_min": 0.25, "y_min": 0.10, "x_max": 0.75, "y_max": 0.90},
    )
    center = TrackResult((280, 200, 360, 260), "detector", "drone", "drone")
    corner = TrackResult((10, 10, 90, 70), "detector", "drone", "drone")

    # Scenario
    #   frames   1-30 : no target                  -> ARANIYOR
    #   frames  31-60 : target outside region      -> HEDEF VAR
    #   frames 61-200 : target in center           -> KILITLENIYOR -> KILITLENDI
    #   frames 100-102: target lost for 3 frames   (within tolerance)
    prev_state = None
    for f in range(1, 201):
        if f <= 30:
            res = None
        elif f <= 60:
            res = corner
        elif 100 <= f <= 102:
            res = None
        else:
            res = center

        event = lm.update(res)
        if lm.state != prev_state:
            print(f"frame {f:3d}: {lm.state.value}")
            prev_state = lm.state
        if event:
            print(f"  >>> LOCK EVENT: {event}")