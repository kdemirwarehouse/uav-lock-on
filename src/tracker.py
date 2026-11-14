"""Tracker module: hybrid detection + tracking for a single target,
with temporal class voting."""

import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return (int(x), int(y), int(x + w), int(y + h))


def _box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def _center_distance(a, b):
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                      (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


def _create_tracker(tracker_type):
    if tracker_type == "csrt":
        return cv2.TrackerCSRT_create()
    if tracker_type == "kcf":
        return cv2.TrackerKCF_create()
    raise ValueError(f"Unknown tracker type: {tracker_type}")


@dataclass
class TrackResult:
    box: tuple          # (x1, y1, x2, y2)
    source: str         # "detector" or "tracker"
    raw_class: str      # class from this frame's detection (None on tracker frames)
    voted_class: str    # majority class over recent detections


class HybridTracker:
    def __init__(self, detector, tracker_type="kcf",
                 detect_every_n_frames=5, vote_window=10,
                 reset_after_missed_frames=15):
        self.detector = detector
        self.tracker_type = tracker_type
        self.detect_every_n = detect_every_n_frames
        self.reset_after_missed = reset_after_missed_frames
        self.tracker = None
        self.frame_idx = 0
        # Remembers only the last `vote_window` classes, older ones drop out
        self.class_history = deque(maxlen=vote_window)
        self.missed_frames = 0      # consecutive frames with no target at all
        self.last_box = None        # last known target box, for re-acquisition

    def _voted_class(self):
        return Counter(self.class_history).most_common(1)[0][0]

    def _forget_target(self):
        """Target is really gone: drop its votes too."""
        self.class_history.clear()
        self.last_box = None

    def _is_new_target(self, box, frame):
        # After a loss, a detection far from where the target was is most
        # likely a different object -> its class must not inherit old votes.
        if self.last_box is None:
            return False
        h, w = frame.shape[:2]
        return _center_distance(box, self.last_box) > 0.5 * math.hypot(w, h)

    def update(self, frame):
        """Return a TrackResult, or None if there is no target."""
        # Run YOLO every N frames, or every frame while we have no target
        run_detector = (self.frame_idx % self.detect_every_n == 0) or self.tracker is None
        self.frame_idx += 1

        if run_detector:
            detections = self.detector.detect(frame)
            if detections:
                # Rule: follow the largest target (most likely the closest)
                target = max(detections, key=lambda d: _box_area(d.box))
                if self.tracker is None and self._is_new_target(target.box, frame):
                    self._forget_target()
                # Re-initialise tracker -> clears accumulated drift
                self.tracker = _create_tracker(self.tracker_type)
                self.tracker.init(frame, _xyxy_to_xywh(target.box))
                self.class_history.append(target.cls_name)
                self.missed_frames = 0
                self.last_box = target.box
                return TrackResult(target.box, "detector",
                                   target.cls_name, self._voted_class())
            # Detector found nothing: let the tracker try

        if self.tracker is not None:
            ok, box = self.tracker.update(frame)
            if ok:
                box = _xywh_to_xyxy(box)
                self.missed_frames = 0
                self.last_box = box
                return TrackResult(box, "tracker", None, self._voted_class())
            # A single KCF failure is common (blur, fast motion). Drop only the
            # tracker so YOLO runs next frame, but keep the votes: wiping them
            # here made the class flip right after a one-frame loss.
            self.tracker = None

        self.missed_frames += 1
        if self.missed_frames >= self.reset_after_missed:
            self._forget_target()
        return None


# Quick standalone test: python src/tracker.py [video_path]
if __name__ == "__main__":
    import sys

    from detector import Detector
    from source import VideoSource

    video = sys.argv[1] if len(sys.argv) > 1 else "videos/test.mp4"
    det = Detector("weights/best.pt", conf_threshold=0.25,
                   target_classes=["drone", "aircraft", "helicopter"])
    trk = HybridTracker(det, tracker_type="kcf", detect_every_n_frames=5)
    src = VideoSource(video)

    cv2.namedWindow("tracker test", cv2.WINDOW_NORMAL)   # resizable window
    while True:
        frame = src.read()
        if frame is None:
            break

        res = trk.update(frame)
        if res:
            # green = YOLO, yellow = tracker
            color = (0, 255, 0) if res.source == "detector" else (0, 255, 255)
            cv2.rectangle(frame, res.box[:2], res.box[2:], color, 3)
            label = f"vote: {res.voted_class}"
            if res.raw_class:
                label += f" | raw: {res.raw_class}"
            cv2.putText(frame, label, (res.box[0], res.box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("tracker test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    cv2.destroyAllWindows()