"""Tracker module: hybrid detection + tracking for a single target."""

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


def _create_tracker(tracker_type):
    if tracker_type == "csrt":
        return cv2.TrackerCSRT_create()
    if tracker_type == "kcf":
        return cv2.TrackerKCF_create()
    raise ValueError(f"Unknown tracker type: {tracker_type}")


class HybridTracker:
    def __init__(self, detector, tracker_type="csrt", detect_every_n_frames=5):
        self.detector = detector
        self.tracker_type = tracker_type
        self.detect_every_n = detect_every_n_frames
        self.tracker = None
        self.frame_idx = 0

    def update(self, frame):
        """Return (box, source) for the target, or (None, None) if lost.
        box is (x1, y1, x2, y2); source is "detector" or "tracker"."""
        # Run YOLO every N frames, or every frame while we have no target
        run_detector = (self.frame_idx % self.detect_every_n == 0) or self.tracker is None
        self.frame_idx += 1

        if run_detector:
            detections = self.detector.detect(frame)
            if detections:
                # Rule: follow the largest target (most likely the closest)
                target = max(detections, key=lambda d: _box_area(d.box))
                # Re-initialise tracker -> clears accumulated drift
                self.tracker = _create_tracker(self.tracker_type)
                self.tracker.init(frame, _xyxy_to_xywh(target.box))
                return target.box, "detector"
            # Detector found nothing: let the tracker try

        if self.tracker is not None:
            ok, box = self.tracker.update(frame)
            if ok:
                return _xywh_to_xyxy(box), "tracker"
            self.tracker = None   # lost -> detector will run on next frame

        return None, None


# Quick standalone test: python src/tracker.py [N]
if __name__ == "__main__":
    import sys
    import time

    from detector import Detector
    from source import VideoSource

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    det = Detector("yolo11n.pt", target_classes=["person"])
    tracker_type = sys.argv[2] if len(sys.argv) > 2 else "csrt"
    trk = HybridTracker(det, tracker_type=tracker_type, detect_every_n_frames=n)
    src = VideoSource(0)

    prev = time.time()
    fps = 0.0
    while True:
        frame = src.read()
        if frame is None:
            break

        box, source = trk.update(frame)
        if box:
            # green = YOLO, yellow = tracker
            color = (0, 255, 0) if source == "detector" else (0, 255, 255)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            cv2.putText(frame, source, (box[0], box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  N={n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("tracker test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    cv2.destroyAllWindows()