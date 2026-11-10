"""Detector module: runs YOLO and returns simple detection objects."""

from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Detection:
    box: tuple        # (x1, y1, x2, y2) in pixels
    conf: float       # confidence, 0.0 - 1.0
    cls_name: str     # e.g. "drone"


class Detector:
    def __init__(self, weights, img_size=640, conf_threshold=0.4, target_classes=None):
        self.model = YOLO(weights)          # loaded once, at startup
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        # Model knows its own classes, e.g. {0: 'aircraft', 1: 'bird', 2: 'drone', ...}
        names = self.model.names
        if target_classes:
            name_to_id = {name: idx for idx, name in names.items()}
            missing = [c for c in target_classes if c not in name_to_id]
            if missing:
                raise ValueError(
                    f"Unknown classes in config: {missing}. "
                    f"Model classes: {list(names.values())}"
                )
            self.target_ids = [name_to_id[c] for c in target_classes]
        else:
            self.target_ids = None          # None -> keep all classes

    def detect(self, frame):
        """Return a list of Detection objects for one frame."""
        result = self.model.predict(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,       # filter 1: confidence
            classes=self.target_ids,        # filter 2: target classes only
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                box=(int(x1), int(y1), int(x2), int(y2)),
                conf=float(box.conf[0]),
                cls_name=self.model.names[int(box.cls[0])],
            ))
        return detections


# Quick standalone test: python src/detector.py [weights] [class_name]
if __name__ == "__main__":
    import sys

    import cv2
    from source import VideoSource

    weights = sys.argv[1] if len(sys.argv) > 1 else "yolo11n.pt"
    target = sys.argv[2] if len(sys.argv) > 2 else "person"

    det = Detector(weights, target_classes=[target])
    src = VideoSource(0)

    while True:
        frame = src.read()
        if frame is None:
            break
        for d in det.detect(frame):
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.cls_name} {d.conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("detector test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    cv2.destroyAllWindows()