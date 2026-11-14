"""Output module: draws the HUD, shows/saves the video and logs lock events."""

import json
import os

import cv2

# Works both as a package import (python main.py -> src.output) and as a
# script (python src/output.py), where src/ itself is on sys.path.
try:
    from .lock_manager import LockState
except ImportError:
    from lock_manager import LockState

# BGR colours. Box colour tells the lock state at a glance.
STATE_COLORS = {
    LockState.SEARCHING: (200, 200, 200),
    LockState.TARGET: (0, 255, 255),      # yellow
    LockState.LOCKING: (0, 165, 255),     # orange
    LockState.LOCKED: (0, 0, 255),        # red
}
REGION_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


class Output:
    def __init__(self, cfg, frame_w, frame_h, fps):
        """cfg: the `output` section of config.yaml."""
        self.show_window = cfg["show_window"]
        self.log_path = cfg["log_path"]
        self.events = []
        self.window_name = "uav-lock-on"
        self.paused = False

        # Text and line sizes follow the frame height, so the HUD looks the
        # same on 480p and 4K videos.
        self.scale = frame_h / 720
        self.thick = max(1, round(2 * self.scale))

        self.writer = None
        if cfg["save_video"]:
            self.video_path = cfg["video_path"]
            _ensure_parent_dir(self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.video_path, fourcc, fps,
                                          (frame_w, frame_h))
            # VideoWriter does not raise on a bad codec/path, it just
            # silently writes nothing -> check explicitly.
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self.video_path}")

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Write an empty log now, so a run with zero locks still leaves a
        # valid JSON file behind.
        self._write_log()

    # ---------------- drawing ----------------

    def _text(self, img, text, org, color, size=0.8):
        # Dark outline first, so text stays readable on bright sky.
        s = size * self.scale
        cv2.putText(img, text, org, FONT, s, (0, 0, 0), self.thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, FONT, s, color, self.thick, cv2.LINE_AA)

    def draw(self, frame, result, state, progress, region, fps):
        """Draw the HUD onto `frame` in place and return it.
        All text is ASCII: cv2.putText cannot render Turkish characters."""
        color = STATE_COLORS[state]
        h, w = frame.shape[:2]

        # Lock region: thin line, it is a reference, not the focus
        cv2.rectangle(frame, region[:2], region[2:], REGION_COLOR, 1)

        # Target box + class labels
        if result is not None:
            x1, y1, x2, y2 = result.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thick + 1)
            label = result.voted_class
            if result.raw_class:   # only on frames where YOLO ran
                label += f" (raw: {result.raw_class})"
            y_text = max(y1 - int(10 * self.scale), int(20 * self.scale))
            self._text(frame, label, (x1, y_text), color, 0.7)

        # Top bar: state name, progress bar, average FPS
        pad = int(15 * self.scale)
        y_line = int(40 * self.scale)
        self._text(frame, state.value, (pad, y_line), color, 1.0)

        bar_w, bar_h = int(w * 0.3), int(18 * self.scale)
        bx1 = (w - bar_w) // 2
        by1 = y_line - bar_h
        cv2.rectangle(frame, (bx1, by1), (bx1 + bar_w, by1 + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + int(bar_w * progress), by1 + bar_h), color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + bar_w, by1 + bar_h), REGION_COLOR, 1)
        self._text(frame, f"{int(progress * 100)}%", (bx1 + bar_w + pad, y_line),
                   REGION_COLOR, 0.7)

        fps_text = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.8 * self.scale, self.thick)
        self._text(frame, fps_text, (w - tw - pad, y_line), REGION_COLOR, 0.8)
        return frame

    # ---------------- sinks ----------------

    def write(self, frame):
        """Save and/or show an already drawn frame.
        Returns False when the user asks to quit."""
        if self.writer is not None:
            self.writer.write(frame)
        if not self.show_window:
            return True

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            self.paused = True
        # While paused, block here but keep listening for p / q
        while self.paused:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("p"):
                self.paused = False
            elif key == ord("q"):
                break
            elif self._window_closed():
                return False
        return key != ord("q") and not self._window_closed()

    def _window_closed(self):
        # Clicking the window's X should stop the program like 'q'
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1

    def log_event(self, event):
        self.events.append({
            "start_s": round(event.start_s, 3),
            "end_s": round(event.end_s, 3),
            "class": event.cls_name,
            "box": [int(v) for v in event.box],
        })
        # Rewrite the file on every event: a crash later in the video
        # does not lose locks that already happened.
        self._write_log()

    def _write_log(self):
        _ensure_parent_dir(self.log_path)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show_window:
            cv2.destroyAllWindows()


# Standalone test with a synthetic scene (no model, no video file):
#   python src/output.py            -> shows a window
#   python src/output.py --headless -> only writes files
if __name__ == "__main__":
    import sys

    import numpy as np
    from lock_manager import LockManager
    from tracker import TrackResult

    W, H, FPS = 1280, 720, 30
    cfg = {
        "show_window": "--headless" not in sys.argv,
        "save_video": True,
        "video_path": "outputs/output_test.mp4",
        "log_path": "logs/output_test_events.json",
    }
    region = {"x_min": 0.25, "y_min": 0.10, "x_max": 0.75, "y_max": 0.90}
    lm = LockManager(W, H, FPS, ["drone"], region, lock_duration_s=2.0)
    out = Output(cfg, W, H, FPS)

    # A box that flies from the left edge into the centre and stays there
    snapshot_saved = False
    for f in range(150):
        frame = np.full((H, W, 3), (235, 206, 135), np.uint8)   # sky blue
        x = min(50 + f * 15, 600)
        box = (x, 330, x + 80, 390)
        res = TrackResult(box, "detector" if f % 5 == 0 else "tracker",
                          "helicopter" if f % 10 == 0 else None, "drone")
        event = lm.update(res)
        if event:
            out.log_event(event)
            print(f"LOCK EVENT: {event}")
        out.draw(frame, res, lm.state, lm.progress, lm.region, fps=FPS)
        if lm.state == LockState.LOCKED and not snapshot_saved:
            # imwrite returns False instead of raising -> check it
            if not cv2.imwrite("outputs/output_test.jpg", frame):
                raise RuntimeError("Could not write outputs/output_test.jpg")
            snapshot_saved = True
        if not out.write(frame):
            break
    out.close()
    print(f"video: {cfg['video_path']}  log: {cfg['log_path']}  snapshot: outputs/output_test.jpg")
