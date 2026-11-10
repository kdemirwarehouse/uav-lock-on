"""Video source module: provides frames from a video file or a webcam."""

import cv2


class VideoSource:
    def __init__(self, path):
        # "0" (from config) or 0 -> webcam, anything else -> file path
        if isinstance(path, str) and path.isdigit():
            path = int(path)

        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {path}")

        # Some webcams report FPS as 0, fall back to 30
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return the next frame, or None when the video ends."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()


# Quick standalone test: python src/source.py [video_path]
if __name__ == "__main__":
    import sys

    src = VideoSource(sys.argv[1] if len(sys.argv) > 1 else 0)
    print(f"FPS: {src.fps:.1f}  Size: {src.width}x{src.height}")

    while True:
        frame = src.read()
        if frame is None:
            break
        cv2.imshow("source test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    cv2.destroyAllWindows()