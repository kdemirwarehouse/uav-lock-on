"""uav-lock-on: detect, track and lock on to a UAV in a video.

Pipeline: VideoSource -> Detector -> HybridTracker -> LockManager -> Output

Usage:
    python main.py
    python main.py --source 0               # webcam
    python main.py --config configs/other.yaml --source videos/x.mp4
"""

import argparse
import time
from collections import defaultdict

import yaml

from src.detector import Detector
from src.lock_manager import LockManager
from src.output import Output
from src.source import VideoSource
from src.tracker import HybridTracker


class Timer:
    """Accumulates wall-clock time per named stage."""

    def __init__(self):
        self.total = defaultdict(float)   # seconds
        self.calls = defaultdict(int)

    def add(self, name, seconds):
        self.total[name] += seconds
        self.calls[name] += 1


class TimedDetector:
    """Wraps a Detector to time each YOLO call.
    The detector runs *inside* HybridTracker.update(), so this proxy is the
    only way to measure it separately without changing the tracker's API."""

    def __init__(self, detector, timer):
        self._detector = detector
        self._timer = timer

    def detect(self, frame):
        t0 = time.perf_counter()
        detections = self._detector.detect(frame)
        self._timer.add("detector", time.perf_counter() - t0)
        return detections


def parse_args():
    p = argparse.ArgumentParser(description="UAV detection, tracking and lock-on")
    p.add_argument("--config", default="configs/config.yaml", help="path to YAML config")
    p.add_argument("--source", default=None,
                   help="overrides source.path in the config; '0' = webcam")
    return p.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_report(timer, n_frames, wall_s):
    """Per-module timing table (ms), ready to paste into the README."""
    def per_frame(name):
        return 1000 * timer.total[name] / n_frames

    def per_call(name):
        calls = timer.calls[name]
        return 1000 * timer.total[name] / calls if calls else 0.0

    # The tracker's time includes the YOLO calls it makes; subtract them to
    # show what the KCF tracker + voting logic alone costs.
    tracker_only = timer.total["tracker"] - timer.total["detector"]

    rows = [
        ("source.read", timer.calls["source"], per_call("source"), per_frame("source")),
        ("detector (YOLO)", timer.calls["detector"], per_call("detector"), per_frame("detector")),
        ("tracker total (YOLO + KCF)", timer.calls["tracker"], per_call("tracker"), per_frame("tracker")),
        ("  of which KCF + voting", timer.calls["tracker"],
         1000 * tracker_only / n_frames, 1000 * tracker_only / n_frames),
        ("lock_manager", timer.calls["lock"], per_call("lock"), per_frame("lock")),
        ("output (draw + save/show)", timer.calls["output"], per_call("output"), per_frame("output")),
    ]

    print("\nPerformance")
    print(f"{'Module':<28}{'Calls':>8}{'ms/call':>10}{'ms/frame':>10}")
    print("-" * 56)
    for name, calls, call_ms, frame_ms in rows:
        print(f"{name:<28}{calls:>8}{call_ms:>10.2f}{frame_ms:>10.2f}")
    print("-" * 56)
    print(f"{'Frames processed':<28}{n_frames:>8}")
    print(f"{'Total time (s)':<28}{wall_s:>8.1f}")
    print(f"{'Average FPS':<28}{n_frames / wall_s:>8.1f}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    source_path = args.source if args.source is not None else cfg["source"]["path"]

    # ---- build the pipeline from config ----
    timer = Timer()
    source = VideoSource(source_path)

    det_cfg = cfg["detector"]
    detector = Detector(
        det_cfg["weights"],
        img_size=det_cfg["img_size"],
        conf_threshold=det_cfg["conf_threshold"],
        target_classes=det_cfg["track_classes"],   # what we TRACK
    )

    trk_cfg = cfg["tracker"]
    tracker = HybridTracker(
        TimedDetector(detector, timer),
        tracker_type=trk_cfg["type"],
        detect_every_n_frames=trk_cfg["detect_every_n_frames"],
        vote_window=trk_cfg["vote_window"],
        reset_after_missed_frames=trk_cfg["reset_after_missed_frames"],
    )

    lock_cfg = cfg["lock"]
    lock = LockManager(
        source.width, source.height, source.fps,
        target_classes=lock_cfg["target_classes"],   # what we LOCK ON
        region=lock_cfg["region"],
        min_box_ratio=lock_cfg["min_box_ratio"],
        lock_duration_s=lock_cfg["lock_duration_s"],
        max_lost_frames=lock_cfg["max_lost_frames"],
    )

    output = Output(cfg["output"], source.width, source.height, source.fps)

    print(f"Source: {source_path}  ({source.width}x{source.height} @ {source.fps:.1f} FPS)")

    # ---- frame loop ----
    n_frames = 0
    t_start = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            frame = source.read()
            t1 = time.perf_counter()
            if frame is None:
                break
            timer.add("source", t1 - t0)

            result = tracker.update(frame)
            t2 = time.perf_counter()
            timer.add("tracker", t2 - t1)

            event = lock.update(result)
            t3 = time.perf_counter()
            timer.add("lock", t3 - t2)

            n_frames += 1
            if event:
                output.log_event(event)
                print(f"LOCK #{len(output.events)}: {event.cls_name} "
                      f"{event.start_s:.2f}s -> {event.end_s:.2f}s")

            # Running average, shown on screen
            avg_fps = n_frames / (t3 - t_start)
            output.draw(frame, result, lock.state, lock.progress, lock.region, avg_fps)
            keep_going = output.write(frame)
            timer.add("output", time.perf_counter() - t3)
            if not keep_going:
                print("Stopped by user.")
                break
    finally:
        # Always release the video writer, otherwise the mp4 is unreadable
        source.release()
        output.close()

    wall_s = time.perf_counter() - t_start
    if n_frames:
        print_report(timer, n_frames, wall_s)
    print(f"\nTotal locks: {len(output.events)}")
    print(f"Lock log:    {output.log_path}")
    if output.writer is not None:
        print(f"Video:       {output.video_path}")


if __name__ == "__main__":
    main()
