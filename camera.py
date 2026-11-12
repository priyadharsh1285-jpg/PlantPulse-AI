"""Laptop camera capture via OpenCV."""
import os
import time

import cv2

from .vision import analyze_frame, motion_between


class CameraError(RuntimeError):
    pass


def _open(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open camera {index}. Close other apps using it, or change the camera index in Settings."
        )
    return cap


def grab(index=0, warmup=5):
    """Grab one frame (BGR). Discards a few frames so auto-exposure can settle."""
    cap = _open(index)
    try:
        frame = None
        for _ in range(max(warmup, 1)):
            ok, f = cap.read()
            if ok:
                frame = f
            time.sleep(0.05)
        if frame is None:
            raise CameraError("Camera opened but returned no frames.")
        return frame
    finally:
        cap.release()


def capture_pair(index=0, gap=2.0, roi=(0, 0, 1, 1)):
    """Capture two frames `gap` seconds apart. Returns (second_frame, motion_score)."""
    cap = _open(index)
    try:
        frames = []
        for i in range(2):
            frame = None
            for _ in range(5):
                ok, f = cap.read()
                if ok:
                    frame = f
                time.sleep(0.05)
            if frame is None:
                raise CameraError("Camera opened but returned no frames.")
            frames.append(frame)
            if i == 0:
                time.sleep(gap)
    finally:
        cap.release()
    _, da = analyze_frame(frames[0], roi)
    _, db = analyze_frame(frames[1], roi)
    return frames[1], motion_between(da, db)
