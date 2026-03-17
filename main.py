"""
main.py

Orchestrator. Wires together webcam_detector, warp_engine, and compositor,
then serves the result as an MJPEG stream via Flask.

Usage:
    python main.py <video_path> [--warp 0|1]

    --warp 0  (default) Full face mapping: webcam face warped into video face shape
    --warp 1  Expression transfer: video face expressions applied to webcam face,
              preserving the human face's own proportions

The video's .npz file must already exist (run prebaker.py first).
Open http://localhost:9002 in a browser to view the stream.
"""

import sys
import os
import time
import argparse
import threading
import cv2
import numpy as np
from flask import Flask, Response

from webcam_detector import WebcamDetector
from warp_engine import get_triangle_indices, warp_face
from compositor import composite


# ---------------------------------------------------------------------------
# Shared state between pipeline thread and Flask thread
# ---------------------------------------------------------------------------

_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()


def _set_frame(frame: np.ndarray) -> None:
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame


def _get_frame() -> np.ndarray | None:
    with _frame_lock:
        return _latest_frame


# ---------------------------------------------------------------------------
# Expression transfer helpers (warp mode 1)
# ---------------------------------------------------------------------------

def _align_landmarks(
    webcam_landmarks: np.ndarray,
    shrek_neutral: np.ndarray,
) -> np.ndarray:
    """
    Translate and scale webcam landmarks to match the center and scale
    of the video's neutral face position. Returns float32.
    """
    webcam_center = webcam_landmarks.mean(axis=0)
    shrek_center  = shrek_neutral.mean(axis=0)
    webcam_span   = webcam_landmarks.max(axis=0) - webcam_landmarks.min(axis=0)
    shrek_span    = shrek_neutral.max(axis=0)    - shrek_neutral.min(axis=0)
    scale = shrek_span / (webcam_span + 1e-6)
    return (webcam_landmarks.astype(np.float32) - webcam_center) * scale + shrek_center


def _expression_warp(
    webcam_frame: np.ndarray,
    webcam_landmarks: np.ndarray,
    shrek_landmarks: np.ndarray,
    shrek_neutral: np.ndarray,
    dst_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    shrek_delta  = shrek_landmarks.astype(np.float32) - shrek_neutral
    aligned      = _align_landmarks(webcam_landmarks, shrek_neutral)
    animated_dst = np.clip(aligned + shrek_delta, 0, [dst_size[1] - 1, dst_size[0] - 1]).astype(np.int16)
    triangles    = get_triangle_indices(animated_dst, dst_size)
    return warp_face(webcam_frame, webcam_landmarks, animated_dst, dst_size, triangles)


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _pipeline(video_path: str, npz_path: str, warp_mode: int) -> None:
    print("Loading pre-baked landmarks...")
    data = np.load(npz_path, allow_pickle=True)
    video_landmarks = data["landmarks"]
    detected        = data["detected"]
    fade_weights    = data["fade_weights"]
    fps, vid_w, vid_h, total_frames = data["meta"].tolist()
    vid_w, vid_h, total_frames = int(vid_w), int(vid_h), int(total_frames)
    frame_duration = 1.0 / fps
    print(f"Loaded: {total_frames} frames, {vid_w}x{vid_h} @ {fps:.2f}fps  |  warp mode: {warp_mode}")

    shrek_neutral = data["neutral"].astype(np.float32) if "neutral" in data else None
    if warp_mode == 1 and shrek_neutral is None:
        print("ERROR: .npz missing neutral pose. Re-run prebaker.py then try again.")
        return

    # Lazy triangle cache (mode 0 only)
    triangle_cache: dict[int, list | None] = {}

    def get_triangles(idx: int) -> list | None:
        if idx not in triangle_cache:
            triangle_cache[idx] = get_triangle_indices(video_landmarks[idx], (vid_h, vid_w)) if detected[idx] else None
        return triangle_cache[idx]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}")
        return

    print("Pipeline running. Open http://localhost:9002 in your browser.")

    with WebcamDetector(width=vid_w, height=vid_h) as detector:
        webcam_gen = detector.frames()
        frame_idx  = 0

        while True:
            loop_start = time.monotonic()

            # --- Read video frame ---
            ret, video_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            video_frame = cv2.resize(video_frame, (vid_w, vid_h))

            # --- Read webcam frame ---
            try:
                webcam_frame, webcam_landmarks, webcam_detected = next(webcam_gen)
            except StopIteration:
                print("Webcam closed.")
                break

            # --- Warp + composite ---
            w = float(fade_weights[frame_idx])

            if w > 0 and webcam_detected:
                if warp_mode == 0:
                    warped, mask = warp_face(
                        webcam_frame, webcam_landmarks,
                        video_landmarks[frame_idx], (vid_h, vid_w),
                        get_triangles(frame_idx),
                    )
                else:
                    warped, mask = _expression_warp(
                        webcam_frame, webcam_landmarks,
                        video_landmarks[frame_idx], shrek_neutral, (vid_h, vid_w),
                    )

                composited = composite(video_frame, warped, mask)
                result = cv2.addWeighted(video_frame, 1.0 - w, composited, w, 0) if w < 1.0 else composited
            else:
                result = video_frame

            _set_frame(result)
            frame_idx = (frame_idx + 1) % total_frames

            # --- Pace to video fps ---
            elapsed    = time.monotonic() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()


# ---------------------------------------------------------------------------
# Flask MJPEG server
# ---------------------------------------------------------------------------

app = Flask(__name__)


def _mjpeg_generator():
    while True:
        frame = _get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>Face Morph</title>
  <style>
    body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }
    img  { max-width: 100%; max-height: 100vh; display: block; }
  </style>
</head>
<body>
  <img src="/stream">
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--warp", type=int, choices=[0, 1], default=0,
                        help="0: full face mapping (default)  1: expression transfer")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"ERROR: video not found: {args.video_path}")
        sys.exit(1)

    npz_path = os.path.splitext(args.video_path)[0] + ".npz"
    if not os.path.exists(npz_path):
        print(f"ERROR: pre-baked data not found: {npz_path}")
        print(f"       Run: python prebaker.py {args.video_path}")
        sys.exit(1)

    t = threading.Thread(target=_pipeline, args=(args.video_path, npz_path, args.warp), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=9002, debug=False, threaded=True)
