"""
main.py

Core pipeline. Reads video + webcam, computes face warp, streams composed
frames as JPEG over a TCP socket for display scripts to consume.

Usage:
    python main.py <video_path> [options]

Options:
    --warp 0|1      0: full face mapping (default)  1: expression transfer
    --width INT     webcam/processing width  (default: video width)
    --height INT    webcam/processing height (default: video height)
    --skip INT      run MediaPipe every N frames, reuse landmarks between (default: 1)
    --port INT      TCP port to serve frames on (default: 9002)

Display scripts connect to this port to receive frames:
    python browser_display.py   (Mac/browser)
    python matrix_display.py    (Pi/RGB matrix)

The video's .npz file must already exist (run prebaker.py first).
"""

import sys
import os
import time
import socket
import struct
import argparse
import threading
import cv2
import numpy as np

from webcam_detector import WebcamDetector
from warp_engine import get_triangle_indices, warp_face
from compositor import composite


# ---------------------------------------------------------------------------
# Shared latest JPEG between pipeline thread and TCP server thread
# ---------------------------------------------------------------------------

_latest_jpeg: bytes | None = None
_jpeg_lock = threading.Lock()


def _set_jpeg(jpeg: bytes) -> None:
    global _latest_jpeg
    with _jpeg_lock:
        _latest_jpeg = jpeg


def _get_jpeg() -> bytes | None:
    with _jpeg_lock:
        return _latest_jpeg


# ---------------------------------------------------------------------------
# TCP server — streams frames to one connected display script at a time
# ---------------------------------------------------------------------------

def _tcp_server(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        print(f"Waiting for display client on port {port}...")
        while True:
            try:
                conn, addr = srv.accept()
                print(f"Display client connected: {addr}")
                last_sent = None
                with conn:
                    while True:
                        jpeg = _get_jpeg()
                        if jpeg is None or jpeg is last_sent:
                            time.sleep(0.005)
                            continue
                        last_sent = jpeg
                        try:
                            header = struct.pack(">I", len(jpeg))
                            conn.sendall(header + jpeg)
                        except (BrokenPipeError, ConnectionResetError):
                            print(f"Display client disconnected: {addr}")
                            break
            except Exception as e:
                print(f"TCP server error: {e}")


# ---------------------------------------------------------------------------
# Expression transfer helpers (warp mode 1)
# ---------------------------------------------------------------------------

def _align_landmarks(
    webcam_landmarks: np.ndarray,
    video_neutral: np.ndarray,
) -> np.ndarray:
    webcam_center = webcam_landmarks.mean(axis=0)
    video_center  = video_neutral.mean(axis=0)
    webcam_span   = webcam_landmarks.max(axis=0) - webcam_landmarks.min(axis=0)
    video_span    = video_neutral.max(axis=0)    - video_neutral.min(axis=0)
    scale = video_span / (webcam_span + 1e-6)
    return (webcam_landmarks.astype(np.float32) - webcam_center) * scale + video_center


def _expression_warp(
    webcam_frame: np.ndarray,
    webcam_landmarks: np.ndarray,
    video_landmarks: np.ndarray,
    video_neutral: np.ndarray,
    dst_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    delta        = video_landmarks.astype(np.float32) - video_neutral
    aligned      = _align_landmarks(webcam_landmarks, video_neutral)
    animated_dst = np.clip(aligned + delta, 0, [dst_size[1] - 1, dst_size[0] - 1]).astype(np.int16)
    triangles    = get_triangle_indices(animated_dst, dst_size)
    return warp_face(webcam_frame, webcam_landmarks, animated_dst, dst_size, triangles)


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _pipeline(
    video_path: str,
    npz_path: str,
    warp_mode: int,
    proc_w: int | None,
    proc_h: int | None,
    skip: int,
) -> None:
    print("Loading pre-baked landmarks...")
    data = np.load(npz_path, allow_pickle=True)
    video_landmarks = data["landmarks"].astype(np.float32)
    detected        = data["detected"]
    fade_weights    = data["fade_weights"]
    fps, vid_w, vid_h, total_frames = data["meta"].tolist()
    vid_w, vid_h, total_frames = int(vid_w), int(vid_h), int(total_frames)
    frame_duration = 1.0 / fps

    # Use video resolution if not overridden
    proc_w = proc_w or vid_w
    proc_h = proc_h or vid_h

    # Scale pre-baked landmarks to processing resolution
    if proc_w != vid_w or proc_h != vid_h:
        sx = proc_w / vid_w
        sy = proc_h / vid_h
        video_landmarks[:, :, 0] *= sx
        video_landmarks[:, :, 1] *= sy
    video_landmarks = video_landmarks.astype(np.int16)

    video_neutral = None
    if "neutral" in data:
        video_neutral = data["neutral"].astype(np.float32)
        if proc_w != vid_w or proc_h != vid_h:
            video_neutral[:, 0] *= proc_w / vid_w
            video_neutral[:, 1] *= proc_h / vid_h

    if warp_mode == 1 and video_neutral is None:
        print("ERROR: .npz missing neutral pose. Re-run prebaker.py then try again.")
        return

    print(f"Loaded: {total_frames} frames, {vid_w}x{vid_h} @ {fps:.2f}fps")
    print(f"Processing at: {proc_w}x{proc_h}  |  warp mode: {warp_mode}  |  MediaPipe skip: {skip}")

    # Lazy triangle cache (mode 0)
    triangle_cache: dict[int, list | None] = {}

    def get_triangles(idx: int) -> list | None:
        if idx not in triangle_cache:
            triangle_cache[idx] = (
                get_triangle_indices(video_landmarks[idx], (proc_h, proc_w))
                if detected[idx] else None
            )
        return triangle_cache[idx]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}")
        return

    # Webcam landmark state for frame skipping
    last_webcam_landmarks: np.ndarray | None = None
    last_webcam_frame:     np.ndarray | None = None
    skip_counter = 0

    with WebcamDetector(width=proc_w, height=proc_h) as detector:
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
            video_frame = cv2.resize(video_frame, (proc_w, proc_h))

            # --- Read webcam frame ---
            try:
                webcam_frame, webcam_landmarks, webcam_detected = next(webcam_gen)
            except StopIteration:
                print("Webcam closed.")
                break

            # --- Frame skipping: only run MediaPipe every `skip` frames ---
            if skip_counter == 0:
                if webcam_detected:
                    last_webcam_landmarks = webcam_landmarks
                    last_webcam_frame     = webcam_frame
            else:
                # reuse last known landmarks, but use current frame pixels
                webcam_landmarks = last_webcam_landmarks
                webcam_detected  = last_webcam_landmarks is not None
                if webcam_detected:
                    webcam_frame = webcam_frame  # use current pixels with old landmarks
            skip_counter = (skip_counter + 1) % skip

            # --- Warp + composite ---
            w = float(fade_weights[frame_idx])

            if w > 0 and webcam_detected and webcam_landmarks is not None:
                if warp_mode == 0:
                    warped, mask = warp_face(
                        webcam_frame, webcam_landmarks,
                        video_landmarks[frame_idx], (proc_h, proc_w),
                        get_triangles(frame_idx),
                    )
                else:
                    warped, mask = _expression_warp(
                        webcam_frame, webcam_landmarks,
                        video_landmarks[frame_idx], video_neutral, (proc_h, proc_w),
                    )

                composited = composite(video_frame, warped, mask)
                result = cv2.addWeighted(video_frame, 1.0 - w, composited, w, 0) if w < 1.0 else composited
            else:
                result = video_frame

            # Encode and publish
            ok, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                _set_jpeg(jpeg.tobytes())

            frame_idx = (frame_idx + 1) % total_frames

            # Pace to video fps
            elapsed    = time.monotonic() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--warp",   type=int, choices=[0, 1], default=0)
    parser.add_argument("--width",  type=int, default=None,
                        help="Processing width (default: video width)")
    parser.add_argument("--height", type=int, default=None,
                        help="Processing height (default: video height)")
    parser.add_argument("--skip",   type=int, default=1,
                        help="Run MediaPipe every N frames (default: 1 = every frame)")
    parser.add_argument("--port",   type=int, default=9002)
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"ERROR: video not found: {args.video_path}")
        sys.exit(1)

    npz_path = os.path.splitext(args.video_path)[0] + ".npz"
    if not os.path.exists(npz_path):
        print(f"ERROR: pre-baked data not found: {npz_path}")
        print(f"       Run: python prebaker.py {args.video_path}")
        sys.exit(1)

    threading.Thread(
        target=_tcp_server, args=(args.port,), daemon=True
    ).start()

    _pipeline(args.video_path, npz_path, args.warp, args.width, args.height, args.skip)
