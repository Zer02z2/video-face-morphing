"""
main_rpi.py

Pi-optimised pipeline. Splits MediaPipe detection and warp/composite into two
threads pinned to separate CPU cores, and adds --skip-warp to reuse the last
warp result for N frames (only re-compositing with the current video frame).

Usage:
    python main_rpi.py <video_path> [options]

Options:
    --warp 0|1          0: full face mapping (default)  1: expression transfer
    --width INT         processing width  (default: video width)
    --height INT        processing height (default: video height)
    --skip INT          run MediaPipe every N frames (default: 1)
    --skip-warp INT     redo triangle warp every N frames, reuse between (default: 1)
    --port INT          TCP port to serve frames on (default: 9002)
    --landmark NORMAL|REDUCED

Display scripts connect to this port:
    python browser_display.py
    python matrix_display.py

The video's .npz file must already exist (run prebaker.py first).
isolcpus=3 is assumed in /boot/firmware/cmdline.txt (core 3 → RGB matrix).
Detection thread → core 1, render thread → core 2, OS/misc → core 0.
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

from core.webcam_detector import WebcamDetector
from core.warp_engine import get_triangle_indices, warp_face
from core.compositor import composite
from core.face_model import get_landmark_indices


# ---------------------------------------------------------------------------
# Shared JPEG (render thread → TCP server)
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
# Shared detection result (detection thread → render thread)
# ---------------------------------------------------------------------------

_latest_detection: tuple | None = None  # (webcam_frame, landmarks, detected)
_detection_lock = threading.Lock()


# ---------------------------------------------------------------------------
# TCP server
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
# Detection thread — MediaPipe on core 1
# ---------------------------------------------------------------------------

def _detection_loop(detector: WebcamDetector, skip: int) -> None:
    global _latest_detection

    try:
        os.sched_setaffinity(0, {1})
        print("Detection thread pinned to core 1")
    except (AttributeError, OSError):
        pass  # macOS or no permission — ignore

    skip_counter = 0
    last_landmarks = None

    for frame, landmarks, detected in detector.frames():
        if skip_counter == 0:
            if detected:
                last_landmarks = landmarks
        else:
            landmarks = last_landmarks
            detected  = last_landmarks is not None
        skip_counter = (skip_counter + 1) % skip

        with _detection_lock:
            _latest_detection = (frame, landmarks, detected)


# ---------------------------------------------------------------------------
# Expression transfer helpers
# ---------------------------------------------------------------------------

def _align_landmarks(webcam_lm: np.ndarray, video_neutral: np.ndarray) -> np.ndarray:
    wc = webcam_lm.mean(axis=0)
    vc = video_neutral.mean(axis=0)
    ws = webcam_lm.max(axis=0) - webcam_lm.min(axis=0)
    vs = video_neutral.max(axis=0) - video_neutral.min(axis=0)
    scale = vs / (ws + 1e-6)
    return (webcam_lm.astype(np.float32) - wc) * scale + vc


def _expression_warp(
    webcam_frame: np.ndarray,
    webcam_lm: np.ndarray,
    video_lm: np.ndarray,
    video_neutral: np.ndarray,
    dst_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    delta        = video_lm.astype(np.float32) - video_neutral
    aligned      = _align_landmarks(webcam_lm, video_neutral)
    animated_dst = np.clip(aligned + delta, 0, [dst_size[1] - 1, dst_size[0] - 1]).astype(np.int16)
    triangles    = get_triangle_indices(animated_dst, dst_size)
    return warp_face(webcam_frame, webcam_lm, animated_dst, dst_size, triangles)


# ---------------------------------------------------------------------------
# Render loop — warp + composite + encode on core 2
# ---------------------------------------------------------------------------

def _pipeline(
    video_path: str,
    npz_path: str,
    warp_mode: int,
    proc_w: int | None,
    proc_h: int | None,
    skip: int,
    skip_warp: int,
    landmark_mode: str,
) -> None:
    try:
        os.sched_setaffinity(0, {2})
        print("Render thread pinned to core 2")
    except (AttributeError, OSError):
        pass

    print("Loading pre-baked landmarks...")
    data = np.load(npz_path, allow_pickle=True)
    landmark_indices = get_landmark_indices(landmark_mode)

    video_landmarks = data["landmarks"].astype(np.float32)
    detected        = data["detected"]
    fade_weights    = data["fade_weights"]
    fps, vid_w, vid_h, total_frames = data["meta"].tolist()
    vid_w, vid_h, total_frames = int(vid_w), int(vid_h), int(total_frames)
    frame_duration = 1.0 / fps

    proc_w = proc_w or vid_w
    proc_h = proc_h or vid_h

    if proc_w != vid_w or proc_h != vid_h:
        video_landmarks[:, :, 0] *= proc_w / vid_w
        video_landmarks[:, :, 1] *= proc_h / vid_h
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
    print(f"Processing at: {proc_w}x{proc_h}  |  warp mode: {warp_mode}  |  "
          f"landmark: {landmark_mode}  |  skip MediaPipe: {skip}  |  skip warp: {skip_warp}")

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

    # Start detection thread
    with WebcamDetector(width=proc_w, height=proc_h, landmark_indices=landmark_indices) as detector:
        threading.Thread(
            target=_detection_loop, args=(detector, skip), daemon=True
        ).start()

        last_warped:  np.ndarray | None = None
        last_mask:    np.ndarray | None = None
        warp_counter: int = 0
        frame_idx:    int = 0

        while True:
            loop_start = time.monotonic()

            # --- Read video frame ---
            ret, video_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            video_frame = cv2.resize(video_frame, (proc_w, proc_h))

            # --- Grab latest detection (non-blocking) ---
            with _detection_lock:
                detection = _latest_detection

            w = float(fade_weights[frame_idx])

            if detection is not None and w > 0:
                webcam_frame, webcam_landmarks, webcam_detected = detection

                if webcam_detected and webcam_landmarks is not None:
                    # Redo warp every skip_warp frames, reuse otherwise
                    if warp_counter == 0 or last_warped is None:
                        if warp_mode == 0:
                            last_warped, last_mask = warp_face(
                                webcam_frame, webcam_landmarks,
                                video_landmarks[frame_idx], (proc_h, proc_w),
                                get_triangles(frame_idx),
                            )
                        else:
                            last_warped, last_mask = _expression_warp(
                                webcam_frame, webcam_landmarks,
                                video_landmarks[frame_idx], video_neutral, (proc_h, proc_w),
                            )
                    warp_counter = (warp_counter + 1) % skip_warp

                    composited = composite(video_frame, last_warped, last_mask)
                    result = cv2.addWeighted(video_frame, 1.0 - w, composited, w, 0) if w < 1.0 else composited
                else:
                    # Face lost — reset warp cache
                    last_warped  = None
                    last_mask    = None
                    warp_counter = 0
                    result = video_frame
            else:
                result = video_frame

            ok, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                _set_jpeg(jpeg.tobytes())

            frame_idx = (frame_idx + 1) % total_frames

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
    parser.add_argument("--warp",      type=int, choices=[0, 1], default=0)
    parser.add_argument("--width",     type=int, default=None)
    parser.add_argument("--height",    type=int, default=None)
    parser.add_argument("--skip",      type=int, default=1,
                        help="Run MediaPipe every N frames (default: 1)")
    parser.add_argument("--skip-warp", type=int, default=1, dest="skip_warp",
                        help="Redo triangle warp every N frames (default: 1)")
    parser.add_argument("--port",      type=int, default=9002)
    parser.add_argument("--landmark",  choices=["NORMAL", "REDUCED", "COARSE"], default="NORMAL")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"ERROR: video not found: {args.video_path}")
        sys.exit(1)

    npz_path = os.path.splitext(args.video_path)[0] + f"_{args.landmark.lower()}.npz"
    if not os.path.exists(npz_path):
        print(f"ERROR: pre-baked data not found: {npz_path}")
        print(f"       Run: python prebaker.py {args.video_path} --landmark {args.landmark}")
        sys.exit(1)

    threading.Thread(target=_tcp_server, args=(args.port,), daemon=True).start()

    _pipeline(
        args.video_path, npz_path,
        args.warp, args.width, args.height,
        args.skip, args.skip_warp, args.landmark,
    )
