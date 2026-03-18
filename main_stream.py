"""
main_stream.py

Runs on Mac. Connects to Pi, receives webcam frames, does face morphing,
pushes processed frames back to Pi for display on the RGB matrix.

Start pi_stream.py on the Pi first, then run:
    python main_stream.py <video_path> --pi-host <pi-ip> [options]

Options:
    --pi-host STR       IP of the Raspberry Pi running pi_stream.py (required)
    --warp 0|1          0: full face mapping (default)  1: expression transfer
    --width INT         processing width  (default: video width)
    --height INT        processing height (default: video height)
    --skip INT          run MediaPipe every N incoming frames (default: 1)
    --skip-warp INT     redo triangle warp every N frames (default: 1)
    --webcam-port INT   port on Pi serving webcam frames (default: 9001)
    --output-port INT   port on Pi receiving processed frames (default: 9002)
    --landmark NORMAL|REDUCED

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
import mediapipe as mp

from core.warp_engine import get_triangle_indices, warp_face
from core.compositor import _feather_mask, _match_color
from core.face_model import create_image_landmarker, landmarks_to_numpy, get_landmark_indices


# ---------------------------------------------------------------------------
# Shared JPEG (pipeline → output server)
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
# Shared detection result (webcam server → pipeline)
# ---------------------------------------------------------------------------

_latest_detection: tuple | None = None  # (frame_bgr, landmarks, detected)
_detection_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recvall(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# Webcam client — connects to Pi, reads frames, runs MediaPipe
# ---------------------------------------------------------------------------

def _webcam_client(pi_host: str, webcam_port: int, landmark_indices: list | None, skip: int) -> None:
    global _latest_detection

    while True:
        try:
            print(f"[webcam] Connecting to Pi at {pi_host}:{webcam_port}...")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((pi_host, webcam_port))
                print("[webcam] Connected.")
                skip_counter   = 0
                last_landmarks = None

                with create_image_landmarker() as landmarker:
                    while True:
                        raw = _recvall(s, 4)
                        if not raw:
                            break
                        length = struct.unpack(">I", raw)[0]
                        jpeg   = _recvall(s, length)
                        if not jpeg:
                            break

                        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        if skip_counter == 0:
                            h, w     = frame.shape[:2]
                            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            result   = landmarker.detect(mp_image)

                            if result.face_landmarks:
                                landmarks      = landmarks_to_numpy(result.face_landmarks[0], w, h, landmark_indices)
                                detected       = True
                                last_landmarks = landmarks
                            else:
                                landmarks = last_landmarks
                                detected  = last_landmarks is not None
                        else:
                            landmarks = last_landmarks
                            detected  = last_landmarks is not None

                        skip_counter = (skip_counter + 1) % skip

                        with _detection_lock:
                            _latest_detection = (frame, landmarks, detected)

        except Exception as e:
            print(f"[webcam] Connection lost: {e} — retrying in 2s...")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Output client — connects to Pi, pushes processed frames
# ---------------------------------------------------------------------------

def _output_client(pi_host: str, output_port: int) -> None:
    while True:
        try:
            print(f"[output] Connecting to Pi at {pi_host}:{output_port}...")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((pi_host, output_port))
                print("[output] Connected.")
                last_sent = None
                while True:
                    jpeg = _get_jpeg()
                    if jpeg is None or jpeg is last_sent:
                        time.sleep(0.005)
                        continue
                    last_sent = jpeg
                    try:
                        header = struct.pack(">I", len(jpeg))
                        s.sendall(header + jpeg)
                    except (BrokenPipeError, ConnectionResetError):
                        break
        except Exception as e:
            print(f"[output] Connection lost: {e} — retrying in 2s...")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Expression transfer helpers
# ---------------------------------------------------------------------------

def _align_landmarks(webcam_lm: np.ndarray, video_neutral: np.ndarray) -> np.ndarray:
    wc    = webcam_lm.mean(axis=0)
    vc    = video_neutral.mean(axis=0)
    ws    = webcam_lm.max(axis=0) - webcam_lm.min(axis=0)
    vs    = video_neutral.max(axis=0) - video_neutral.min(axis=0)
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
# Pipeline — render loop on Mac
# ---------------------------------------------------------------------------

def _pipeline(
    video_path: str,
    npz_path: str,
    warp_mode: int,
    proc_w: int | None,
    proc_h: int | None,
    skip_warp: int,
    feather_radius: int,
    landmark_mode: str,
) -> None:
    print("Loading pre-baked landmarks...")
    data            = np.load(npz_path, allow_pickle=True)
    video_landmarks = data["landmarks"].astype(np.float32)
    detected        = data["detected"]
    fade_weights    = data["fade_weights"]
    fps, vid_w, vid_h, total_frames = data["meta"].tolist()
    vid_w, vid_h, total_frames = int(vid_w), int(vid_h), int(total_frames)
    frame_duration  = 1.0 / fps

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
          f"landmark: {landmark_mode}  |  skip warp: {skip_warp}")

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

    last_warped:    np.ndarray | None = None
    last_mask:      np.ndarray | None = None
    last_corrected: np.ndarray | None = None
    last_alpha:     np.ndarray | None = None
    warp_counter: int = 0
    frame_idx:    int = 0

    print("Pipeline running. Waiting for Pi webcam frames...")

    while True:
        loop_start = time.monotonic()

        ret, video_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue
        video_frame = cv2.resize(video_frame, (proc_w, proc_h))

        with _detection_lock:
            detection = _latest_detection

        w = float(fade_weights[frame_idx])

        if detection is not None and w > 0:
            webcam_frame, webcam_landmarks, webcam_detected = detection

            if webcam_detected and webcam_landmarks is not None:
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
                    last_alpha     = _feather_mask(last_mask, feather_radius)[:, :, np.newaxis]
                    last_corrected = _match_color(last_warped, video_frame, last_mask).astype(np.float32)
                warp_counter = (warp_counter + 1) % skip_warp

                blended    = video_frame.astype(np.float32) * (1.0 - last_alpha) + last_corrected * last_alpha
                composited = np.clip(blended, 0, 255).astype(np.uint8)
                result = cv2.addWeighted(video_frame, 1.0 - w, composited, w, 0) if w < 1.0 else composited
            else:
                last_warped    = None
                last_mask      = None
                last_corrected = None
                last_alpha     = None
                warp_counter   = 0
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
    parser.add_argument("--warp",         type=int, choices=[0, 1], default=0)
    parser.add_argument("--width",        type=int, default=None)
    parser.add_argument("--height",       type=int, default=None)
    parser.add_argument("--pi-host",      required=True, dest="pi_host",
                        help="IP of the Raspberry Pi running pi_stream.py")
    parser.add_argument("--skip",         type=int, default=1,
                        help="Run MediaPipe every N incoming frames (default: 1)")
    parser.add_argument("--skip-warp",      type=int, default=1, dest="skip_warp",
                        help="Redo triangle warp every N frames (default: 1)")
    parser.add_argument("--feather-radius", type=int, default=8, dest="feather_radius",
                        help="Feather edge softness in pixels (default: 8)")
    parser.add_argument("--webcam-port",  type=int, default=9001, dest="webcam_port",
                        help="Port on Pi serving webcam frames (default: 9001)")
    parser.add_argument("--output-port",  type=int, default=9002, dest="output_port",
                        help="Port on Pi receiving processed frames (default: 9002)")
    parser.add_argument("--landmark",     choices=["NORMAL", "REDUCED", "COARSE"], default="NORMAL")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"ERROR: video not found: {args.video_path}")
        sys.exit(1)

    npz_path = os.path.splitext(args.video_path)[0] + f"_{args.landmark.lower()}.npz"
    if not os.path.exists(npz_path):
        print(f"ERROR: pre-baked data not found: {npz_path}")
        print(f"       Run: python prebaker.py {args.video_path} --landmark {args.landmark}")
        sys.exit(1)

    landmark_indices = get_landmark_indices(args.landmark)

    threading.Thread(
        target=_webcam_client,
        args=(args.pi_host, args.webcam_port, landmark_indices, args.skip),
        daemon=True,
    ).start()

    threading.Thread(
        target=_output_client,
        args=(args.pi_host, args.output_port),
        daemon=True,
    ).start()

    _pipeline(args.video_path, npz_path, args.warp, args.width, args.height, args.skip_warp, args.landmark, args.feather_radius)
