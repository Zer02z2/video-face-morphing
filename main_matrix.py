"""
main_matrix.py

Pi version of main.py — outputs composed frames to an RGB LED matrix
instead of a Flask MJPEG stream.

Usage:
    sudo python3 main_matrix.py <video_path> [--warp 0|1] \
        --led-chain=3 --led-parallel=3 --led-rows=64 --led-cols=64 \
        --led-pwm-bits=7 --led-pwm-dither-bits=1 \
        --led-slowdown-gpio=3 --led-pwm-lsb-nanoseconds=50

The video's .npz file must already exist (run prebaker.py first).
"""

import sys
import os
import time
import argparse
import threading
import cv2
import numpy as np
from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions

from webcam_detector import WebcamDetector
from warp_engine import get_triangle_indices, warp_face
from compositor import composite


# ---------------------------------------------------------------------------
# Shared state between pipeline thread and display loop
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

    triangle_cache: dict[int, list | None] = {}

    def get_triangles(idx: int) -> list | None:
        if idx not in triangle_cache:
            triangle_cache[idx] = get_triangle_indices(video_landmarks[idx], (vid_h, vid_w)) if detected[idx] else None
        return triangle_cache[idx]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}")
        return

    print("Pipeline running.")

    with WebcamDetector(width=vid_w, height=vid_h) as detector:
        webcam_gen = detector.frames()
        frame_idx  = 0

        while True:
            loop_start = time.monotonic()

            ret, video_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            video_frame = cv2.resize(video_frame, (vid_w, vid_h))

            try:
                webcam_frame, webcam_landmarks, webcam_detected = next(webcam_gen)
            except StopIteration:
                print("Webcam closed.")
                break

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

            elapsed    = time.monotonic() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Face Morph — RGB Matrix output")

    parser.add_argument("video_path")
    parser.add_argument("--warp", type=int, choices=[0, 1], default=0,
                        help="0: full face mapping (default)  1: expression transfer")

    parser.add_argument("--led-rows",                 type=int,  default=64)
    parser.add_argument("--led-cols",                 type=int,  default=64)
    parser.add_argument("--led-chain",                type=int,  default=1,   dest="led_chain")
    parser.add_argument("--led-parallel",             type=int,  default=1,   dest="led_parallel")
    parser.add_argument("--led-pwm-bits",             type=int,  default=7,   dest="led_pwm_bits")
    parser.add_argument("--led-pwm-dither-bits",      type=int,  default=1,   dest="led_pwm_dither_bits")
    parser.add_argument("--led-pwm-lsb-nanoseconds",  type=int,  default=50,  dest="led_pwm_lsb_nanoseconds")
    parser.add_argument("--led-slowdown-gpio",        type=int,  default=3,   dest="led_slowdown_gpio")
    parser.add_argument("--led-brightness",           type=int,  default=100, dest="led_brightness")
    parser.add_argument("--led-hardware-mapping",     default="regular",      dest="led_hardware_mapping")
    parser.add_argument("--led-pixel-mapper",         default="",             dest="led_pixel_mapper")
    parser.add_argument("--led-show-refresh",         action="store_true",    dest="led_show_refresh")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        print(f"ERROR: video not found: {args.video_path}")
        sys.exit(1)

    npz_path = os.path.splitext(args.video_path)[0] + ".npz"
    if not os.path.exists(npz_path):
        print(f"ERROR: pre-baked data not found: {npz_path}")
        print(f"       Run: python prebaker.py {args.video_path}")
        sys.exit(1)

    options = RGBMatrixOptions()
    options.rows                = args.led_rows
    options.cols                = args.led_cols
    options.chain_length        = args.led_chain
    options.parallel            = args.led_parallel
    options.pwm_bits            = args.led_pwm_bits
    options.pwm_dither_bits     = args.led_pwm_dither_bits
    options.pwm_lsb_nanoseconds = args.led_pwm_lsb_nanoseconds
    options.gpio_slowdown       = args.led_slowdown_gpio
    options.brightness          = args.led_brightness
    options.hardware_mapping    = args.led_hardware_mapping
    options.pixel_mapper_config = args.led_pixel_mapper
    options.show_refresh_rate   = args.led_show_refresh
    options.drop_privileges     = False

    matrix   = RGBMatrix(options=options)
    matrix_w = matrix.width
    matrix_h = matrix.height
    print(f"[*] Matrix: {matrix_w}x{matrix_h} pixels")

    threading.Thread(
        target=_pipeline,
        args=(args.video_path, npz_path, args.warp),
        daemon=True,
    ).start()

    canvas = matrix.CreateFrameCanvas()

    try:
        while True:
            frame = _get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # BGR (OpenCV) → RGB, resize to matrix dimensions
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (matrix_w, matrix_h))

            pil_img = Image.fromarray(resized, "RGB")
            canvas.SetImage(pil_img)
            canvas = matrix.SwapOnVSync(canvas)
            canvas.SetImage(pil_img)  # sync back-buffer

    except KeyboardInterrupt:
        matrix.Clear()
        print("\n[*] Shutting down.")


if __name__ == "__main__":
    main()
