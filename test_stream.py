"""
test_stream.py

Listens for a TCP connection from matrix_display.py on the Pi and streams
a rotating square to it.

Usage:
    python test_stream.py [--port 9002] [--speed 1.0]

Then on Pi:
    sudo python3 display/matrix_display.py --host <mac-ip> --port 9002

Options:
    --port INT      port to listen on (default: 9002)
    --speed FLOAT   rotation speed multiplier (default: 1.0, higher = faster)
    --width INT     frame width (default: 192)
    --height INT    frame height (default: 192)
"""

import argparse
import math
import socket
import struct
import time
import cv2
import numpy as np


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def _rotate_points(pts: np.ndarray, angle_rad: float, cx: float, cy: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    pts = pts.astype(np.float32)
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    rotated = np.empty_like(pts)
    rotated[:, 0] = pts[:, 0] * cos_a - pts[:, 1] * sin_a
    rotated[:, 1] = pts[:, 0] * sin_a + pts[:, 1] * cos_a
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


def stream(port: int, speed: float, width: int, height: int) -> None:
    local_ip = _get_local_ip()
    print(f"Listening on {local_ip}:{port}")
    print(f"On Pi, run: sudo python3 display/matrix_display.py --host {local_ip} --port {port}")

    cx, cy = width / 2, height / 2
    side = min(width, height) / math.sqrt(2)
    half = side / 2
    square = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])

    angle = 0.0
    degrees_per_second = 90.0 * speed

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        while True:
            conn, addr = srv.accept()
            print(f"Pi connected: {addr}")
            t_last = time.monotonic()
            with conn:
                while True:
                    t_now = time.monotonic()
                    dt = t_now - t_last
                    t_last = t_now
                    angle += math.radians(degrees_per_second * dt)

                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    pts = _rotate_points(square, angle, cx, cy).astype(np.int32)
                    cv2.fillPoly(frame, [pts.reshape(-1, 1, 2)], color=(0, 80, 180))

                    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        continue

                    data = jpeg.tobytes()
                    try:
                        conn.sendall(struct.pack(">I", len(data)) + data)
                    except (BrokenPipeError, ConnectionResetError):
                        print(f"Pi disconnected: {addr}")
                        break

                    time.sleep(1 / 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int,   default=9002)
    parser.add_argument("--speed",  type=float, default=1.0,
                        help="Rotation speed multiplier (default: 1.0)")
    parser.add_argument("--width",  type=int,   default=192)
    parser.add_argument("--height", type=int,   default=192)
    args = parser.parse_args()

    stream(args.port, args.speed, args.width, args.height)
