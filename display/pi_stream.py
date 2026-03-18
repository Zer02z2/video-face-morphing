"""
pi_stream.py

Runs on Raspberry Pi. Start this first — it listens on two ports:
  1. Webcam server (9001):  Mac connects to read webcam frames.
  2. Display server (9002): Mac connects to push processed frames → RGB matrix.

Usage:
    sudo python3 pi_stream.py [options]

Options:
    --webcam-port INT     port Mac will read webcam frames from (default: 9001)
    --output-port INT     port Mac will push processed frames to (default: 9002)
    --webcam-device INT   webcam index (default: 0)
    --webcam-width INT    capture width (default: 640)
    --webcam-height INT   capture height (default: 480)
    --led-*               RGB matrix hardware options
"""

import argparse
import io
import socket
import struct
import time
import threading
import cv2
from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions


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


def _send_frame(sock: socket.socket, jpeg: bytes) -> None:
    header = struct.pack(">I", len(jpeg))
    sock.sendall(header + jpeg)


# ---------------------------------------------------------------------------
# Thread 1: webcam server — Mac connects to read frames
# ---------------------------------------------------------------------------

def _webcam_server(webcam_port: int, device: int, width: int, height: int) -> None:
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"ERROR: could not open webcam device {device}")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", webcam_port))
        srv.listen(1)
        print(f"[webcam] Listening on port {webcam_port} — waiting for Mac...")
        while True:
            try:
                conn, addr = srv.accept()
                print(f"[webcam] Mac connected: {addr}")
                with conn:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            continue
                        try:
                            _send_frame(conn, jpeg.tobytes())
                        except (BrokenPipeError, ConnectionResetError):
                            print("[webcam] Mac disconnected.")
                            break
            except Exception as e:
                print(f"[webcam] Error: {e}")


# ---------------------------------------------------------------------------
# Thread 2: display server — Mac connects to push processed frames
# ---------------------------------------------------------------------------

def _matrix_server(output_port: int, matrix: RGBMatrix) -> None:
    matrix_w = matrix.width
    matrix_h = matrix.height
    canvas   = matrix.CreateFrameCanvas()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", output_port))
        srv.listen(1)
        print(f"[matrix] Listening on port {output_port} — waiting for Mac...")
        while True:
            try:
                conn, addr = srv.accept()
                print(f"[matrix] Mac connected: {addr}")
                with conn:
                    while True:
                        raw = _recvall(conn, 4)
                        if not raw:
                            break
                        length = struct.unpack(">I", raw)[0]
                        jpeg   = _recvall(conn, length)
                        if not jpeg:
                            break
                        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                        img = img.resize((matrix_w, matrix_h), Image.BILINEAR)
                        canvas.SetImage(img)
                        canvas = matrix.SwapOnVSync(canvas)
                        canvas.SetImage(img)
                print("[matrix] Mac disconnected.")
            except Exception as e:
                print(f"[matrix] Error: {e}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pi stream: webcam server + matrix display server")

    parser.add_argument("--webcam-port",   type=int, default=9001,    dest="webcam_port")
    parser.add_argument("--output-port",   type=int, default=9002,    dest="output_port")
    parser.add_argument("--webcam-device", type=int, default=0,       dest="webcam_device")
    parser.add_argument("--webcam-width",  type=int, default=640,     dest="webcam_width")
    parser.add_argument("--webcam-height", type=int, default=480,     dest="webcam_height")

    parser.add_argument("--led-rows",                type=int,  default=64)
    parser.add_argument("--led-cols",                type=int,  default=64)
    parser.add_argument("--led-chain",               type=int,  default=3,   dest="led_chain")
    parser.add_argument("--led-parallel",            type=int,  default=3,   dest="led_parallel")
    parser.add_argument("--led-pwm-bits",            type=int,  default=7,   dest="led_pwm_bits")
    parser.add_argument("--led-pwm-dither-bits",     type=int,  default=1,   dest="led_pwm_dither_bits")
    parser.add_argument("--led-pwm-lsb-nanoseconds", type=int, default=50,   dest="led_pwm_lsb_nanoseconds")
    parser.add_argument("--led-slowdown-gpio",       type=int,  default=3,   dest="led_slowdown_gpio")
    parser.add_argument("--led-brightness",          type=int,  default=100, dest="led_brightness")
    parser.add_argument("--led-hardware-mapping",    default="regular",      dest="led_hardware_mapping")
    parser.add_argument("--led-pixel-mapper",        default="",             dest="led_pixel_mapper")
    parser.add_argument("--led-show-refresh",        action="store_true", default=False, dest="led_show_refresh")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

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

    matrix = RGBMatrix(options=options)
    print(f"Matrix: {matrix.width}x{matrix.height}")

    threading.Thread(
        target=_webcam_server,
        args=(args.webcam_port, args.webcam_device, args.webcam_width, args.webcam_height),
        daemon=True,
    ).start()

    try:
        _matrix_server(args.output_port, matrix)
    except KeyboardInterrupt:
        matrix.Clear()
        print("\nShutting down.")


if __name__ == "__main__":
    main()
