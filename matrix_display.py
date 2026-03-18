"""
matrix_display.py

Reads the MJPEG stream from main.py and displays it on an RGB LED matrix.
Run with system Python (no mediapipe needed — only Pillow + rgbmatrix).

Usage:
    sudo python3 matrix_display.py [--host localhost] [--port 9002] \
        --led-chain=3 --led-parallel=3 --led-rows=64 --led-cols=64 \
        --led-slowdown-gpio=3
"""

import argparse
import io
import time
import urllib.request

from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions


# ---------------------------------------------------------------------------
# MJPEG stream reader
# ---------------------------------------------------------------------------

def iter_frames(host: str, port: int):
    """
    Generator that yields PIL Images from an MJPEG stream.
    Reconnects automatically if the connection drops.
    """
    url = f"http://{host}:{port}/stream"
    while True:
        try:
            print(f"Connecting to {url} ...")
            stream = urllib.request.urlopen(url, timeout=10)
            print("Connected.")
            buf = b""
            while True:
                buf += stream.read(4096)
                a = buf.find(b"\xff\xd8")  # JPEG start marker
                b = buf.find(b"\xff\xd9")  # JPEG end marker
                if a != -1 and b != -1:
                    jpg = buf[a:b + 2]
                    buf = buf[b + 2:]
                    yield Image.open(io.BytesIO(jpg)).convert("RGB")
        except Exception as e:
            print(f"Stream error: {e} — retrying in 2s...")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RGB Matrix MJPEG display")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9002)

    parser.add_argument("--led-rows",                type=int,  default=64)
    parser.add_argument("--led-cols",                type=int,  default=64)
    parser.add_argument("--led-chain",               type=int,  default=1,   dest="led_chain")
    parser.add_argument("--led-parallel",            type=int,  default=1,   dest="led_parallel")
    parser.add_argument("--led-pwm-bits",            type=int,  default=7,   dest="led_pwm_bits")
    parser.add_argument("--led-pwm-dither-bits",     type=int,  default=1,   dest="led_pwm_dither_bits")
    parser.add_argument("--led-pwm-lsb-nanoseconds", type=int,  default=50,  dest="led_pwm_lsb_nanoseconds")
    parser.add_argument("--led-slowdown-gpio",       type=int,  default=3,   dest="led_slowdown_gpio")
    parser.add_argument("--led-brightness",          type=int,  default=100, dest="led_brightness")
    parser.add_argument("--led-hardware-mapping",    default="regular",      dest="led_hardware_mapping")
    parser.add_argument("--led-pixel-mapper",        default="",             dest="led_pixel_mapper")
    parser.add_argument("--led-show-refresh",        action="store_true",    dest="led_show_refresh")

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

    matrix   = RGBMatrix(options=options)
    matrix_w = matrix.width
    matrix_h = matrix.height
    print(f"Matrix: {matrix_w}x{matrix_h}")

    canvas = matrix.CreateFrameCanvas()

    try:
        for img in iter_frames(args.host, args.port):
            img = img.resize((matrix_w, matrix_h), Image.BILINEAR)
            canvas.SetImage(img)
            canvas = matrix.SwapOnVSync(canvas)
            canvas.SetImage(img)
    except KeyboardInterrupt:
        matrix.Clear()
        print("\nShutting down.")


if __name__ == "__main__":
    main()
