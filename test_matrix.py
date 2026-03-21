"""
test_matrix.py

Displays a rotating square directly on the RGB LED matrix.
Run on the Pi with sudo.

Usage:
    sudo python3 test_matrix.py [--speed 1.0] [--size 1.0] [--led-*]

Options:
    --speed FLOAT   rotation speed multiplier (default: 1.0)
    --size  INT     canvas size in pixels to render at before sending to matrix (default: matrix width)
"""

import argparse
import math
import time
from PIL import Image, ImageDraw
from rgbmatrix import RGBMatrix, RGBMatrixOptions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Rotation speed multiplier (default: 1.0)")
    parser.add_argument("--size",  type=int, default=None,
                        help="Canvas size in pixels to render at (default: matrix width)")
    parser.add_argument("--led-rows",                 type=int,  default=64)
    parser.add_argument("--led-cols",                 type=int,  default=64)
    parser.add_argument("--led-chain",                type=int,  default=3,   dest="led_chain")
    parser.add_argument("--led-parallel",             type=int,  default=3,   dest="led_parallel")
    parser.add_argument("--led-pwm-bits",             type=int,  default=7,   dest="led_pwm_bits")
    parser.add_argument("--led-pwm-dither-bits",      type=int,  default=1,   dest="led_pwm_dither_bits")
    parser.add_argument("--led-pwm-lsb-nanoseconds",  type=int,  default=50,  dest="led_pwm_lsb_nanoseconds")
    parser.add_argument("--led-slowdown-gpio",        type=int,  default=3,   dest="led_slowdown_gpio")
    parser.add_argument("--led-brightness",           type=int,  default=100, dest="led_brightness")
    parser.add_argument("--led-hardware-mapping",     default="regular",      dest="led_hardware_mapping")
    parser.add_argument("--led-pixel-mapper",         default="",             dest="led_pixel_mapper")
    return parser.parse_args()


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
    options.drop_privileges     = False

    matrix = RGBMatrix(options=options)
    w, h   = matrix.width, matrix.height
    canvas = matrix.CreateFrameCanvas()

    size = args.size or w
    cx, cy = size / 2, size / 2
    half   = size / math.sqrt(2) / 2
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]

    angle              = 0.0
    degrees_per_second = 90.0 * args.speed
    t_last             = time.monotonic()

    print(f"Matrix: {w}x{h} | render size: {size}x{size} | speed: {args.speed} — Ctrl+C to stop")

    try:
        while True:
            t_now  = time.monotonic()
            angle += math.radians(degrees_per_second * (t_now - t_last))
            t_last = t_now

            cos_a, sin_a = math.cos(angle), math.sin(angle)
            pts = [
                (cx + x * cos_a - y * sin_a,
                 cy + x * sin_a + y * cos_a)
                for x, y in corners
            ]

            img  = Image.new("RGB", (size, size), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.polygon(pts, fill=(0, 80, 180))
            if size != w or size != h:
                img = img.resize((w, h), Image.BILINEAR)

            canvas.SetImage(img)
            canvas = matrix.SwapOnVSync(canvas)
            canvas.SetImage(img)

    except KeyboardInterrupt:
        matrix.Clear()
        print("\nDone.")


if __name__ == "__main__":
    main()
