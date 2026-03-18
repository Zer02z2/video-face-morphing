"""
browser_display.py

Connects to main.py's TCP frame stream and serves it as MJPEG via Flask.
Run with the same Python/venv as main.py.

Usage:
    python browser_display.py [--host localhost] [--port 9002] [--serve-port 9003]

Open http://localhost:9003 in a browser.
"""

import argparse
import socket
import struct
import time
import threading
from flask import Flask, Response


# ---------------------------------------------------------------------------
# TCP frame reader — reconnects automatically
# ---------------------------------------------------------------------------

_latest_jpeg: bytes | None = None
_jpeg_lock = threading.Lock()


def _tcp_reader(host: str, port: int) -> None:
    global _latest_jpeg
    while True:
        try:
            print(f"Connecting to main.py at {host}:{port}...")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                print("Connected.")
                while True:
                    # Read 4-byte length header
                    raw = _recvall(s, 4)
                    if not raw:
                        break
                    length = struct.unpack(">I", raw)[0]
                    jpeg   = _recvall(s, length)
                    if not jpeg:
                        break
                    with _jpeg_lock:
                        _latest_jpeg = jpeg
        except Exception as e:
            print(f"Connection lost: {e} — retrying in 2s...")
            time.sleep(2)


def _recvall(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# Flask MJPEG server
# ---------------------------------------------------------------------------

app = Flask(__name__)


def _mjpeg_generator():
    last_sent = None
    while True:
        with _jpeg_lock:
            jpeg = _latest_jpeg
        if jpeg is None or jpeg is last_sent:
            time.sleep(0.005)
            continue
        last_sent = jpeg
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
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
    parser.add_argument("--host",       default="localhost")
    parser.add_argument("--port",       type=int, default=9002,
                        help="Port main.py is serving on")
    parser.add_argument("--serve-port", type=int, default=9003,
                        help="Port to serve browser on")
    args = parser.parse_args()

    threading.Thread(
        target=_tcp_reader, args=(args.host, args.port), daemon=True
    ).start()

    app.run(host="0.0.0.0", port=args.serve_port, debug=False, threaded=True)
