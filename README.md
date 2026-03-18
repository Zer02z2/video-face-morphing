# Video Face Morphing

Real-time face replacement pipeline: warps a live webcam face onto a looping video character using MediaPipe landmark detection, Delaunay triangulation, and affine warping. Output can be previewed in a browser or displayed on an RGB LED matrix via Raspberry Pi.

---

## Project Structure

```
.
├── prebaker.py              # Offline: extract & save landmarks from a video
├── main_rpi.py              # Pipeline for Pi 5 (or any Linux host) — standalone
├── main_stream.py           # Pipeline for Mac — streams to/from a Pi 4
├── core/
│   ├── face_model.py        # MediaPipe model download + landmarker factory
│   ├── webcam_detector.py   # Webcam capture + MediaPipe detection
│   ├── warp_engine.py       # Delaunay triangulation + vectorized affine warp
│   └── compositor.py        # Feathering, color correction, alpha blending
└── display/
    ├── browser_display.py   # MJPEG Flask server for browser preview
    ├── matrix_display.py    # RGB LED matrix display (Pi 4, standalone)
    └── pi_stream.py         # Pi 4 combined webcam server + matrix display
```

---

## Python Environment Setup (pyenv)

Use **Python 3.11** — it has the best compatibility with MediaPipe, OpenCV, and Numba.

### 1. Install pyenv

**macOS:**
```bash
brew install pyenv
```

**Linux / Raspberry Pi OS:**
```bash
curl https://pyenv.run | bash
```

Add to your shell profile (`~/.zshrc`, `~/.bashrc`, or `~/.profile`):
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload your shell:
```bash
exec "$SHELL"
```

### 2. Install build dependencies

**macOS** (Homebrew):
```bash
brew install openssl readline sqlite3 xz zlib
```

**Raspberry Pi OS / Debian:**
```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### 3. Install Python 3.11 and create a virtual environment

```bash
pyenv install 3.11.9
pyenv local 3.11.9          # pins this version in the project directory

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 4. Install Python dependencies

**Mac / Pi 5 (full pipeline):**
```bash
pip install opencv-python mediapipe numpy numba flask Pillow
```

**Pi 4 display-only scripts** (`matrix_display.py`): these use the system Python that comes with the rgbmatrix library — see the RGB Matrix Setup section below.

---

## Step 1: Pre-bake Landmarks (run once per video)

Before running the live pipeline you must extract facial landmarks from the video file and save them to a `.npz` file alongside the video.

```bash
python prebaker.py path/to/video.mp4
```

This creates `path/to/video_normal.npz`. You can also generate a faster reduced or coarse set:

```bash
python prebaker.py path/to/video.mp4 --landmark REDUCED   # 68-point subset
python prebaker.py path/to/video.mp4 --landmark COARSE    # 20-point subset
```

| Mode | Landmarks | Triangles | Speed |
|------|-----------|-----------|-------|
| NORMAL | 478 | ~900 | slowest |
| REDUCED | 68 | ~130 | ~7x faster warp |
| COARSE | 20 | ~30 | ~30x faster warp |

The MediaPipe model (~6 MB) is downloaded automatically on first use.

---

## Deployment Options

There are three ways to run the pipeline depending on your hardware.

---

### Option A: Mac + Browser Preview

Run on Mac. Preview output in a browser at `http://localhost:9003`.

**Terminal 1 — pipeline:**
```bash
python main_rpi.py path/to/video.mp4 --landmark COARSE
```

**Terminal 2 — browser display:**
```bash
python display/browser_display.py
```

Open `http://localhost:9003` in a browser.

**All flags for `main_rpi.py`:**
```
--warp 0|1              0: full face swap (default)  1: expression transfer
--width INT             processing width (default: video width)
--height INT            processing height (default: video height)
--skip INT              run MediaPipe every N frames (default: 1)
--skip-warp INT         redo warp every N frames (default: 1)
--feather-radius INT    edge softness in pixels (default: 8)
--no-color-correct      disable color correction for speed
--landmark NORMAL|REDUCED|COARSE
--port INT              TCP port to serve frames on (default: 9002)
```

---

### Option B: Pi 5 (compute) + Pi 4 (RGB matrix display)

Pi 5 runs the full pipeline. Pi 4 connects to it and drives the LED matrix.

**On Pi 5 — terminal 1:**
```bash
python main_rpi.py path/to/video.mp4 --landmark COARSE --port 9002
```

**On Pi 4 — terminal 1:**
```bash
sudo python3 display/matrix_display.py --host <pi5-ip> --port 9002
```

---

### Option C: Mac (compute) + Pi 4 (webcam + RGB matrix)

Mac handles all computation. Pi 4 serves webcam frames to Mac and receives processed frames to display on the matrix.

**Step 1 — On Pi 4, start the stream server first:**
```bash
sudo python3 display/pi_stream.py
```

**Step 2 — On Mac, connect and run the pipeline:**
```bash
python main_stream.py path/to/video.mp4 --pi-host <pi4-ip> --landmark COARSE
```

**All flags for `main_stream.py`:**
```
--pi-host STR           IP of the Pi running pi_stream.py (required)
--warp 0|1              0: full face swap (default)  1: expression transfer
--width INT             processing width
--height INT            processing height
--skip INT              run MediaPipe every N frames (default: 1)
--skip-warp INT         redo warp every N frames (default: 1)
--feather-radius INT    edge softness in pixels (default: 8)
--no-color-correct      disable color correction
--landmark NORMAL|REDUCED|COARSE
--webcam-port INT       port Pi serves webcam on (default: 9001)
--output-port INT       port Pi receives processed frames on (default: 9002)
```

---

## RGB Matrix Setup (Raspberry Pi)

The `rgbmatrix` Python library must be installed from source and **only works with the system Python** (not a venv). Run matrix display scripts with `sudo`.

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-dev python3-pillow python3-pip git
```

### 2. Clone and build rpi-rgb-led-matrix

```bash
git clone https://github.com/hzeller/rpi-rgb-led-matrix.git
cd rpi-rgb-led-matrix

make build-python PYTHON=$(which python3)
sudo make install-python PYTHON=$(which python3)
```

### 3. Isolate core 3 for the matrix refresh thread

The matrix library pins its refresh thread to a dedicated CPU core. To prevent the OS from scheduling work there, add `isolcpus=3` to the kernel boot parameters:

```bash
sudo nano /boot/firmware/cmdline.txt
```

Append `isolcpus=3` to the end of the existing line (do not add a new line):
```
... rootwait isolcpus=3
```

Reboot:
```bash
sudo reboot
```

### 4. Install Pillow for system Python

```bash
sudo pip3 install Pillow --break-system-packages
```

### 5. Run the display script

```bash
sudo python3 display/matrix_display.py --host <compute-host-ip> --port 9002
```

Or with custom matrix hardware settings:
```bash
sudo python3 display/matrix_display.py \
  --host <compute-host-ip> \
  --led-rows 64 \
  --led-cols 64 \
  --led-chain 3 \
  --led-parallel 3 \
  --led-brightness 80 \
  --led-slowdown-gpio 3
```

**All `--led-*` flags:**
```
--led-rows INT                  panel height in pixels (default: 64)
--led-cols INT                  panel width in pixels (default: 64)
--led-chain INT                 number of panels chained (default: 3)
--led-parallel INT              number of parallel chains (default: 3)
--led-pwm-bits INT              color depth, 1–11 (default: 7)
--led-pwm-dither-bits INT       dithering bits (default: 1)
--led-pwm-lsb-nanoseconds INT   PWM timing (default: 50)
--led-slowdown-gpio INT         GPIO slowdown for faster Pi (default: 3)
--led-brightness INT            0–100 (default: 100)
--led-hardware-mapping STR      pin mapping name (default: regular)
--led-pixel-mapper STR          pixel mapper config (default: "")
--led-show-refresh              print refresh rate to stdout
```

---

## Pi 4 Stream Server (`pi_stream.py`) — Option C only

`pi_stream.py` runs on Pi 4 and combines both roles: webcam server (Mac reads frames from it) and matrix display server (Mac pushes processed frames to it).

```bash
sudo python3 display/pi_stream.py
```

With options:
```bash
sudo python3 display/pi_stream.py \
  --webcam-port 9001 \
  --output-port 9002 \
  --webcam-device 0 \
  --webcam-width 640 \
  --webcam-height 480 \
  --led-rows 64 \
  --led-cols 64 \
  --led-chain 3 \
  --led-parallel 3 \
  --led-brightness 80
```

---

## Browser Display

`browser_display.py` connects to the pipeline's TCP stream and re-serves it as MJPEG over HTTP.

```bash
python display/browser_display.py --host localhost --port 9002 --serve-port 9003
```

Open `http://localhost:9003` in a browser.

---

## Warp Modes

| Mode | Flag | Description |
|------|------|-------------|
| Full face swap | `--warp 0` (default) | Warps your entire face into the shape of the video character's face |
| Expression transfer | `--warp 1` | Extracts your facial expression delta and applies it to the character's neutral pose |

Expression transfer requires the `.npz` to contain a neutral pose (prebaker computes this automatically from the first 30 detected frames).

---

## Performance Tips

- Use `--landmark COARSE` for maximum speed — 20 landmarks produce ~30 triangles vs ~900 for NORMAL.
- Use `--no-color-correct` to disable per-warp color matching if the color difference is acceptable.
- Use `--skip-warp 3` or higher to reuse the last warp for several frames — the blend is still per-frame but the expensive Delaunay solve is skipped.
- Numba JIT is used automatically if `numba` is installed — it compiles the pixel mapping loop to native ARM/x86 code on first run and caches the binary for subsequent runs.
