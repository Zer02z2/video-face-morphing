# Video Face Morphing

Real-time face replacement: maps your webcam face onto a character in a video using facial landmark detection and Delaunay triangulation warping.

Two warp modes:
- **Mode 0** — full mapping: your face is warped into the character's exact face shape
- **Mode 1** — expression transfer: the character's expressions are applied to your face, preserving your natural proportions

---

## 1. Setup

**Requires Python 3.10+**

```bash
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python numpy flask Pillow
```

On first run, the MediaPipe face landmarker model (~6MB) is downloaded automatically.

---

## 2. Pre-bake the video

Run once per video. Extracts facial landmarks from every frame and saves them alongside the video.

```bash
python prebaker.py path/to/video.mp4
```

This produces `path/to/video.npz` in the same directory. It also:
- Removes detection runs shorter than 6 frames (noise rejection)
- Filters out frames where the face is too turned sideways
- Pre-computes per-frame fade weights for smooth transitions

---

## 3. Run

### Browser (Mac / development)

```bash
python main.py path/to/video.mp4
```

Open **http://localhost:9002** in a browser.

**Warp mode:**
```bash
python main.py path/to/video.mp4 --warp 1
```

### RGB LED Matrix (Raspberry Pi)

```bash
sudo python3 main_matrix.py path/to/video.mp4 \
    --led-rows=64 --led-cols=64 \
    --led-chain=3 --led-parallel=3 \
    --led-slowdown-gpio=3
```

All `--led-*` flags default to sensible values if not provided. Pass only what differs from your physical setup. See `main_matrix.py --help` for the full list.

**Warp mode on Pi:**
```bash
sudo python3 main_matrix.py path/to/video.mp4 --warp 1 \
    --led-chain=3 --led-parallel=3
```
