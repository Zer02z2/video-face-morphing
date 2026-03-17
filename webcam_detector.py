"""
webcam_detector.py

Module: captures webcam frames and returns facial landmarks via MediaPipe.

Yields (frame, landmarks, detected) per frame:
    frame      — BGR numpy array (H, W, 3)
    landmarks  — (478, 2) int16 array of pixel (x, y) coords, or None
    detected   — bool

Standalone test (draws landmarks on live feed):
    python webcam_detector.py
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Generator

from face_model import create_image_landmarker, landmarks_to_numpy


class WebcamDetector:
    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._landmarker = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at device {self.device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._landmarker = create_image_landmarker().__enter__()
        return self

    def __exit__(self, *args):
        if self._cap:
            self._cap.release()
        if self._landmarker:
            self._landmarker.__exit__(*args)

    def frames(self) -> Generator[tuple[np.ndarray, np.ndarray | None, bool], None, None]:
        """Yield (frame, landmarks, detected) until webcam closes or generator is stopped."""
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = landmarks_to_numpy(result.face_landmarks[0], w, h)
                yield frame, landmarks, True
            else:
                yield frame, None, False


# --- Standalone test ---

def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    for (x, y) in landmarks:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    return vis


if __name__ == "__main__":
    print("Webcam detector test — press Q to quit")

    with WebcamDetector() as detector:
        for frame, landmarks, detected in detector.frames():
            if detected:
                vis = _draw_landmarks(frame, landmarks)
                label = f"landmarks: {len(landmarks)}"
            else:
                vis = frame.copy()
                label = "no face detected"

            cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("webcam_detector test", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
