"""
face_model.py

Shared helper: downloads the MediaPipe face landmarker model on first use
and provides a factory for creating landmarker instances.

MediaPipe 0.10+ uses the Tasks API with an explicit .task model file.
The face_landmarker model produces 478 landmarks (468 face mesh + 10 iris).
"""

import os
import urllib.request
import mediapipe as mp

LANDMARK_COUNT = 478

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def ensure_model() -> str:
    """Download the model file if not present. Returns local path."""
    if not os.path.exists(_MODEL_PATH):
        print("Downloading face_landmarker.task model (~6MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH


def create_video_landmarker():
    """
    Landmarker for pre-recorded video / offline processing.
    Call detect_for_video(mp_image, timestamp_ms) per frame.
    Timestamps must be monotonically increasing.
    """
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def create_image_landmarker():
    """
    Landmarker for single images / live webcam frames.
    Call detect(mp_image) per frame. No timestamp required.
    """
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def landmarks_to_numpy(face_landmarks, width: int, height: int):
    """Convert a mediapipe FaceLandmarkerResult face_landmarks[0] to (N, 2) int16 array."""
    import numpy as np
    return np.array(
        [(int(l.x * width), int(l.y * height)) for l in face_landmarks],
        dtype=np.int16,
    )
