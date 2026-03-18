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

LANDMARK_COUNT = 478  # full MediaPipe set

# 68-point subset — ~130 triangles vs ~900, ~7x faster warp loop on Pi
REDUCED_LANDMARK_INDICES = [
    # Jaw line (17)
    162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    # Right eyebrow (5)
    70, 63, 105, 66, 107,
    # Left eyebrow (5)
    336, 296, 334, 293, 300,
    # Nose bridge (4)
    168, 197, 195, 5,
    # Nose bottom (5)
    64, 98, 97, 327, 294,
    # Right eye (6)
    33, 160, 158, 133, 153, 144,
    # Left eye (6)
    362, 385, 387, 263, 373, 380,
    # Outer mouth (12)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner mouth (8)
    78, 82, 13, 312, 308, 317, 14, 87,
]


def get_landmark_indices(mode: str) -> list | None:
    """Return landmark index list for the given mode, or None for full 478."""
    if mode.upper() == "REDUCED":
        return REDUCED_LANDMARK_INDICES
    return None

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


def landmarks_to_numpy(face_landmarks, width: int, height: int, indices: list | None = None):
    """
    Convert a mediapipe FaceLandmarkerResult face_landmarks[0] to (N, 2) int16 array.
    If indices is provided, only those landmark positions are returned.
    """
    import numpy as np
    all_pts = np.array(
        [(int(l.x * width), int(l.y * height)) for l in face_landmarks],
        dtype=np.int16,
    )
    if indices is not None:
        return all_pts[indices]
    return all_pts
