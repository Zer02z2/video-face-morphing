"""
prebaker.py

Offline script. Run once per video to extract and save facial landmarks.

Usage:
    python prebaker.py <video_path>

Output:
    <video_path_without_extension>.npz alongside the video file
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp

from face_model import create_video_landmarker, landmarks_to_numpy, LANDMARK_COUNT, get_landmark_indices


def prebake(video_path: str, landmark_mode: str = "NORMAL") -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmark_indices = get_landmark_indices(landmark_mode)
    lm_count = len(landmark_indices) if landmark_indices is not None else LANDMARK_COUNT
    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames  |  landmark mode: {landmark_mode} ({lm_count} pts)")

    landmarks_out = np.zeros((total_frames, lm_count, 2), dtype=np.int16)
    detected_out = np.zeros(total_frames, dtype=bool)

    with create_video_landmarker() as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                pts = landmarks_to_numpy(result.face_landmarks[0], width, height, landmark_indices)
                x_span = pts[:, 0].max() - pts[:, 0].min()
                y_span = pts[:, 1].max() - pts[:, 1].min()
                if y_span > 0 and (x_span / y_span) >= 0.7:
                    landmarks_out[frame_idx] = pts
                    detected_out[frame_idx] = True

            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                pct = frame_idx / total_frames * 100
                print(f"  {frame_idx}/{total_frames} ({pct:.1f}%)", end="\r")

    cap.release()
    print()

    detected_count = detected_out.sum()
    print(f"Face detected in {detected_count}/{total_frames} frames ({detected_count/total_frames*100:.1f}%)")

    # --- Remove short detection runs (< 6 frames) ---
    i = 0
    while i < total_frames:
        if detected_out[i]:
            j = i
            while j < total_frames and detected_out[j]:
                j += 1
            if j - i < 6:
                detected_out[i:j] = False
            i = j
        else:
            i += 1

    kept = detected_out.sum()
    print(f"After removing short runs: {kept}/{total_frames} frames ({kept/total_frames*100:.1f}%)")

    # --- Compute per-frame fade weights ---
    # First 3 frames of each run: ramp up (1/3, 2/3, 1.0)
    # Last 3 frames of each run:  ramp down (2/3, 1/3, 0... approaching 0 at end)
    # Short runs (3-5 frames): take min(ramp_in, ramp_out) so they blend correctly
    fade_weights = np.zeros(total_frames, dtype=np.float32)
    i = 0
    while i < total_frames:
        if detected_out[i]:
            j = i
            while j < total_frames and detected_out[j]:
                j += 1
            run_len = j - i
            for k in range(run_len):
                ramp_in  = min(1.0, (k + 1) / 3.0)
                ramp_out = min(1.0, (run_len - k) / 3.0)
                fade_weights[i + k] = min(ramp_in, ramp_out)
            i = j
        else:
            i += 1

    # --- Compute neutral pose: average of first 30 detected frames ---
    detected_indices = np.where(detected_out)[0][:30]
    neutral = landmarks_out[detected_indices].mean(axis=0).astype(np.float32)

    output_path = os.path.splitext(video_path)[0] + f"_{landmark_mode.lower()}.npz"
    np.savez_compressed(
        output_path,
        landmarks=landmarks_out,
        detected=detected_out,
        fade_weights=fade_weights,
        meta=np.array([fps, width, height, total_frames]),
        neutral=neutral,
    )
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--landmark", choices=["NORMAL", "REDUCED"], default="NORMAL",
                        help="NORMAL: all 478 landmarks  REDUCED: 68-point subset (~7x faster warp)")
    args = parser.parse_args()
    prebake(args.video_path, args.landmark)
