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


def prebake(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    landmarks_out = np.zeros((total_frames, 468, 2), dtype=np.int16)
    detected_out = np.zeros(total_frames, dtype=bool)

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts = np.array(
                    [(int(p.x * width), int(p.y * height)) for p in lm],
                    dtype=np.int16,
                )
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

    output_path = os.path.splitext(video_path)[0] + ".npz"
    np.savez_compressed(
        output_path,
        landmarks=landmarks_out,
        detected=detected_out,
        meta=np.array([fps, width, height, total_frames]),
    )
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prebaker.py <video_path>")
        sys.exit(1)
    prebake(sys.argv[1])
