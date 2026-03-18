"""
compositor.py

Module: blends a warped face patch onto a video frame and returns the final
composed image as a raw BGR numpy array (the "buffer").

Main entry point:
    result = composite(video_frame, warped_face, mask)

Encoding to JPEG (for MJPEG stream) is done in main.py, not here.
Pushing to an RGB matrix (for Pi) is also done outside this module.

Standalone test (blends a shifted copy of your webcam face onto itself):
    python compositor.py
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color correction
# ---------------------------------------------------------------------------

def _match_color(
    src: np.ndarray,
    dst_frame: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Adjust src colors so its mean/std per channel matches the corresponding
    region in dst_frame. This compensates for lighting and skin-tone differences.

    Args:
        src:       warped face BGR image (full frame size)
        dst_frame: video frame BGR image (full frame size)
        mask:      uint8 mask — 255 inside face region

    Returns:
        Color-corrected copy of src (same shape/dtype).
    """
    result = src.astype(np.float32)
    face_px = mask > 0

    for c in range(3):
        src_pixels = src[:, :, c][face_px].astype(np.float32)
        dst_pixels = dst_frame[:, :, c][face_px].astype(np.float32)

        if src_pixels.std() < 1e-6:
            continue

        scale = dst_pixels.std() / src_pixels.std()
        shift = dst_pixels.mean() - src_pixels.mean() * scale

        result[:, :, c] = np.clip(result[:, :, c] * scale + shift, 0, 255)

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask feathering
# ---------------------------------------------------------------------------

def _feather_mask(mask: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    Erode then blur the mask to create a soft feathered edge.
    Returns a float32 mask in [0.0, 1.0].
    """
    ksize = max(1, blur_radius // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(mask, kernel, iterations=1)
    blurred = cv2.GaussianBlur(eroded, (blur_radius | 1, blur_radius | 1), 0)
    return blurred.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def composite(
    video_frame: np.ndarray,
    warped_face: np.ndarray,
    mask: np.ndarray,
    color_correct: bool = True,
    feather_radius: int = 50,
) -> np.ndarray:
    """
    Blend warped_face onto video_frame using a feathered alpha mask.

    Args:
        video_frame:    BGR video frame (the base canvas)
        warped_face:    BGR warped face output from warp_engine.warp_face()
        mask:           uint8 mask from warp_engine.warp_face() — 255 inside face
        color_correct:  if True, adjust warped_face colors to match video lighting
        feather_radius: size of the soft edge blend (pixels). Higher = softer.

    Returns:
        Composed BGR frame as uint8 numpy array — ready to encode or display.
    """
    if color_correct:
        warped_face = _match_color(warped_face, video_frame, mask)

    alpha = _feather_mask(mask, feather_radius)          # (H, W) float32 [0,1]
    alpha3 = alpha[:, :, np.newaxis]                     # (H, W, 1) for broadcast

    base = video_frame.astype(np.float32)
    face = warped_face.astype(np.float32)

    blended = base * (1.0 - alpha3) + face * alpha3
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import mediapipe as mp
    from warp_engine import get_triangle_indices, warp_face
    from face_model import create_image_landmarker, landmarks_to_numpy

    print("Compositor test — warps and blends your face onto a shifted copy of itself")
    print("Press Q to quit, C to toggle color correction, F to toggle feathering")

    cap = cv2.VideoCapture(0)

    color_correct = True
    feather = True

    with create_image_landmarker() as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = landmarks_to_numpy(result.face_landmarks[0], w, h)

                # Shift destination landmarks slightly to demonstrate warp + blend
                offset = np.array([30, 20], dtype=np.int16)
                dst_landmarks = np.clip(landmarks + offset, 0, [w - 1, h - 1]).astype(np.int16)

                indices = get_triangle_indices(dst_landmarks, (h, w))
                warped, mask = warp_face(frame, landmarks, dst_landmarks, (h, w), indices)
                result = composite(
                    frame, warped, mask,
                    color_correct=color_correct,
                )

                # Labels
                cv2.putText(result, f"color_correct={color_correct}  feather={feather}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("compositor test", result)
            else:
                cv2.putText(frame, "no face detected", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("compositor test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                color_correct = not color_correct
            elif key == ord("f"):
                feather = not feather

    cap.release()
    cv2.destroyAllWindows()
