"""
warp_engine.py

Module: given two sets of facial landmarks and a source frame, warps the source
face into the shape/position described by the destination landmarks.

Main entry point:
    warped, mask = warp_face(src_frame, src_landmarks, dst_landmarks, dst_size)

Standalone test (draws Delaunay triangulation on live webcam feed):
    python warp_engine.py
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Delaunay triangulation
# ---------------------------------------------------------------------------

def get_triangle_indices(
    landmarks: np.ndarray,
    frame_size: tuple[int, int],
) -> list[tuple[int, int, int]]:
    """
    Compute Delaunay triangulation over landmark points.

    Args:
        landmarks:   (N, 2) array of (x, y) pixel coordinates
        frame_size:  (height, width) — defines the bounding rect for Subdiv2D

    Returns:
        List of (i, j, k) index triples into the landmarks array.
        Call once per video frame (pre-baked landmarks change shape frame to frame)
        and cache the result for that frame index.
    """
    h, w = frame_size
    subdiv = cv2.Subdiv2D((0, 0, w, h))

    for x, y in landmarks:
        # clamp to stay inside Subdiv2D rect
        subdiv.insert((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))

    # getTriangleList returns (N, 6): x1,y1,x2,y2,x3,y3 as float
    triangle_list = subdiv.getTriangleList()

    # Reverse lookup: rounded coordinate -> landmark index
    coord_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(landmarks)}

    indices = []
    for t in triangle_list:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]

        # Discard triangles with any vertex outside the frame
        if not all(0 <= p[0] < w and 0 <= p[1] < h for p in pts):
            continue

        tri_idx = [coord_to_idx.get(p) for p in pts]
        if None not in tri_idx:
            indices.append(tuple(tri_idx))

    return indices


# ---------------------------------------------------------------------------
# Per-triangle affine warp
# ---------------------------------------------------------------------------

def _warp_triangle(
    src: np.ndarray,
    dst: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
) -> None:
    """
    Affine-warp one triangle from src into dst (modifies dst in-place).

    Args:
        src:      source BGR image (webcam frame)
        dst:      destination BGR image (being built up, same size as video frame)
        src_tri:  (3, 2) float32 — triangle vertices in src
        dst_tri:  (3, 2) float32 — triangle vertices in dst
    """
    sx, sy, sw, sh = cv2.boundingRect(src_tri)
    dx, dy, dw, dh = cv2.boundingRect(dst_tri)

    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return

    # Clip dst bounding rect to image bounds
    dst_h, dst_w = dst.shape[:2]
    dx2, dy2 = min(dx + dw, dst_w), min(dy + dh, dst_h)
    if dx >= dst_w or dy >= dst_h:
        return

    src_tri_off = src_tri - np.array([sx, sy], dtype=np.float32)
    dst_tri_off = dst_tri - np.array([dx, dy], dtype=np.float32)

    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_tri_off.astype(np.int32), 255)

    src_patch = src[sy:sy + sh, sx:sx + sw]
    if src_patch.size == 0:
        return

    M = cv2.getAffineTransform(src_tri_off, dst_tri_off)
    warped = cv2.warpAffine(
        src_patch, M, (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Paste only inside the triangle mask
    dst_roi = dst[dy:dy2, dx:dx2]
    mask_roi = mask[:dy2 - dy, :dx2 - dx]
    warped_roi = warped[:dy2 - dy, :dx2 - dx]

    np.copyto(dst_roi, warped_roi, where=mask_roi[:, :, np.newaxis].astype(bool))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warp_face(
    src_frame: np.ndarray,
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    dst_size: tuple[int, int],
    triangle_indices: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the face in src_frame into the shape/position of dst_landmarks.

    Args:
        src_frame:        BGR webcam frame
        src_landmarks:    (468, 2) int16 — landmarks from webcam detector
        dst_landmarks:    (468, 2) int16 — pre-baked landmarks for this video frame
        dst_size:         (height, width) of the output (should match video frame)
        triangle_indices: output of get_triangle_indices(dst_landmarks, dst_size).
                          Pass a cached value to avoid recomputing every frame.
                          If None, computed here (slower).

    Returns:
        warped:  (H, W, 3) BGR image — warped face pixels, black elsewhere
        mask:    (H, W)    uint8      — 255 inside face convex hull, 0 outside
    """
    h, w = dst_size
    warped = np.zeros((h, w, 3), dtype=np.uint8)

    if triangle_indices is None:
        triangle_indices = get_triangle_indices(dst_landmarks, dst_size)

    for i0, i1, i2 in triangle_indices:
        src_tri = src_landmarks[[i0, i1, i2]].astype(np.float32)
        dst_tri = dst_landmarks[[i0, i1, i2]].astype(np.float32)
        _warp_triangle(src_frame, warped, src_tri, dst_tri)

    # Mask = convex hull of destination landmarks
    hull = cv2.convexHull(dst_landmarks.astype(np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    return warped, mask


# ---------------------------------------------------------------------------
# Standalone test — draws Delaunay triangulation on live webcam feed
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import mediapipe as mp
    from face_model import create_image_landmarker, landmarks_to_numpy

    print("Warp engine test — shows Delaunay triangulation on webcam")
    print("Press Q to quit")

    cap = cv2.VideoCapture(0)

    with create_image_landmarker() as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            vis = frame.copy()

            if result.face_landmarks:
                landmarks = landmarks_to_numpy(result.face_landmarks[0], w, h)
                indices = get_triangle_indices(landmarks, (h, w))

                for i0, i1, i2 in indices:
                    pts = landmarks[[i0, i1, i2]].astype(np.int32)
                    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 200, 0), thickness=1)

                cv2.putText(vis, f"{len(indices)} triangles", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            else:
                cv2.putText(vis, "no face detected", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            cv2.imshow("warp_engine test", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
