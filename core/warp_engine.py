"""
warp_engine.py

Warps a source face into the shape described by destination landmarks using
Delaunay triangulation + affine transforms.

Optimised: instead of N per-triangle warpAffine calls in a Python loop, builds
a full-frame remap in three vectorised passes:
  1. Triangle label map  — N fillConvexPoly calls (C++), one per triangle
  2. Batch inverse affines — single np.linalg.solve across all triangles at once
  3. Pixel mapping        — element-wise NumPy on all face pixels, then one cv2.remap

Main entry point:
    warped, mask = warp_face(src_frame, src_landmarks, dst_landmarks, dst_size)

Standalone test (draws Delaunay triangulation on live webcam feed):
    python -m core.warp_engine
"""

import cv2
import numpy as np

# Numba JIT — compiles the pixel mapping loop to native ARM/x86 code.
# parallel=True splits rows across CPU cores via OpenMP.
# cache=True saves the compiled binary so the first-run penalty only happens once.
# Falls back to plain NumPy if numba isn't installed.
try:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _build_remap(label_map: np.ndarray, affines: np.ndarray,
                     map_x: np.ndarray, map_y: np.ndarray) -> None:
        h, w = label_map.shape
        for y in numba.prange(h):
            for x in range(w):
                tri = label_map[y, x]
                if tri < 0:
                    continue
                map_x[y, x] = affines[tri, 0, 0] * x + affines[tri, 0, 1] * y + affines[tri, 0, 2]
                map_y[y, x] = affines[tri, 1, 0] * x + affines[tri, 1, 1] * y + affines[tri, 1, 2]

    _NUMBA = True
    print("[warp_engine] Numba JIT enabled")

except ImportError:
    _NUMBA = False


# ---------------------------------------------------------------------------
# Delaunay triangulation
# ---------------------------------------------------------------------------

def get_triangle_indices(
    landmarks: np.ndarray,
    frame_size: tuple[int, int],
) -> list[tuple[int, int, int]]:
    """
    Compute Delaunay triangulation over landmark points.

    Returns list of (i, j, k) index triples into the landmarks array.
    Cache the result per video frame — recomputing every frame is wasteful.
    """
    h, w = frame_size
    subdiv = cv2.Subdiv2D((0, 0, w, h))

    for x, y in landmarks:
        subdiv.insert((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))

    triangle_list = subdiv.getTriangleList()
    coord_to_idx  = {(int(x), int(y)): i for i, (x, y) in enumerate(landmarks)}

    indices = []
    for t in triangle_list:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        if not all(0 <= p[0] < w and 0 <= p[1] < h for p in pts):
            continue
        tri_idx = [coord_to_idx.get(p) for p in pts]
        if None not in tri_idx:
            indices.append(tuple(tri_idx))

    return indices


# ---------------------------------------------------------------------------
# Public API — vectorised warp
# ---------------------------------------------------------------------------

def warp_face(
    src_frame: np.ndarray,
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    dst_size: tuple[int, int],
    triangle_indices: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the face in src_frame into the shape of dst_landmarks.

    Args:
        src_frame:        BGR webcam frame
        src_landmarks:    (N, 2) int16 — landmarks from webcam detector
        dst_landmarks:    (N, 2) int16 — pre-baked landmarks for this video frame
        dst_size:         (height, width) of the output
        triangle_indices: cached output of get_triangle_indices — pass this to
                          avoid recomputing Delaunay every frame.

    Returns:
        warped:  (H, W, 3) BGR — warped face pixels, black elsewhere
        mask:    (H, W)    uint8 — 255 inside face convex hull, 0 outside
    """
    h, w = dst_size

    if triangle_indices is None:
        triangle_indices = get_triangle_indices(dst_landmarks, dst_size)

    if not triangle_indices:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    src_lm  = src_landmarks.astype(np.float32)
    dst_lm  = dst_landmarks.astype(np.float32)
    tri_idx = np.asarray(triangle_indices, dtype=np.int32)   # (n_tri, 3)
    n_tri   = len(tri_idx)

    dst_tris = dst_lm[tri_idx]   # (n_tri, 3, 2)
    src_tris = src_lm[tri_idx]   # (n_tri, 3, 2)

    # ------------------------------------------------------------------
    # 1. Triangle label map
    #    Each pixel gets the index of the triangle that contains it.
    #    Uses int32 to support up to ~2 billion triangles (overkill, safe).
    # ------------------------------------------------------------------
    label_map = np.full((h, w), -1, dtype=np.int32)
    for i in range(n_tri):
        cv2.fillConvexPoly(label_map, dst_tris[i].astype(np.int32), i)

    # ------------------------------------------------------------------
    # 2. Batch inverse affine matrices (dst → src) — no Python loop
    #
    #    For each triangle, affine M (2×3) satisfies:
    #        M @ [dst_x, dst_y, 1]^T = [src_x, src_y]^T
    #
    #    Stacking the 3 triangle vertices:
    #        P @ M^T = src_tris   where P = [dst_x, dst_y, 1] per row
    #    Solve all n_tri systems at once:
    #        M^T = linalg.solve(P, src_tris)
    # ------------------------------------------------------------------
    ones = np.ones((n_tri, 3, 1), dtype=np.float32)
    P    = np.concatenate([dst_tris, ones], axis=2)   # (n_tri, 3, 3)
    M_T  = np.linalg.solve(P, src_tris)               # (n_tri, 3, 2)
    affines = M_T.transpose(0, 2, 1)                  # (n_tri, 2, 3)

    # ------------------------------------------------------------------
    # 3. Pixel → source coordinate mapping
    #
    #    For every pixel inside a triangle:
    #        src_x = a*dst_x + b*dst_y + c
    #        src_y = d*dst_x + e*dst_y + f
    #
    #    Numba path: parallel native loop across rows (uses all CPU cores).
    #    NumPy path: vectorised gather + element-wise ops (fallback).
    # ------------------------------------------------------------------
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    if _NUMBA:
        _build_remap(label_map, affines, map_x, map_y)
    else:
        valid  = label_map >= 0
        ys, xs = np.where(valid)
        labels = label_map[ys, xs]
        xs_f   = xs.astype(np.float32)
        ys_f   = ys.astype(np.float32)
        a = affines[labels, 0, 0];  b = affines[labels, 0, 1];  c = affines[labels, 0, 2]
        d = affines[labels, 1, 0];  e = affines[labels, 1, 1];  f = affines[labels, 1, 2]
        map_x[ys, xs] = a * xs_f + b * ys_f + c
        map_y[ys, xs] = d * xs_f + e * ys_f + f

    # Single remap — one C++ call replaces N warpAffine calls
    warped = cv2.remap(src_frame, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    warped[label_map < 0] = 0

    # Convex hull mask
    hull = cv2.convexHull(dst_lm)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    return warped, mask


# ---------------------------------------------------------------------------
# Standalone test — draws Delaunay triangulation on live webcam feed
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import mediapipe as mp
    from core.face_model import create_image_landmarker, landmarks_to_numpy

    print("Warp engine test — shows Delaunay triangulation on webcam")
    print("Press Q to quit")

    cap = cv2.VideoCapture(0)

    with create_image_landmarker() as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)
            vis      = frame.copy()

            if result.face_landmarks:
                landmarks = landmarks_to_numpy(result.face_landmarks[0], w, h)
                indices   = get_triangle_indices(landmarks, (h, w))
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
