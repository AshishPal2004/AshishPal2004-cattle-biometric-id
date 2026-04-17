"""
utils/muzzle_extractor.py
-------------------------
Step 2 of the pipeline: extract the muzzle (nose/nostrils) region from a
previously detected face/head ROI.

Two strategies are available:

heuristic  (default)
    Take the bottom-centre quarter of the face ROI.  Fast and cascade-agnostic.

watershed  (optional, enable with `use_watershed=True`)
    Applies GrabCut + watershed to try to isolate the dark muzzle disk.
    Works best when lighting is consistent; falls back to heuristic on failure.

Public API
----------
extract_muzzle(face_roi_bgr, use_watershed=False) -> np.ndarray (BGR crop)
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

# Heuristic crop: bottom fraction of the face, centred horizontally
_ROI_Y_START  = 0.50   # start at 50 % down the face
_ROI_X_MARGIN = 0.15   # strip 15 % from each side


# ── Heuristic extractor ───────────────────────────────────────────────────────

def _heuristic_muzzle(face: np.ndarray) -> np.ndarray:
    """Bottom-centre crop of the face ROI."""
    h, w = face.shape[:2]
    y0 = int(h * _ROI_Y_START)
    x0 = int(w * _ROI_X_MARGIN)
    x1 = w - x0
    crop = face[y0:, x0:x1]
    if crop.size == 0:
        return face   # safety: return full face if arithmetic fails
    return crop


# ── Watershed extractor ───────────────────────────────────────────────────────

def _watershed_muzzle(face: np.ndarray) -> np.ndarray:
    """
    Attempt to segment the muzzle disk using GrabCut seeded in the
    bottom-centre of the face.  Falls back to the heuristic on any failure.
    """
    h, w = face.shape[:2]
    if h < 40 or w < 40:
        return _heuristic_muzzle(face)

    try:
        # Seed rectangle covering the expected muzzle area
        rect_x = int(w * 0.20)
        rect_y = int(h * 0.45)
        rect_w = w - 2 * rect_x
        rect_h = h - rect_y
        if rect_w < 10 or rect_h < 10:
            return _heuristic_muzzle(face)

        mask   = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(
            face, mask,
            (rect_x, rect_y, rect_w, rect_h),
            bgd_model, fgd_model,
            iterCount=3,
            mode=cv2.GC_INIT_WITH_RECT,
        )

        # Probable and definite foreground
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Morphological clean-up
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _heuristic_muzzle(face)

        # Pick the largest contour that sits in the lower half
        lower_half = [(c, cv2.boundingRect(c)) for c in contours
                      if cv2.boundingRect(c)[1] > h * 0.35]
        if not lower_half:
            lower_half = [(c, cv2.boundingRect(c)) for c in contours]

        best_c, (bx, by, bw, bh) = max(lower_half, key=lambda item: item[1][2] * item[1][3])

        if bw < 10 or bh < 10:
            return _heuristic_muzzle(face)

        return face[by : by + bh, bx : bx + bw]

    except cv2.error:
        return _heuristic_muzzle(face)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_muzzle(
    face_roi_bgr: np.ndarray,
    use_watershed: bool = False,
) -> np.ndarray:
    """
    Extract the muzzle ROI from a cattle face crop.

    Parameters
    ----------
    face_roi_bgr  : BGR face / head crop (output of detector.detect_face).
    use_watershed : if True, attempt watershed segmentation before falling
                    back to the heuristic bottom-centre crop.

    Returns
    -------
    np.ndarray  BGR crop of the muzzle region.
    """
    if face_roi_bgr is None or face_roi_bgr.size == 0:
        raise ValueError("extract_muzzle: received an empty face ROI.")

    if use_watershed:
        return _watershed_muzzle(face_roi_bgr)
    return _heuristic_muzzle(face_roi_bgr)
