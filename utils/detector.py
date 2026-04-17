"""
utils/detector.py
-----------------
Step 1 of the pipeline: locate the cattle face / head region.

Strategy
--------
1. Try Haar-cascade detection (primary path).
   The default cascade is trained on **human** frontal faces and is used as a
   placeholder.  Swap it for a cattle-specific XML or a YOLO model for
   production use — see README "Replacing the Haar Cascade".
2. If no detection or cascade unavailable → fall back to a centre crop of the
   full image (keeps the pipeline runnable without a trained detector).

Public API
----------
load_cascade(cascade_path=None) -> cv2.CascadeClassifier | None
detect_face(img_bgr, cascade=None, min_size=(60,60), scale_factor=1.1,
            min_neighbors=4) -> np.ndarray (BGR crop)
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

# Fraction of image used for the centre-crop fallback
_FALLBACK_FRACTION = 0.70


# ── Cascade loader ────────────────────────────────────────────────────────────

def load_cascade(cascade_path: str | None = None) -> "cv2.CascadeClassifier | None":
    """
    Load a Haar-cascade XML file.

    Parameters
    ----------
    cascade_path : path to an XML cascade file.
                   If None, uses OpenCV's bundled frontal-face cascade.

    Returns
    -------
    cv2.CascadeClassifier on success, None if the file cannot be loaded.
    """
    if cascade_path is None:
        # Use the frontal-face placeholder bundled with OpenCV
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    return cascade


# ── Fallback crop ─────────────────────────────────────────────────────────────

def _centre_crop(img: np.ndarray, fraction: float = _FALLBACK_FRACTION) -> np.ndarray:
    """Return the central `fraction` × `fraction` crop of *img*."""
    h, w = img.shape[:2]
    dy = int(h * (1 - fraction) / 2)
    dx = int(w * (1 - fraction) / 2)
    dy = max(dy, 0)
    dx = max(dx, 0)
    return img[dy : h - dy, dx : w - dx]


# ── Main function ─────────────────────────────────────────────────────────────

def detect_face(
    img_bgr: np.ndarray,
    cascade: "cv2.CascadeClassifier | None" = None,
    min_size: tuple[int, int] = (60, 60),
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
) -> np.ndarray:
    """
    Detect the face/head ROI in a cattle image.

    Parameters
    ----------
    img_bgr       : input image in BGR format (H×W×3 uint8).
    cascade       : pre-loaded CascadeClassifier; if None, the function
                    attempts to load the default frontal-face cascade.
    min_size      : minimum detection box (width, height) in pixels.
    scale_factor  : image pyramid scale factor for the Haar detector.
    min_neighbors : minimum neighbour rectangles required to keep a detection.

    Returns
    -------
    np.ndarray  BGR crop containing the face/head ROI.
                Falls back to a centre crop when detection fails.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("detect_face: received an empty image.")

    # Try Haar detection
    if cascade is None:
        cascade = load_cascade()

    if cascade is not None and not cascade.empty():
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        if len(detections) > 0:
            # Use the largest detected region
            x, y, w, h = max(detections, key=lambda r: r[2] * r[3])
            return img_bgr[y : y + h, x : x + w]

    # Fallback: centre crop
    return _centre_crop(img_bgr)
