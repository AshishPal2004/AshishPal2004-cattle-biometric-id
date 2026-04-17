"""
pipeline.py
-----------
Orchestrates Steps 1-5 for a single cattle image:

    Input image
        │
        ▼  Step 1 – detector.py          (face detection + centre-crop fallback)
        ▼  Step 2 – muzzle_extractor.py  (bottom-centre or watershed crop)
        ▼  Step 3 – enhancer.py          (resize → grayscale → CLAHE → histeq)
        ▼  Steps 4/5 – feature_extractor.py  (HOG + LBP → float32 vector)
        │
        └─→ 1-D float32 feature vector

The pipeline stops before classification so that the same `process_image()`
call can be used during both training (batch) and inference (single image).

Public API
----------
process_image(image_path_or_array,
              cascade_path=None,
              use_watershed=False,
              debug=False,
              canonical_size=None) -> np.ndarray | tuple

    Returns the feature vector (np.ndarray, float32).
    When debug=True returns (feature_vector, debug_dict) where debug_dict
    contains the intermediate BGR/grayscale arrays at each stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from utils.detector          import detect_face, load_cascade
from utils.muzzle_extractor  import extract_muzzle
from utils.enhancer          import enhance, CANONICAL_SIZE
from utils.feature_extractor import extract_features


# ── Type alias ────────────────────────────────────────────────────────────────

ImageSource = Union[str, Path, np.ndarray]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_image(source: ImageSource) -> np.ndarray:
    """Load an image from a file path or return the array as-is."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"process_image: cannot load '{source}'.")
        return img
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("process_image: received an empty numpy array.")
        return source
    raise TypeError(f"process_image: unsupported image type {type(source)}.")


# ── Public API ────────────────────────────────────────────────────────────────

def process_image(
    image: ImageSource,
    cascade_path: str | None = None,
    use_watershed: bool = False,
    debug: bool = False,
    canonical_size: int | None = None,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Run the full preprocessing pipeline for one cattle image.

    Parameters
    ----------
    image          : file path (str/Path) or BGR numpy array.
    cascade_path   : path to a Haar-cascade XML file.
                     None → use OpenCV's built-in frontal-face cascade.
    use_watershed  : enable watershed-based muzzle segmentation (Step 2).
    debug          : if True, return intermediate crops as well.
    canonical_size : override CANONICAL_SIZE (128) for the enhancer.

    Returns
    -------
    feature_vector : np.ndarray float32, shape (D,)
    debug_dict     : (only when debug=True) dict with keys:
                       'original'   – raw BGR image
                       'face_roi'   – face/head BGR crop
                       'muzzle_roi' – muzzle BGR crop
                       'enhanced'   – 128×128 uint8 grayscale
    """
    # --- Load ---
    img = _load_image(image)

    # --- Step 1: Face detection ---
    cascade = load_cascade(cascade_path)
    face_roi = detect_face(img, cascade=cascade)

    # --- Step 2: Muzzle extraction ---
    muzzle_roi = extract_muzzle(face_roi, use_watershed=use_watershed)

    # --- Step 3: Enhancement ---
    enhanced = enhance(muzzle_roi, canonical_size=canonical_size)

    # --- Steps 4/5: Feature extraction ---
    features = extract_features(enhanced)

    if debug:
        return features, {
            "original":   img,
            "face_roi":   face_roi,
            "muzzle_roi": muzzle_roi,
            "enhanced":   enhanced,
        }

    return features
