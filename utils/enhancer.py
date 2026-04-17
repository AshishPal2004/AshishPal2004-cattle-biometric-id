"""
utils/enhancer.py
-----------------
Step 3 of the pipeline: normalise a raw muzzle crop into a consistent
128×128 grayscale image ready for feature extraction.

Processing order
----------------
1. Convert to grayscale
2. Resize to CANONICAL_SIZE × CANONICAL_SIZE (bicubic)
3. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
4. Global histogram equalisation

Public API
----------
enhance(img_bgr_or_gray, canonical_size=None, clip_limit=2.0,
        tile_grid=(8,8)) -> np.ndarray  (uint8 grayscale, CANONICAL_SIZE²)
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Defaults (can be overridden per-call) ─────────────────────────────────────

CANONICAL_SIZE = 128          # both width and height
_DEFAULT_CLIP  = 2.0          # CLAHE clip limit
_DEFAULT_TILE  = (8, 8)       # CLAHE tile grid


# ── Public API ────────────────────────────────────────────────────────────────

def enhance(
    img: np.ndarray,
    canonical_size: int | None = None,
    clip_limit: float = _DEFAULT_CLIP,
    tile_grid: tuple[int, int] = _DEFAULT_TILE,
) -> np.ndarray:
    """
    Preprocess a muzzle crop for feature extraction.

    Parameters
    ----------
    img            : input image in BGR (3-channel) or grayscale (1-channel/2D).
    canonical_size : target square side in pixels; defaults to CANONICAL_SIZE.
    clip_limit     : CLAHE clip limit (1.5–3.0 is a useful range).
    tile_grid      : CLAHE tile grid size (rows, cols).

    Returns
    -------
    np.ndarray  2-D uint8 grayscale array of shape
                (canonical_size, canonical_size).
    """
    if img is None or img.size == 0:
        raise ValueError("enhance: received an empty image.")

    size = canonical_size if canonical_size is not None else CANONICAL_SIZE

    # 1. Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Resize
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    gray  = clahe.apply(gray)

    # 4. Global histogram equalisation
    gray = cv2.equalizeHist(gray)

    return gray
