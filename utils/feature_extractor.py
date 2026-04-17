"""
utils/feature_extractor.py
--------------------------
Steps 4 & 5 of the pipeline: compute a HOG descriptor and a uniform-LBP
histogram from a 128×128 grayscale muzzle image, then concatenate them into
a single float32 vector.

HOG  — Histogram of Oriented Gradients (skimage)
LBP  — Local Binary Pattern (uniform, 8 neighbours, radius 1)  (skimage)

Default dimensions (CANONICAL_SIZE = 128):
  HOG  : depends on cell/block config  ≈ 1764 dims for (8×8) cells / (2×2) blocks
  LBP  : 10 uniform patterns + 1 non-uniform bin = 59 bins (R=1,P=8,method='uniform')
  Total: HOG_DIM + 59

Tuning knobs
------------
HOG_PIXELS_PER_CELL  (config default: (8, 8))  — smaller → richer but larger vector
HOG_CELLS_PER_BLOCK  (config default: (2, 2))
HOG_ORIENTATIONS     (config default: 9)
LBP_RADIUS           (config default: 1)
LBP_N_POINTS         (config default: 8)

Public API
----------
extract_features(gray_128x128) -> np.ndarray  float32, 1-D
feature_dim(canonical_size)    -> int          (offline dimension query)
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog, local_binary_pattern

from .enhancer import CANONICAL_SIZE


# ── HOG config ────────────────────────────────────────────────────────────────

HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ── LBP config ────────────────────────────────────────────────────────────────

LBP_RADIUS   = 1
LBP_N_POINTS = 8 * LBP_RADIUS   # 8 neighbours
LBP_METHOD   = "uniform"

# uniform LBP: P*(P-1)+3 bins when method='uniform'
_LBP_N_BINS  = LBP_N_POINTS * (LBP_N_POINTS - 1) + 3  # = 59 for P=8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hog_features(gray: np.ndarray) -> np.ndarray:
    """Return a 1-D float32 HOG descriptor."""
    fd = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return fd.astype(np.float32)


def _lbp_features(gray: np.ndarray) -> np.ndarray:
    """Return a normalised uniform-LBP histogram as float32."""
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    # Histogram over integer codes 0 .. P*(P-1)+2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=_LBP_N_BINS,
        range=(0, _LBP_N_BINS),
    )
    # Normalise to sum=1 (probability distribution)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(gray: np.ndarray) -> np.ndarray:
    """
    Compute the combined HOG + LBP feature vector.

    Parameters
    ----------
    gray : 2-D uint8 grayscale array of shape (CANONICAL_SIZE, CANONICAL_SIZE).
           Must be the output of `enhancer.enhance()`.

    Returns
    -------
    np.ndarray  1-D float32 feature vector  (HOG_DIM + LBP_N_BINS,).
    """
    if gray is None or gray.ndim != 2:
        raise ValueError(
            "extract_features: expected a 2-D grayscale array; "
            f"got shape {getattr(gray, 'shape', None)}."
        )
    if gray.shape != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(
            f"extract_features: expected ({CANONICAL_SIZE}, {CANONICAL_SIZE}); "
            f"got {gray.shape}.  Run enhancer.enhance() first."
        )

    hog_vec = _hog_features(gray)
    lbp_vec = _lbp_features(gray)
    return np.concatenate([hog_vec, lbp_vec])


def feature_dim(canonical_size: int = CANONICAL_SIZE) -> int:
    """
    Return the feature vector length without processing an actual image.
    Useful for sanity checks and pre-allocating arrays.
    """
    dummy = np.zeros((canonical_size, canonical_size), dtype=np.uint8)
    return len(extract_features(dummy))
