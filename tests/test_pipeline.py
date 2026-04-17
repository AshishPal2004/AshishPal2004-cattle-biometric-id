"""
tests/test_pipeline.py
----------------------
Unit and integration tests for the HOG+LBP+SVM MVP pipeline.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── Allow import from repo root ───────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.detector          import detect_face, load_cascade, _centre_crop
from utils.muzzle_extractor  import extract_muzzle
from utils.enhancer          import enhance, CANONICAL_SIZE
from utils.feature_extractor import extract_features, feature_dim
from utils.classifier        import train, predict, load_model
from pipeline                import process_image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_bgr():
    """256×256 coloured BGR image (simulated cattle photo)."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = (80, 120, 60)   # greenish background
    # "head" ellipse
    cv2.ellipse(img, (128, 100), (90, 70), 0, 0, 360, (60, 90, 40), -1)
    # "muzzle" ellipse in lower half
    cv2.ellipse(img, (128, 180), (50, 35), 0, 0, 360, (40, 60, 30), -1)
    return img


@pytest.fixture
def dummy_gray(dummy_bgr):
    return cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2GRAY)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Detector
# ══════════════════════════════════════════════════════════════════════════════

class TestDetector:
    def test_returns_non_empty(self, dummy_bgr):
        roi = detect_face(dummy_bgr)
        assert roi is not None
        assert roi.size > 0

    def test_output_is_bgr(self, dummy_bgr):
        roi = detect_face(dummy_bgr)
        assert roi.ndim == 3
        assert roi.shape[2] == 3

    def test_fallback_centre_crop(self, dummy_bgr):
        # Pass a cascade=None to trigger fallback path
        roi = detect_face(dummy_bgr, cascade=None)
        h, w = dummy_bgr.shape[:2]
        assert roi.shape[0] <= h
        assert roi.shape[1] <= w

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            detect_face(np.array([]))

    def test_centre_crop_shape(self, dummy_bgr):
        crop = _centre_crop(dummy_bgr, fraction=0.5)
        h, w = dummy_bgr.shape[:2]
        assert crop.shape[0] <= h
        assert crop.shape[1] <= w
        assert crop.size > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Muzzle Extractor
# ══════════════════════════════════════════════════════════════════════════════

class TestMuzzleExtractor:
    def test_heuristic_non_empty(self, dummy_bgr):
        muzzle = extract_muzzle(dummy_bgr, use_watershed=False)
        assert muzzle is not None
        assert muzzle.size > 0

    def test_heuristic_is_bgr(self, dummy_bgr):
        muzzle = extract_muzzle(dummy_bgr, use_watershed=False)
        assert muzzle.ndim == 3

    def test_watershed_non_empty(self, dummy_bgr):
        muzzle = extract_muzzle(dummy_bgr, use_watershed=True)
        assert muzzle is not None
        assert muzzle.size > 0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            extract_muzzle(np.array([]))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Enhancer
# ══════════════════════════════════════════════════════════════════════════════

class TestEnhancer:
    def test_output_shape(self, dummy_bgr):
        result = enhance(dummy_bgr)
        assert result.shape == (CANONICAL_SIZE, CANONICAL_SIZE)

    def test_output_is_uint8(self, dummy_bgr):
        result = enhance(dummy_bgr)
        assert result.dtype == np.uint8

    def test_custom_size(self, dummy_bgr):
        result = enhance(dummy_bgr, canonical_size=64)
        assert result.shape == (64, 64)

    def test_grayscale_input(self, dummy_gray):
        result = enhance(dummy_gray)
        assert result.shape == (CANONICAL_SIZE, CANONICAL_SIZE)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            enhance(np.array([]))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureExtractor:
    def test_output_is_1d_float32(self):
        gray = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8)
        feat = extract_features(gray)
        assert feat.ndim == 1
        assert feat.dtype == np.float32

    def test_feature_dim_consistent(self):
        gray = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8)
        feat = extract_features(gray)
        assert len(feat) == feature_dim()

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            extract_features(np.zeros((64, 64), dtype=np.uint8))

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            extract_features(np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8))

    def test_different_images_give_different_vectors(self, dummy_bgr):
        g1 = enhance(dummy_bgr)
        g2 = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8)
        f1 = extract_features(g1)
        f2 = extract_features(g2)
        assert not np.allclose(f1, f2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline (end-to-end preprocessing)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipeline:
    def test_returns_1d_float32(self, dummy_bgr, tmp_path):
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), dummy_bgr)
        feat = process_image(img_path)
        assert feat.ndim == 1
        assert feat.dtype == np.float32

    def test_accepts_ndarray(self, dummy_bgr):
        feat = process_image(dummy_bgr)
        assert feat.ndim == 1

    def test_debug_mode(self, dummy_bgr):
        feat, dbg = process_image(dummy_bgr, debug=True)
        assert feat.ndim == 1
        for key in ("original", "face_roi", "muzzle_roi", "enhanced"):
            assert key in dbg
            assert dbg[key].size > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            process_image("/nonexistent/path/image.jpg")

    def test_watershed_path(self, dummy_bgr):
        feat = process_image(dummy_bgr, use_watershed=True)
        assert feat.ndim == 1

    def test_feature_dim_stable(self, dummy_bgr):
        f1 = process_image(dummy_bgr)
        f2 = process_image(dummy_bgr)
        assert f1.shape == f2.shape
        assert np.allclose(f1, f2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Classifier (train / predict / load)
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifier:
    @pytest.fixture
    def small_dataset(self):
        """6 samples, 2 classes, each with a feature vector of size feature_dim()."""
        dim = feature_dim()
        rng = np.random.default_rng(0)
        X = rng.random((6, dim)).astype(np.float32)
        y = ["cattle_001"] * 3 + ["cattle_002"] * 3
        return X, y

    def test_train_saves_artefacts(self, small_dataset, tmp_path):
        X, y = small_dataset
        train(X, y, use_pca=True, n_components=10, run_cv=False,
              model_dir=str(tmp_path))
        assert (tmp_path / "cattle_svm.joblib").exists()
        assert (tmp_path / "label_encoder.joblib").exists()

    def test_predict_returns_correct_format(self, small_dataset, tmp_path):
        X, y = small_dataset
        train(X, y, use_pca=True, n_components=10, run_cv=False,
              model_dir=str(tmp_path))
        results = predict(X[0], model_dir=str(tmp_path))
        assert len(results) == 1
        cattle_id, conf = results[0]
        assert isinstance(cattle_id, str)
        assert 0.0 <= conf <= 1.0

    def test_predict_batch(self, small_dataset, tmp_path):
        X, y = small_dataset
        train(X, y, use_pca=True, n_components=10, run_cv=False,
              model_dir=str(tmp_path))
        results = predict(X, model_dir=str(tmp_path))
        assert len(results) == len(X)

    def test_load_model_missing_raises(self, tmp_path):
        from utils.classifier import load_model
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path))

    def test_train_single_class_raises(self, tmp_path):
        dim = feature_dim()
        X = np.zeros((4, dim), dtype=np.float32)
        y = ["cattle_001"] * 4
        with pytest.raises(ValueError):
            train(X, y, run_cv=False, model_dir=str(tmp_path))

    def test_no_pca_option(self, small_dataset, tmp_path):
        X, y = small_dataset
        train(X, y, use_pca=False, run_cv=False, model_dir=str(tmp_path))
        results = predict(X[:2], model_dir=str(tmp_path))
        assert len(results) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 7. Full smoke test: generate → train → predict
# ══════════════════════════════════════════════════════════════════════════════

class TestSmokeTest:
    def test_full_pipeline(self, tmp_path):
        """End-to-end: create synthetic data, train, predict."""
        # Generate tiny dataset
        from scripts.generate_dummy_data import generate_dataset
        train_dir = tmp_path / "train"
        generate_dataset(train_dir, n_cattle=3, n_images=5, seed=99)

        # Build feature matrix
        from pipeline import process_image
        X_list, y_list = [], []
        for cattle_dir in sorted(train_dir.iterdir()):
            for img_path in sorted(cattle_dir.glob("*.jpg")):
                feat = process_image(img_path)
                X_list.append(feat)
                y_list.append(cattle_dir.name)

        X = np.stack(X_list).astype(np.float32)

        # Train
        model_dir = str(tmp_path / "models")
        train(X, y_list, use_pca=True, n_components=20, run_cv=False,
              model_dir=model_dir)

        # Predict a sample image from each class
        for cattle_dir in sorted(train_dir.iterdir()):
            img_path = next(cattle_dir.glob("*.jpg"))
            feat     = process_image(img_path)
            results  = predict(feat, model_dir=model_dir)
            assert len(results) == 1
            cattle_id, conf = results[0]
            assert conf >= 0.0
