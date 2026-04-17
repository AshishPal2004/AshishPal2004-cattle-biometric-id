"""
utils/classifier.py
-------------------
Steps 6 & 7 of the pipeline: train and run an SVM classifier on the
HOG+LBP feature vectors produced by feature_extractor.py.

Pipeline components
-------------------
    StandardScaler  →  optional PCA(n_components)  →  SVC (RBF kernel)

Persisted artefacts (saved with joblib)
---------------------------------------
    models/cattle_svm.joblib      — fitted sklearn Pipeline
    models/label_encoder.joblib   — fitted LabelEncoder (int ↔ cattle ID)

Public API
----------
train(X, y, use_pca=True, n_components=100, svm_C=1.0,
      model_dir="models") -> sklearn.pipeline.Pipeline
predict(X, model_dir="models") -> list[tuple[str, float]]
load_model(model_dir="models") -> tuple[Pipeline, LabelEncoder]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ── File names ────────────────────────────────────────────────────────────────

_SVM_FILE   = "cattle_svm.joblib"
_ENC_FILE   = "label_encoder.joblib"


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: Sequence[str],
    use_pca: bool = True,
    n_components: int = 100,
    svm_C: float = 1.0,
    svm_gamma: str = "scale",
    run_cv: bool = True,
    cv_folds: int = 5,
    model_dir: str = "models",
) -> Pipeline:
    """
    Fit a StandardScaler → (PCA) → SVM pipeline on feature matrix *X*.

    Parameters
    ----------
    X            : (n_samples, n_features) float32 feature matrix.
    y            : (n_samples,) cattle ID strings (class labels).
    use_pca      : include a PCA step.
    n_components : number of PCA components (capped to min(n_samples, n_features)).
    svm_C        : SVM regularisation parameter.
    svm_gamma    : SVM kernel coefficient ('scale', 'auto', or float).
    run_cv       : print stratified k-fold CV accuracy before fitting on full data.
    cv_folds     : number of CV folds (skipped if any class has fewer samples).
    model_dir    : directory where artefacts are saved.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    X = np.asarray(X, dtype=np.float32)
    y = list(y)

    if len(X) == 0:
        raise ValueError("train: feature matrix is empty.")
    if len(X) != len(y):
        raise ValueError(f"train: X has {len(X)} rows but y has {len(y)} labels.")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    if n_classes < 2:
        raise ValueError("train: need at least 2 cattle IDs to train a classifier.")

    # Cap PCA components
    max_components = min(len(X), X.shape[1])
    if use_pca:
        n_components = min(n_components, max_components)

    # Build pipeline
    steps: list[tuple] = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("svm", SVC(C=svm_C, kernel="rbf", gamma=svm_gamma, probability=True)))
    pipe = Pipeline(steps)

    # Optional cross-validation
    if run_cv:
        min_samples_per_class = min(np.bincount(y_enc))
        actual_folds = min(cv_folds, int(min_samples_per_class))
        if actual_folds >= 2:
            cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
            print(
                f"[CV] {actual_folds}-fold accuracy: "
                f"{scores.mean():.3f} ± {scores.std():.3f}  "
                f"(folds: {np.round(scores, 3).tolist()})"
            )
        else:
            print(f"[CV] Skipped: not enough samples per class for {cv_folds}-fold CV.")

    # Fit on full dataset
    pipe.fit(X, y_enc)

    # Save artefacts
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(model_dir) / _SVM_FILE)
    joblib.dump(le,   Path(model_dir) / _ENC_FILE)
    print(f"[Classifier] Saved → {Path(model_dir) / _SVM_FILE}")
    print(f"[Classifier] Saved → {Path(model_dir) / _ENC_FILE}")

    return pipe


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model(model_dir: str = "models") -> tuple[Pipeline, LabelEncoder]:
    """
    Load a previously trained pipeline and label encoder.

    Returns
    -------
    (Pipeline, LabelEncoder)

    Raises
    ------
    FileNotFoundError if either artefact is missing.
    """
    svm_path = Path(model_dir) / _SVM_FILE
    enc_path = Path(model_dir) / _ENC_FILE

    for p in (svm_path, enc_path):
        if not p.exists():
            raise FileNotFoundError(
                f"load_model: {p} not found.  "
                "Run scripts/train_pipeline.py first."
            )

    pipe = joblib.load(svm_path)
    le   = joblib.load(enc_path)
    return pipe, le


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    X: np.ndarray,
    model_dir: str = "models",
) -> list[tuple[str, float]]:
    """
    Predict cattle IDs for one or more feature vectors.

    Parameters
    ----------
    X         : (n_samples, n_features) or (n_features,) float32 array.
    model_dir : directory containing the saved artefacts.

    Returns
    -------
    list of (cattle_id: str, confidence: float) tuples, one per sample.
    """
    pipe, le = load_model(model_dir)

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[np.newaxis, :]   # single sample → (1, n_features)

    proba   = pipe.predict_proba(X)            # (n_samples, n_classes)
    cls_idx = proba.argmax(axis=1)
    confs   = proba.max(axis=1)
    labels  = le.inverse_transform(cls_idx)

    return list(zip(labels, confs.tolist()))
