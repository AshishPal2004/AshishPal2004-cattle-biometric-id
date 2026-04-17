"""
scripts/train_pipeline.py
-------------------------
Build the cattle identification model from a labelled image dataset.

Expected dataset layout:
    data/raw/
    ├── cattle_001/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    └── cattle_002/
        └── photo1.jpg

Each sub-folder name is treated as the cattle ID.

Usage
-----
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --data data/raw --no-pca --no-cv
    python scripts/train_pipeline.py --watershed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline          import process_image
from utils.classifier  import train


# ── Supported image extensions ────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    data_dir: str | Path,
    cascade_path: str | None,
    use_watershed: bool,
) -> tuple[np.ndarray, list[str]]:
    """
    Walk *data_dir*, process every image, and return (X, y).

    Returns
    -------
    X : float32 array of shape (n_samples, n_features)
    y : list of cattle ID strings, length n_samples
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        sys.exit(f"[train] ERROR: dataset directory '{data_dir}' does not exist.\n"
                 "       Run: python scripts/generate_dummy_data.py  to create test data.")

    X_list, y_list = [], []
    errors = 0

    cattle_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not cattle_dirs:
        sys.exit(f"[train] ERROR: no sub-directories found in '{data_dir}'.")

    print(f"[train] Dataset: '{data_dir}'  —  {len(cattle_dirs)} cattle IDs found")

    for cattle_dir in cattle_dirs:
        cattle_id  = cattle_dir.name
        img_paths  = [p for p in cattle_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]

        if not img_paths:
            print(f"  [WARN] No images in '{cattle_dir}' — skipping.")
            continue

        for img_path in sorted(img_paths):
            try:
                feat = process_image(
                    img_path,
                    cascade_path=cascade_path,
                    use_watershed=use_watershed,
                )
                X_list.append(feat)
                y_list.append(cattle_id)
            except Exception as exc:
                print(f"  [WARN] {img_path.name}: {exc}")
                errors += 1

        print(f"  [{cattle_id}] {len(img_paths)} images  ({errors} errors so far)")

    if not X_list:
        sys.exit("[train] ERROR: no features were extracted.  Check your dataset.")

    X = np.stack(X_list).astype(np.float32)
    print(f"\n[train] Feature matrix: {X.shape}  |  classes: {len(set(y_list))}")
    if errors:
        print(f"[train] WARNING: {errors} image(s) failed to process.")

    return X, y_list


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the HOG+LBP+SVM cattle identification pipeline."
    )
    parser.add_argument("--data",    default="data/raw",
                        help="Dataset root directory (default: data/raw)")
    parser.add_argument("--models",  default="models",
                        help="Directory to save model artefacts (default: models)")
    parser.add_argument("--cascade", default=None,
                        help="Path to Haar-cascade XML (default: OpenCV built-in)")
    parser.add_argument("--no-pca",  action="store_true",
                        help="Skip PCA dimensionality reduction")
    parser.add_argument("--no-cv",   action="store_true",
                        help="Skip 5-fold cross-validation")
    parser.add_argument("--watershed", action="store_true",
                        help="Use watershed segmentation for muzzle extraction")
    parser.add_argument("--pca-components", type=int, default=100,
                        help="Number of PCA components (default: 100)")
    parser.add_argument("--svm-c",   type=float, default=1.0,
                        help="SVM regularisation C (default: 1.0)")
    args = parser.parse_args()

    X, y = build_dataset(
        data_dir=args.data,
        cascade_path=args.cascade,
        use_watershed=args.watershed,
    )

    print("\n[train] Fitting classifier …")
    train(
        X=X,
        y=y,
        use_pca=not args.no_pca,
        n_components=args.pca_components,
        svm_C=args.svm_c,
        run_cv=not args.no_cv,
        model_dir=args.models,
    )
    print("[train] Training complete.")


if __name__ == "__main__":
    main()
