"""
scripts/evaluate.py
-------------------
Evaluate the trained model on a labelled test set and report accuracy metrics.

Expected layout (same as the training dataset):
    data/test/
    ├── cattle_001/
    │   └── photo1.jpg
    └── cattle_002/
        └── photo1.jpg

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --data data/test --models models
    python scripts/evaluate.py --data data/test --watershed --no-cm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline          import process_image
from utils.classifier  import predict, load_model


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _collect_predictions(
    data_dir: Path,
    cascade_path: str | None,
    use_watershed: bool,
    model_dir: str,
) -> tuple[list[str], list[str], list[float]]:
    """Return (true_labels, pred_labels, confidences)."""
    true_labels, pred_labels, confidences = [], [], []
    errors = 0

    cattle_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not cattle_dirs:
        sys.exit(f"[evaluate] ERROR: no sub-directories in '{data_dir}'.")

    for cattle_dir in cattle_dirs:
        cattle_id = cattle_dir.name
        img_paths = [p for p in cattle_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]

        for img_path in sorted(img_paths):
            try:
                feat = process_image(img_path, cascade_path=cascade_path,
                                     use_watershed=use_watershed)
                results = predict(feat, model_dir=model_dir)
                pred_id, conf = results[0]
                true_labels.append(cattle_id)
                pred_labels.append(pred_id)
                confidences.append(conf)
            except Exception as exc:
                print(f"  [WARN] {img_path.name}: {exc}")
                errors += 1

    if errors:
        print(f"[evaluate] WARNING: {errors} image(s) failed to process.")

    return true_labels, pred_labels, confidences


def _print_results(
    true_labels: list[str],
    pred_labels: list[str],
    confidences: list[float],
    show_cm: bool,
) -> None:
    """Print accuracy, per-class breakdown, and optional confusion matrix."""
    n_total  = len(true_labels)
    n_correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy  = n_correct / n_total if n_total else 0.0

    print(f"\n[Results]")
    print(f"  Total images : {n_total}")
    print(f"  Correct      : {n_correct}")
    print(f"  Accuracy     : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    print(f"  Avg confidence (correct):   "
          f"{np.mean([c for t, p, c in zip(true_labels, pred_labels, confidences) if t == p] or [0]):.3f}")
    print(f"  Avg confidence (incorrect): "
          f"{np.mean([c for t, p, c in zip(true_labels, pred_labels, confidences) if t != p] or [0]):.3f}")

    # Per-class breakdown
    classes = sorted(set(true_labels))
    print(f"\n  {'Cattle ID':<20}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print("  " + "-" * 46)
    for cls in classes:
        idxs    = [i for i, t in enumerate(true_labels) if t == cls]
        cls_tot = len(idxs)
        cls_ok  = sum(pred_labels[i] == cls for i in idxs)
        cls_acc = cls_ok / cls_tot if cls_tot else 0.0
        print(f"  {cls:<20}  {cls_ok:>7}  {cls_tot:>5}  {cls_acc:>6.3f}")

    if not show_cm:
        return

    # Simple text confusion matrix
    print(f"\n  Confusion matrix  (rows=true, cols=predicted):")
    header = "  " + " " * 20 + "  " + "  ".join(f"{c[:8]:>8}" for c in classes)
    print(header)
    for true_cls in classes:
        row = []
        for pred_cls in classes:
            count = sum(
                1 for t, p in zip(true_labels, pred_labels)
                if t == true_cls and p == pred_cls
            )
            row.append(f"{count:>8}")
        print(f"  {true_cls:<20}  " + "  ".join(row))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the cattle identification model on a test set."
    )
    parser.add_argument("--data",    default="data/test",
                        help="Test dataset root (default: data/test)")
    parser.add_argument("--models",  default="models",
                        help="Directory with model artefacts (default: models)")
    parser.add_argument("--cascade", default=None,
                        help="Path to Haar-cascade XML (default: OpenCV built-in)")
    parser.add_argument("--watershed", action="store_true",
                        help="Use watershed muzzle segmentation")
    parser.add_argument("--no-cm",   action="store_true",
                        help="Skip printing the confusion matrix")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        sys.exit(f"[evaluate] ERROR: test directory '{data_dir}' does not exist.")

    # Verify model exists before processing images
    try:
        load_model(args.models)
    except FileNotFoundError as exc:
        sys.exit(f"[evaluate] ERROR: {exc}")

    print(f"[evaluate] Test set: '{data_dir}'")
    true_labels, pred_labels, confidences = _collect_predictions(
        data_dir, args.cascade, args.watershed, args.models
    )

    if not true_labels:
        sys.exit("[evaluate] ERROR: no predictions were collected.")

    _print_results(true_labels, pred_labels, confidences, show_cm=not args.no_cm)


if __name__ == "__main__":
    main()
