"""
scripts/predict.py
------------------
Identify a single cattle image using the trained HOG+LBP+SVM model.

Usage
-----
    python scripts/predict.py path/to/cattle.jpg
    python scripts/predict.py path/to/cattle.jpg --debug
    python scripts/predict.py path/to/cattle.jpg --models models --cascade path/to/cascade.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline         import process_image
from utils.classifier import predict, load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify a cattle from a muzzle image."
    )
    parser.add_argument("image",
                        help="Path to the input image")
    parser.add_argument("--models",  default="models",
                        help="Directory containing model artefacts (default: models)")
    parser.add_argument("--cascade", default=None,
                        help="Path to Haar-cascade XML (default: OpenCV built-in)")
    parser.add_argument("--watershed", action="store_true",
                        help="Use watershed segmentation for muzzle extraction")
    parser.add_argument("--debug",   action="store_true",
                        help="Save intermediate crops to the current directory")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"[predict] ERROR: image '{img_path}' not found.")

    # Run preprocessing pipeline
    if args.debug:
        features, dbg = process_image(
            img_path,
            cascade_path=args.cascade,
            use_watershed=args.watershed,
            debug=True,
        )
        # Save intermediate images
        cv2.imwrite("debug_original.jpg",   dbg["original"])
        cv2.imwrite("debug_face_roi.jpg",   dbg["face_roi"])
        cv2.imwrite("debug_muzzle_roi.jpg", dbg["muzzle_roi"])
        cv2.imwrite("debug_enhanced.jpg",   dbg["enhanced"])
        print("[predict] Debug images saved:")
        print("  debug_original.jpg  → raw input")
        print("  debug_face_roi.jpg  → after Step 1 (face detection)")
        print("  debug_muzzle_roi.jpg → after Step 2 (muzzle extraction)")
        print("  debug_enhanced.jpg  → after Step 3 (enhancement)")
    else:
        features = process_image(
            img_path,
            cascade_path=args.cascade,
            use_watershed=args.watershed,
        )

    # Classify
    try:
        results = predict(features, model_dir=args.models)
    except FileNotFoundError as exc:
        sys.exit(f"[predict] ERROR: {exc}")

    cattle_id, confidence = results[0]

    print(f"\n[Result]")
    print(f"  Predicted cattle ID : {cattle_id}")
    print(f"  Confidence          : {confidence:.3f}  ({confidence * 100:.1f} %)")


if __name__ == "__main__":
    main()
