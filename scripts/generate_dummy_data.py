"""
scripts/generate_dummy_data.py
------------------------------
Create synthetic cattle muzzle images for testing the pipeline without a
real dataset.

Each "cattle" gets a unique random background colour plus randomly placed
ellipses and circles to simulate muzzle texture patterns.

Usage
-----
    python scripts/generate_dummy_data.py --cattle 5 --images 8
    python scripts/generate_dummy_data.py --cattle 5 --images 8 --test 3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def _make_muzzle_image(
    rng: np.random.Generator,
    base_color: tuple[int, int, int],
    size: int = 256,
) -> np.ndarray:
    """
    Generate a synthetic 'muzzle' image for a single cattle.

    Parameters
    ----------
    rng        : seeded random generator (ensures reproducibility per cattle).
    base_color : BGR background colour that is unique per cattle.
    size       : output image size in pixels (square).

    Returns
    -------
    BGR uint8 ndarray of shape (size, size, 3).
    """
    img = np.full((size, size, 3), base_color, dtype=np.uint8)

    # Add noise layer
    noise = rng.integers(0, 30, size=(size, size, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Draw random ellipses (nostrils / pattern)
    n_ellipses = rng.integers(3, 8)
    for _ in range(n_ellipses):
        cx = int(rng.integers(size // 4, 3 * size // 4))
        cy = int(rng.integers(size // 3, 2 * size // 3))
        axes = (int(rng.integers(10, 35)), int(rng.integers(8, 25)))
        angle = int(rng.integers(0, 180))
        color = (
            int(rng.integers(20, 80)),
            int(rng.integers(20, 80)),
            int(rng.integers(20, 80)),
        )
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, thickness=-1)

    # Draw small circles for texture
    n_circles = int(rng.integers(5, 15))
    for _ in range(n_circles):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))
        r  = int(rng.integers(2, 8))
        color = tuple(int(v) for v in rng.integers(60, 200, size=3))
        cv2.circle(img, (cx, cy), r, color, thickness=-1)

    # Mild Gaussian blur to smooth synthetic texture
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return img


def generate_dataset(
    output_dir: str | Path,
    n_cattle: int,
    n_images: int,
    seed: int = 42,
) -> None:
    """
    Generate and save synthetic muzzle images.

    Parameters
    ----------
    output_dir : destination root directory.
    n_cattle   : number of unique cattle IDs to create.
    n_images   : number of images per cattle.
    seed       : master random seed.
    """
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    for i in range(1, n_cattle + 1):
        cattle_id = f"cattle_{i:03d}"
        cattle_dir = output_dir / cattle_id
        cattle_dir.mkdir(parents=True, exist_ok=True)

        # Fixed base colour per cattle for identity consistency
        base_color = tuple(int(v) for v in rng.integers(80, 200, size=3))
        cattle_rng = np.random.default_rng(seed + i)

        for j in range(1, n_images + 1):
            img = _make_muzzle_image(cattle_rng, base_color)
            # Small per-image variation in brightness
            delta = int(rng.integers(-15, 15))
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            path = cattle_dir / f"photo_{j:03d}.jpg"
            cv2.imwrite(str(path), img)

        print(f"  [{cattle_id}] {n_images} images → {cattle_dir}")

    print(f"\n[generate_dummy_data] Done.  Dataset written to '{output_dir}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic cattle muzzle images for pipeline testing."
    )
    parser.add_argument("--cattle", type=int, default=5,
                        help="Number of cattle IDs to create (default: 5)")
    parser.add_argument("--images", type=int, default=8,
                        help="Images per cattle ID (default: 8)")
    parser.add_argument("--test",   type=int, default=0,
                        help="If >0, also create data/test/ with this many images "
                             "per cattle (default: 0 = skip)")
    parser.add_argument("--out", default="data/raw",
                        help="Output directory for training data (default: data/raw)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"[generate_dummy_data] Generating training data → '{args.out}'")
    generate_dataset(args.out, args.cattle, args.images, seed=args.seed)

    if args.test > 0:
        test_dir = str(Path(args.out).parent / "test")
        print(f"[generate_dummy_data] Generating test data    → '{test_dir}'")
        generate_dataset(test_dir, args.cattle, args.test, seed=args.seed + 1000)


if __name__ == "__main__":
    main()
