# Cattle Biometric Identification System
## Muzzle Pattern Recognition — MVP

---

## Folder Structure

```
cattle_id/
├── data/
│   ├── raw/                    # Training images  (cattle_001/, cattle_002/, …)
│   └── test/                   # Evaluation images (same sub-folder layout)
├── models/
│   ├── cattle_svm.joblib       # Trained SVM pipeline  (auto-generated)
│   └── label_encoder.joblib    # Label encoder         (auto-generated)
├── utils/
│   ├── __init__.py
│   ├── detector.py             # Step 1 – face detection + fallback crop
│   ├── muzzle_extractor.py     # Step 2 – muzzle ROI extraction
│   ├── enhancer.py             # Step 3 – CLAHE + histogram equalisation
│   ├── feature_extractor.py    # Steps 4/5 – HOG + LBP → feature vector
│   └── classifier.py           # Steps 6/7 – SVM train/predict
├── scripts/
│   ├── generate_dummy_data.py  # Create synthetic test images
│   ├── train_pipeline.py       # Train the model on data/raw/
│   ├── predict.py              # Identify a single cattle image
│   └── evaluate.py             # Evaluate model on data/test/
├── pipeline.py                 # Orchestrates Steps 1-5 for one image
└── requirements.txt
```

---

## Pipeline Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│  Step 1 · detector.py           │
│  Haar Cascade face detection    │
│  → fallback: centre crop        │
└────────────────┬────────────────┘
                 │  face ROI
                 ▼
┌─────────────────────────────────┐
│  Step 2 · muzzle_extractor.py   │
│  Heuristic: bottom-centre crop  │
│  Optional: watershed segment    │
└────────────────┬────────────────┘
                 │  muzzle ROI
                 ▼
┌─────────────────────────────────┐
│  Step 3 · enhancer.py           │
│  Resize → Grayscale             │
│  → CLAHE → Hist. Equalisation   │
└────────────────┬────────────────┘
                 │  128×128 gray
                 ▼
┌─────────────────────────────────┐
│  Steps 4/5 · feature_extractor  │
│  HOG descriptor                 │
│  LBP histogram (uniform)        │
│  concat → BoHoG+LBP vector      │
└────────────────┬────────────────┘
                 │  ~1600-dim float32
                 ▼
┌─────────────────────────────────┐
│  Steps 6/7 · classifier.py      │
│  StandardScaler → PCA(100)      │
│  → SVM (RBF kernel)             │
│  → Cattle ID + confidence       │
└─────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2a. Use your own images
Organise images under `data/raw/` like this:
```
data/raw/
├── cattle_001/
│   ├── photo1.jpg
│   └── photo2.jpg
└── cattle_002/
    └── photo1.jpg
```
Each folder name is the cattle's ID (any string).

### 2b. Or generate synthetic test data
```bash
python scripts/generate_dummy_data.py --cattle 5 --images 8
```

### 3. Train
```bash
python scripts/train_pipeline.py
```
Options:
| Flag | Effect |
|------|--------|
| `--data data/raw` | Dataset root (default: `data/raw`) |
| `--no-pca` | Skip PCA dimensionality reduction |
| `--no-cv` | Skip 5-fold cross-validation |
| `--watershed` | Use watershed muzzle segmentation |

### 4. Predict
```bash
python scripts/predict.py path/to/cattle.jpg
python scripts/predict.py path/to/cattle.jpg --debug   # shows intermediate crops
```

### 5. Evaluate on a test set
```bash
python scripts/evaluate.py --data data/test
```

---

## Module Reference

| Module | Role | Key functions |
|--------|------|---------------|
| `utils/detector.py` | Haar-cascade face detection, centre-crop fallback | `detect_face()` |
| `utils/muzzle_extractor.py` | Extract muzzle ROI (heuristic or watershed) | `extract_muzzle()` |
| `utils/enhancer.py` | Resize, CLAHE, histogram equalisation | `enhance()` |
| `utils/feature_extractor.py` | HOG + LBP → concatenated vector | `extract_features()` |
| `utils/classifier.py` | Train SVM pipeline; load & predict | `train()`, `predict()` |
| `pipeline.py` | End-to-end preprocessing for one image | `process_image()` |

---

## Replacing the Haar Cascade

The default Haar cascade is trained for **human faces** and is used as a
placeholder.  For production:

1. Obtain or train a cattle-specific Haar cascade (or use a YOLOv8 nano model).
2. Pass the XML path to `load_cascade(cascade_path="path/to/cattle.xml")` in
   `pipeline.py`.

The rest of the pipeline is cascade-agnostic.

---

## Tuning Tips

| Parameter | Where | What to try |
|-----------|-------|-------------|
| `CANONICAL_SIZE` | `enhancer.py` | Increase to 192×192 for richer HOG |
| `clip_limit` | `enhancer.py` | 1.5–3.0; higher = more contrast |
| `HOG_PIXELS_PER_CELL` | `feature_extractor.py` | (4,4) for finer detail |
| `svm_C` | `classifier.py` | Grid search 0.1, 1, 10, 100 |
| `n_components` | `classifier.py` | 50–200 depending on dataset size |
| `use_watershed` | `pipeline.py` | Enable when lighting is consistent |
