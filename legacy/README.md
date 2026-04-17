# Legacy — CNN Embedding Approach

These files contain an earlier deep-learning (ResNet-50 + ArcFace / Triplet loss)
experiment that has been superseded by the HOG+LBP+SVM MVP pipeline described in
the project README.

| File | Description |
|------|-------------|
| `embedding_model.py` | CNN backbone, projection head, ArcFace/Triplet loss, `CattleIdentifier` gallery class |
| `test_features.py` | GradCAM visualisation script for the CNN model |
| `config.py` | CUDA/model-path config for the CNN approach |

To use the CNN pipeline, install PyTorch + torchvision and run
`embedding_model.py` directly.  The MVP pipeline in `pipeline.py` does **not**
depend on these files.
