"""
embedding_model.py
------------------
CNN-based embedding model for cattle muzzle identification.

Architecture
------------
  YOLO crop → ResNet-50 / EfficientNet-B2 backbone (pretrained)
            → Global Average Pooling
            → FC projection head
            → L2-normalised 128-dim embedding

Training
--------
  ArcFace loss (closed-set)  or  Triplet / SupCon (open-set / re-ID)
  Optimizer : AdamW + cosine LR schedule
  Augments  : random flip, colour jitter, random erasing

Inference
---------
  build_gallery()  → stores one embedding per cattle ID
  identify()       → cosine similarity query vs gallery, returns top-k
"""

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

EMBED_DIM    = 128
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
EPOCHS       = 40
LR           = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS= 5


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms(train: bool = True) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MuzzleEmbedder(nn.Module):
    """
    Backbone + projection head that produces L2-normalised embeddings.

    Parameters
    ----------
    backbone    : 'resnet50' | 'efficientnet_b2'
    embed_dim   : output embedding dimension
    n_classes   : number of cattle IDs (required when using ArcFace/CE head)
    pretrained  : load ImageNet weights
    freeze_base : freeze all backbone weights except the last block
    """

    def __init__(
        self,
        backbone: Literal["resnet50", "efficientnet_b2"] = "resnet50",
        embed_dim: int = EMBED_DIM,
        n_classes: int | None = None,
        pretrained: bool = True,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Backbone ───────────────────────────────────────────────────────
        weights = "DEFAULT" if pretrained else None

        if backbone == "resnet50":
            base       = models.resnet50(weights=weights)
            feat_dim   = base.fc.in_features           # 2048
            base.fc    = nn.Identity()                 # remove classifier
            self.backbone = base

        elif backbone == "efficientnet_b2":
            base            = models.efficientnet_b2(weights=weights)
            feat_dim        = base.classifier[1].in_features  # 1408
            base.classifier = nn.Identity()
            self.backbone   = base

        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}")

        # ── Optionally freeze all but last residual block ──────────────────
        if freeze_base:
            for name, param in self.backbone.named_parameters():
                if backbone == "resnet50" and "layer4" in name:
                    break
                param.requires_grad_(False)

        # ── Projection head ────────────────────────────────────────────────
        # BN → FC → BN → ReLU → FC
        # Two-layer head gives better embedding geometry than a single linear.
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embed_dim, bias=False),
        )
        # ── Optional classification head (for ArcFace / cross-entropy) ────
        self.classifier = (
            nn.Linear(embed_dim, n_classes, bias=False)
            if n_classes is not None else None
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x             : (B, 3, H, W)
        return_logits : also return raw classifier logits (training only)

        Returns
        -------
        embeddings : (B, embed_dim)  L2-normalised
        logits     : (B, n_classes)  only when return_logits=True
        """
        feat = self.backbone(x)                    # (B, feat_dim)
        emb  = self.head(feat)                     # (B, embed_dim)
        emb  = F.normalize(emb, p=2, dim=1)        # L2 norm → unit sphere

        if return_logits:
            if self.classifier is None:
                raise RuntimeError("Model has no classification head. Pass n_classes at init.")
            return emb, self.classifier(emb)

        return emb


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class ArcFaceLoss(nn.Module):
    """
    ArcFace margin loss for closed-set classification.
    Penalises the angular distance between an embedding and its class centre.
    Works best when the full cattle roster is known at training time.

    Reference: Deng et al., 2019 (https://arxiv.org/abs/1801.07698)
    """

    def __init__(self, embed_dim: int, n_classes: int, s: float = 32.0, m: float = 0.20):
        super().__init__()
        self.s = s
        self.m = m
        # Learnable class centres on the unit sphere
        self.W = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Cosine similarity between embedding and each class centre
        W_norm  = F.normalize(self.W, p=2, dim=1)
        cosine  = F.linear(embeddings, W_norm)        # (B, n_classes)
        cosine  = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Add angular margin m to the target class only
        theta   = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        cosine_m = torch.cos(theta + self.m * one_hot)

        logits   = self.s * cosine_m
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """
    Batch-hard triplet loss for open-set / re-identification scenarios.
    Mines the hardest positive and hardest negative within each batch.
    Use this when new cattle will be enrolled without retraining.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)   # (B, B)

        same  = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff  = ~same

        # Hardest positive (largest distance, same class)
        pos = (dist * same.float()).max(dim=1).values

        # Hardest negative (smallest distance, different class — set same=inf)
        dist_neg = dist.masked_fill(same, float("inf"))
        neg = dist_neg.min(dim=1).values

        loss = F.relu(pos - neg + self.margin)
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MuzzleDataset(Dataset):
    """
    Simple dataset that loads pre-cropped muzzle images from a folder tree:

        dataset/
            cow_001/   img1.jpg, img2.jpg, ...
            cow_002/   ...

    If you have raw images + YOLO detections, run a one-time crop script first
    and then point this dataset at the cropped folder.
    """

    def __init__(self, root: str, transform=None):
        self.dataset   = ImageFolder(root=root, transform=transform)
        self.classes   = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(
    dataset_dir: str,
    save_path: str          = "models/muzzle_embedder.pt",
    backbone: str           = "resnet50",
    loss_type: str          = "arcface",   # 'arcface' | 'triplet'
    epochs: int             = EPOCHS,
    batch_size: int         = BATCH_SIZE,
    lr: float               = LR,
    val_split: float        = 0.15,
    device: str | None      = None,
) -> MuzzleEmbedder:
    """
    Train the embedding model.

    Parameters
    ----------
    dataset_dir : root of pre-cropped muzzle dataset
    save_path   : where to write the trained model weights
    backbone    : 'resnet50' or 'efficientnet_b2'
    loss_type   : 'arcface' (closed-set) or 'triplet' (open-set / re-ID)
    epochs      : total training epochs
    batch_size  : images per batch
    lr          : peak learning rate
    val_split   : fraction of data held out for validation
    device      : 'cuda' / 'mps' / 'cpu' — auto-detected if None

    Returns
    -------
    Trained MuzzleEmbedder (also saved to save_path).
    """
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    print(f"[Train] Device: {device}")

    # ── Dataset ────────────────────────────────────────────────────────────
    full_ds = MuzzleDataset(dataset_dir, transform=get_transforms(train=True))
    n_classes = len(full_ds.classes)
    print(f"[Train] {len(full_ds)} images | {n_classes} cattle IDs")

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    # Validation uses test-time transforms
    val_ds.dataset = MuzzleDataset(dataset_dir, transform=get_transforms(train=False))

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # ── Model ──────────────────────────────────────────────────────────────
    model = MuzzleEmbedder(
        backbone=backbone,
        embed_dim=EMBED_DIM,
        n_classes=(n_classes if loss_type == "arcface" else None),
        pretrained=True,
        freeze_base=False,
    ).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    if loss_type == "arcface":
        criterion = ArcFaceLoss(EMBED_DIM, n_classes).to(device)
    elif loss_type == "triplet":
        criterion = TripletLoss(margin=0.3)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    # ── Optimiser + scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": model.head.parameters(),     "lr": lr},
        {"params": criterion.parameters(),      "lr": lr},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=lr * 0.01,
        
    )

    # ── Loop ───────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_rank1    = 0.0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if loss_type == "arcface":
                embeddings = model(imgs)
                loss = criterion(embeddings, labels)
            else:
                embeddings = model(imgs)
                loss = criterion(embeddings, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation (embedding-space rank-1 accuracy)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            all_embs, all_labels = [], []
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                embs = model(imgs)
                all_embs.append(embs.cpu())
                all_labels.append(labels.cpu())
                if loss_type == "arcface":
                    val_loss += criterion(embs, labels).item()
                else:
                    val_loss += criterion(embs, labels).item()

        val_loss /= max(len(val_loader), 1)

        # Rank-1 retrieval accuracy on validation set
        all_embs   = torch.cat(all_embs)
        all_labels = torch.cat(all_labels)
        rank1      = _rank1_accuracy(all_embs, all_labels)

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  rank-1={rank1:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if rank1 > best_rank1:
            best_rank1    = rank1
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes":     full_ds.classes,
                "embed_dim":   EMBED_DIM,
                "backbone":    backbone,
                "n_classes":   n_classes,
            }, save_path)
            print(f"    ✓ Best model saved → {save_path}")

    print(f"\n[Train] Done. Best val loss: {best_val_loss:.4f}")
    return model


def _rank1_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Leave-one-out rank-1 retrieval accuracy."""
    sim    = embeddings @ embeddings.T
    sim.fill_diagonal_(-2.0)          # exclude self
    preds  = sim.argmax(dim=1)
    return (labels[preds] == labels).float().mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# 6. INFERENCE — GALLERY + COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

class CattleIdentifier:
    """
    Load a trained MuzzleEmbedder, build an embedding gallery from a folder
    of known cattle images, then identify query images via cosine similarity.

    Usage
    -----
    >>> ident = CattleIdentifier("models/muzzle_embedder.pt")
    >>> ident.build_gallery("dataset/")
    >>> cattle_id, score = ident.identify("query.jpg")
    """

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_transforms(train=False)

        ckpt = torch.load(model_path, map_location=self.device)
       
        self.classes = ckpt.get("classes", [])
        n_classes = ckpt.get("n_classes", None)
        state_dict = ckpt.get("model_state") or ckpt.get("state_dict")
        self.model = MuzzleEmbedder(
            backbone=ckpt.get("backbone", "resnet50"),
            embed_dim=ckpt.get("embed_dim", EMBED_DIM),
            n_classes=n_classes,
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.gallery_embs: torch.Tensor | None = None
        self.gallery_ids:  list[str]           = []

    # ── Gallery ───────────────────────────────────────────────────────────

    def build_gallery(
        self,
        gallery_dir: str,
        aggregation: Literal["mean", "max"] = "mean",
    ) -> None:
        """
        Embed all images in gallery_dir and store one representative
        embedding per cattle ID (mean or max pooling over images).

        Parameters
        ----------
        gallery_dir : same folder structure as training (cow_001/, cow_002/, ...)
        aggregation : 'mean'  → average all embeddings for the same ID
                      'max'   → take the component-wise maximum (max-pooling)
        """
        gallery_dir = Path(gallery_dir)
        per_class: dict[str, list[torch.Tensor]] = {}

        for cls_dir in sorted(gallery_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cattle_id = cls_dir.name
            embs = []
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                emb = self._embed_path(str(img_path))
                if emb is not None:
                    embs.append(emb)
            if embs:
                per_class[cattle_id] = embs

        if not per_class:
            raise ValueError(f"No images found in {gallery_dir}")

        ids, vecs = [], []
        for cattle_id, embs in sorted(per_class.items()):
            stack = torch.stack(embs)               # (N, embed_dim)
            if aggregation == "mean":
                rep = stack.mean(dim=0)
            else:
                rep = stack.max(dim=0).values
            ids.append(cattle_id)
            vecs.append(F.normalize(rep, p=2, dim=0))

        self.gallery_ids  = ids
        self.gallery_embs = torch.stack(vecs)       # (n_cattle, embed_dim)
        print(f"[Gallery] Built: {len(ids)} cattle IDs")

    # ── Query ─────────────────────────────────────────────────────────────

    def identify(
        self,
        image: str | np.ndarray,
        top_k: int = 3,
        threshold: float = 0.50,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Identify a cattle from a muzzle image.

        Parameters
        ----------
        image     : file path or pre-cropped numpy BGR array
        top_k     : return the top-k candidates
        threshold : cosine similarity below this → label as 'unknown'

        Returns
        -------
        cattle_id   : predicted ID (or 'unknown')
        confidence  : cosine similarity of the best match [0, 1]
        top_k_hits  : [(cattle_id, score), ...]  for all top-k candidates
        """
        if self.gallery_embs is None:
            raise RuntimeError("Call build_gallery() first.")

        emb   = self._embed(image)                  # (1, embed_dim)
        sims  = (self.gallery_embs @ emb.T).squeeze()   # (n_cattle,)
        probs = (sims + 1) / 2                      # cosine [-1,1] → [0,1]

        top_idx    = probs.argsort(descending=True)[:top_k]
        top_k_hits = [(self.gallery_ids[i], probs[i].item()) for i in top_idx]

        best_id, best_score = top_k_hits[0]
        if best_score < threshold:
            best_id = "unknown"

        return best_id, best_score, top_k_hits

    # ── Helpers ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed(self, image: str | np.ndarray) -> torch.Tensor:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            import cv2
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return self.model(tensor)                    # (1, embed_dim)

    def _embed_path(self, path: str) -> torch.Tensor | None:
        try:
            return self._embed(path).squeeze(0)     # (embed_dim,)
        except Exception as exc:
            print(f"  [WARN] Could not embed {path}: {exc}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train cattle muzzle embedder")
    parser.add_argument("--dataset",  default="dataset",              help="Training dataset root")
    parser.add_argument("--model",    default="models/muzzle_embedder.pt")
    parser.add_argument("--backbone", default="resnet50",             choices=["resnet50", "efficientnet_b2"])
    parser.add_argument("--loss",     default="arcface",              choices=["arcface", "triplet"])
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--batch",    type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",       type=float, default=LR)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        save_path=args.model,
        backbone=args.backbone,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )