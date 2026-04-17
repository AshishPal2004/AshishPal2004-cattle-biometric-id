"""
Visualizes what the CNN is actually focusing on using GradCAM.
If heatmap lights up on the muzzle texture → model is learning correctly.
If heatmap lights up on background → model is memorizing, not learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from embedding_model import CattleIdentifier, get_transforms
from PIL import Image

MODEL_PATH = "models/best_triplet.pt"

class GradCAM:
    def __init__(self, model):
        self.model    = model
        self.gradient = None
        self.activation = None
        # Hook into last conv layer of ResNet
        if hasattr(model.backbone, "layer4"):
         target_layer = model.backbone.layer4[-1].conv3
        else:
             target_layer = model.backbone.features[-1][0]    
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def generate(self, image_tensor):
        self.model.zero_grad()
        emb = self.model(image_tensor)
        # Backprop through embedding norm
        emb.norm().backward()
        weights  = self.gradient.mean(dim=[2, 3], keepdim=True)
        cam      = (weights * self.activation).sum(dim=1).squeeze()
        cam      = F.relu(cam)
        cam      = cam - cam.min()
        cam      = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def visualize_gradcam(image_path: str, save_path: str = "gradcam_output.jpg"):
    ident     = CattleIdentifier(MODEL_PATH)
    ident.model.eval()
    transform = get_transforms(train=False)

    img_pil    = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(ident.device)
    img_tensor.requires_grad_(True)

    cam        = GradCAM(ident.model)
    heatmap    = cam.generate(img_tensor)

    # Overlay on original image
    img_cv     = cv2.imread(image_path)
    img_cv     = cv2.resize(img_cv, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap_colored, 0.5, 0)
    cv2.imwrite(save_path, overlay)
    print(f"[GradCAM] Saved → {save_path}")
    print("  RED   = CNN focused here (high attention)")
    print("  BLUE  = CNN ignored here")
    print("  Good result: RED covers muzzle texture, not background")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ").strip()
    visualize_gradcam(path)