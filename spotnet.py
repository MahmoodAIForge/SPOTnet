"""
SPOTnet: Weakly-Supervised Specular Reflection Detection for Minimal Access Surgery.

This module provides the SPOTnet detector (a lightweight DetUNet, ~0.77M parameters)
together with tiled full-resolution inference and simple helper functions.

Example
-------
    import cv2
    from spotnet import load_spotnet, predict_mask

    net = load_spotnet("weights/spotnet.pt")
    img = cv2.cvtColor(cv2.imread("frame.png"), cv2.COLOR_BGR2RGB)
    mask = predict_mask(net, img)          # binary specular mask (H, W), values in {0, 1}
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DetUNet", "to_tensor", "tiled_inference", "load_spotnet", "predict_mask"]

# Default inference settings (as used in the paper).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE = 384
OVERLAP = 96
THRESHOLD = 0.13


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and SiLU."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DetUNet(nn.Module):
    """Lightweight U-Net specular detector (~0.77M parameters at base=32)."""

    def __init__(self, base=32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.b = ConvBlock(base * 4, base * 4)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.avg_pool2d(s1, 2))
        s3 = self.e3(F.avg_pool2d(s2, 2))
        b = self.b(s3)
        x = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, s2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, s1], dim=1))
        return self.head(x)


def to_tensor(img_rgb):
    """Convert an HxWx3 uint8 RGB image to a normalised CxHxW float tensor."""
    if not img_rgb.flags["C_CONTIGUOUS"]:
        img_rgb = np.ascontiguousarray(img_rgb)
    return torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0


@torch.no_grad()
def tiled_inference(net, img_rgb, tile=TILE, overlap=OVERLAP, device=DEVICE):
    """Run the detector on a full-resolution image via overlapping tiles.

    Returns a soft specular probability map (H, W) with values in [0, 1].
    """
    net.eval()
    H, W = img_rgb.shape[:2]
    X = to_tensor(img_rgb).unsqueeze(0).to(device)
    stride = tile - overlap
    pad_h = (math.ceil((H - tile) / stride) * stride + tile - H) if H > tile else tile - H
    pad_w = (math.ceil((W - tile) / stride) * stride + tile - W) if W > tile else tile - W
    X = F.pad(X, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = X.shape[-2:]
    acc = torch.zeros((H, W), device=device)
    cnt = torch.zeros((H, W), device=device)
    for y in range(0, Hp - tile + 1, stride):
        for x in range(0, Wp - tile + 1, stride):
            pred = torch.sigmoid(net(X[:, :, y:y + tile, x:x + tile]))[0, 0]
            ye, xe = min(y + tile, H), min(x + tile, W)
            acc[y:ye, x:xe] += pred[:ye - y, :xe - x]
            cnt[y:ye, x:xe] += 1
    return (acc / cnt.clamp(min=1)).cpu().numpy()


def load_spotnet(ckpt_path, base=32, device=DEVICE):
    """Load the SPOTnet detector from a checkpoint saved as {'net': state_dict}."""
    net = DetUNet(base).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    net.load_state_dict(state)
    return net


def predict_mask(net, img_rgb, threshold=THRESHOLD, tile=TILE, overlap=OVERLAP, device=DEVICE):
    """Predict a binary specular mask (H, W) with values in {0, 1}."""
    soft = tiled_inference(net, img_rgb, tile=tile, overlap=overlap, device=device)
    return (soft > threshold).astype(np.float32)


if __name__ == "__main__":
    # Quick sanity check: instantiate the model and report its parameter count.
    n = sum(p.numel() for p in DetUNet(32).parameters())
    print(f"DetUNet(base=32) parameters: {n:,} ({n / 1e6:.2f}M)")
