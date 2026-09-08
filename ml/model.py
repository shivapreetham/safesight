"""RQSA-MIL model definition.

Vendored from the research repo (nude-MIL): IoU-conditioned spatial attention
over a MobileNetV2 backbone, trained with weighted multiple-instance learning.
Reference: RQSA-MIL paper. Architecture must match the training code exactly
so the released checkpoint loads without key remapping.
"""

import torch
import torch.nn as nn
import torchvision


class IoUConditionedSpatialAttention(nn.Module):
    """Spatial attention gate conditioned on a per-region IoU quality score.

    High-IoU regions receive gentle attention while low-IoU regions receive
    aggressive background suppression. At inference time no ground-truth IoU
    exists, so a pseudo-IoU (0.5) is fed instead; this is the paper default.
    """

    def __init__(self, in_channels: int = 1280, iou_embed_dim: int = 64,
                 gate_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.iou_embed_dim = iou_embed_dim
        self.gate_channels = gate_channels

        self.iou_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, iou_embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels + iou_embed_dim, gate_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, iou_weights: torch.Tensor):
        B, C, H, W = features.shape
        if iou_weights.dim() == 1:
            iou_weights = iou_weights.unsqueeze(1)
        iou_emb = self.iou_embed(iou_weights)
        iou_emb_spatial = iou_emb.view(B, self.iou_embed_dim, 1, 1).expand(
            B, self.iou_embed_dim, H, W)
        conditioned = torch.cat([features, iou_emb_spatial], dim=1)
        attention_mask = self.gate(conditioned)
        return features * attention_mask, attention_mask


class MobileNetMILRQSA(nn.Module):
    """MobileNetV2 backbone + IoU-conditioned attention + linear head.

    forward(x, iou_weights) -> (logits (B, 1), attention_masks (B, 1, 7, 7))
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        # Weights always come from the trained checkpoint, so no ImageNet init.
        base = torchvision.models.mobilenet_v2(weights=None)
        self.backbone = base.features
        self.attention = IoUConditionedSpatialAttention(
            in_channels=1280, iou_embed_dim=64, gate_channels=256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x: torch.Tensor, iou_weights: torch.Tensor):
        features = self.backbone(x)
        attended, masks = self.attention(features, iou_weights)
        pooled = self.pool(attended).flatten(1)
        logits = self.fc(self.dropout(pooled))
        return logits, masks

    @torch.no_grad()
    def score_regions(self, regions: torch.Tensor, pseudo_iou: float = 0.5) -> torch.Tensor:
        """Score a batch of 224x224 regions with the test-time pseudo-IoU.

        Returns per-region probabilities (N,).
        """
        iou = torch.full((regions.shape[0], 1), pseudo_iou,
                         device=regions.device, dtype=regions.dtype)
        logits, _ = self.forward(regions, iou)
        return torch.sigmoid(logits.squeeze(-1))
