"""Phase 3 & Phase 4: End-to-End Multi-Task Network & Peril-Conditioned Severity Head.

Implements:
- FR-7: MultiTaskDisasterNet with Inception-ResNet-v2 (num_classes=0, 1536-D) + optional
  handcrafted fusion, 3-stage unfreezing schedule (Stage A/B/C), and BatchNorm2d freeze
  guard when physical batch size < 32.
- FR-8: Mutually exclusive imbalance loss arms:
    (a) EffectiveNumberFocalLoss (Cui et al., beta=0.999, gamma=2.0)
    (b) LogitAdjustedCrossEntropy (Menon et al., tau=1.0)
    (c) Standard CrossEntropyLoss with label smoothing (paired with ClassBalancedSampler)
- FR-9: PerilConditionedSeverityHead computing sum_k p(peril=k) * head_k(f) with a shared
  low-rank adapter for minority perils (flood, landslide).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

try:
    import timm
except Exception:
    timm = None


class EffectiveNumberFocalLoss(nn.Module):
    """Class-Balanced Focal Loss using effective number of samples (1 - beta) / (1 - beta^n_k)."""

    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        counts = np.maximum(1.0, np.asarray(class_counts, dtype=np.float64))
        eff_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / eff_num
        weights = weights / np.sum(weights) * len(counts)
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        w = self.class_weights[targets]
        return torch.mean(w * focal_factor * ce)


class LogitAdjustedCrossEntropy(nn.Module):
    """Logit-Adjusted Cross-Entropy Loss: CE(logits + tau * log(pi), y)."""

    def __init__(
        self,
        class_counts: Sequence[int],
        tau: float = 1.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        counts = np.maximum(1.0, np.asarray(class_counts, dtype=np.float64))
        priors = counts / np.sum(counts)
        log_priors = tau * np.log(priors)
        self.register_buffer("logit_adjustment", torch.tensor(log_priors, dtype=torch.float32))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.logit_adjustment.unsqueeze(0)
        return F.cross_entropy(adjusted, targets, label_smoothing=self.label_smoothing)


def build_class_balanced_sampler(labels: Sequence[int], power: float = 0.5) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler with inverse-sqrt class frequency weights."""
    y = np.asarray(labels, dtype=np.int64)
    counts = np.maximum(1, np.bincount(y))
    class_weights = np.power(counts.astype(np.float64), -power)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
    )


class PerilConditionedSeverityHead(nn.Module):
    """Phase 4: Peril-conditioned severity head with shared low-rank adapter for flood & landslide.

    Computes:
        logits_severity = sum_{k=1}^K p(peril = k) * head_k(f)
    where minority classes (flood_idx=1, landslide_idx=3) share a low-rank residual adapter
    on top of a shared base projection to prevent overfitting on N=222 / N=64 samples.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_perils: int = 6,
        num_severity: int = 3,
        low_rank: int = 16,
        minority_peril_indices: Tuple[int, ...] = (1, 3),
    ) -> None:
        super().__init__()
        self.num_perils = num_perils
        self.num_severity = num_severity
        self.minority_peril_indices = set(minority_peril_indices)

        # Shared base severity linear layer
        self.shared_base = nn.Linear(in_features, num_severity)
        # Per-peril specific heads for majority perils
        self.peril_heads = nn.ModuleList([nn.Linear(in_features, num_severity) for _ in range(num_perils)])
        # Shared low-rank adapter for minority perils (flood, landslide)
        self.minority_down = nn.Linear(in_features, low_rank, bias=False)
        self.minority_up = nn.Linear(low_rank, num_severity, bias=False)

    def forward(self, feat: torch.Tensor, peril_probs: torch.Tensor) -> torch.Tensor:
        """Compute peril-conditioned severity logits.

        Args:
            feat: [B, in_features] trunk representation
            peril_probs: [B, K] detached or soft peril probability distribution
        """
        base_out = self.shared_base(feat)
        min_adapter = self.minority_up(F.gelu(self.minority_down(feat)))

        per_peril_logits: List[torch.Tensor] = []
        for k in range(self.num_perils):
            if k in self.minority_peril_indices:
                # Shared base + shared low-rank adapter + small peril residual
                logits_k = base_out + min_adapter + 0.25 * self.peril_heads[k](feat)
            else:
                logits_k = 0.5 * base_out + 0.5 * self.peril_heads[k](feat)
            per_peril_logits.append(logits_k.unsqueeze(1))

        # Stack to [B, K, num_severity] and weight by peril_probs [B, K, 1]
        stacked = torch.cat(per_peril_logits, dim=1)
        conditioned = torch.sum(stacked * peril_probs.unsqueeze(-1), dim=1)
        return conditioned


class MultiTaskDisasterNet(nn.Module):
    """Multi-task Inception-ResNet-v2 with staged unfreezing and BatchNorm freeze discipline."""

    def __init__(
        self,
        num_perils: int = 6,
        num_severity: int = 3,
        num_informativeness: int = 2,
        handcrafted_dim: int = 36,
        trunk_dim: int = 512,
        dropout: float = 0.4,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required to construct MultiTaskDisasterNet")

        self.backbone = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=0)
        self.backbone_dim = int(getattr(self.backbone, "num_features", 1536))
        self.handcrafted_dim = handcrafted_dim
        self.freeze_bn = freeze_bn

        if handcrafted_dim > 0:
            self.handcrafted_proj = nn.Sequential(
                nn.LayerNorm(handcrafted_dim),
                nn.Linear(handcrafted_dim, 64),
                nn.GELU(),
            )
            fusion_in = self.backbone_dim + 64
        else:
            self.handcrafted_proj = None
            fusion_in = self.backbone_dim

        self.trunk = nn.Sequential(
            nn.Linear(fusion_in, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.head_disaster_type = nn.Linear(trunk_dim, num_perils)
        self.head_damage = PerilConditionedSeverityHead(
            in_features=trunk_dim,
            num_perils=num_perils,
            num_severity=num_severity,
        )
        self.head_informativeness = nn.Linear(trunk_dim, num_informativeness)

        # Default to Stage A (backbone frozen, heads trainable)
        self.set_unfreeze_stage("A")

    def train(self, mode: bool = True) -> "MultiTaskDisasterNet":
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        return self

    def set_unfreeze_stage(self, stage: str) -> None:
        """Apply Stage A, Stage B, or Stage C unfreezing schedule (FR-7)."""
        stage_u = stage.strip().upper()
        for param in self.backbone.parameters():
            param.requires_grad = False

        if stage_u in ("B", "C"):
            for block_name in ("mixed_7a", "repeat_2", "block8", "conv2d_7b"):
                block = getattr(self.backbone, block_name, None)
                if block is not None:
                    for param in block.parameters():
                        param.requires_grad = True

        if stage_u == "C":
            mixed_6a = getattr(self.backbone, "mixed_6a", None)
            if mixed_6a is not None:
                for param in mixed_6a.parameters():
                    param.requires_grad = True
            repeat_1 = getattr(self.backbone, "repeat_1", None)
            if repeat_1 is not None and isinstance(repeat_1, nn.Sequential):
                for subblock in list(repeat_1)[10:20]:
                    for param in subblock.parameters():
                        param.requires_grad = True

        if self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

    def parameter_groups(self, stage: str = "A") -> List[Dict[str, object]]:
        """Return differential learning-rate parameter groups for AdamW."""
        stage_u = stage.strip().upper()
        head_params: List[nn.Parameter] = []
        for mod in (self.handcrafted_proj, self.trunk, self.head_disaster_type, self.head_damage, self.head_informativeness):
            if mod is not None:
                head_params.extend([p for p in mod.parameters() if p.requires_grad])

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_lr = 1e-3 if stage_u == "A" else (3e-4 if stage_u == "B" else 1e-4)
        bb_lr = 1e-5 if stage_u == "B" else 5e-6

        groups: List[Dict[str, object]] = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": bb_lr})
        return groups

    def forward_from_embeddings(
        self,
        pooled_1536: torch.Tensor,
        handcrafted: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the multi-task heads given pre-extracted 1,536-D pooled features."""
        if self.handcrafted_proj is not None and handcrafted is not None:
            hc = self.handcrafted_proj(handcrafted)
            fused = torch.cat([pooled_1536, hc], dim=1)
        elif self.handcrafted_proj is not None:
            zeros = torch.zeros((pooled_1536.size(0), 64), device=pooled_1536.device, dtype=pooled_1536.dtype)
            fused = torch.cat([pooled_1536, zeros], dim=1)
        else:
            fused = pooled_1536

        feat = self.trunk(fused)
        logits_peril = self.head_disaster_type(feat)
        peril_probs = F.softmax(logits_peril, dim=-1)
        logits_damage = self.head_damage(feat, peril_probs)
        logits_info = self.head_informativeness(feat)
        return {
            "disaster_type": logits_peril,
            "damage": logits_damage,
            "informativeness": logits_info,
            "peril_probs": peril_probs,
        }

    def forward(
        self,
        images: torch.Tensor,
        handcrafted: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat_maps = self.backbone.forward_features(images)
        pooled = self.backbone.forward_head(feat_maps, pre_logits=True)
        return self.forward_from_embeddings(pooled, handcrafted=handcrafted)

