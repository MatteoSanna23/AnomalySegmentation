# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)
import math


class MaskClassificationLossArcFace(Mask2FormerLoss):
    """
    Custom Loss function extending Mask2Former.

    It replaces the standard Cross Entropy classification loss with
    **ArcFace (Additive Angular Margin Loss)**.

    Why?
    Standard Cross Entropy only encourages correct classification.
    ArcFace enforces a geometric margin on the hypersphere, pushing class clusters
    to be more compact and separated. This is crucial for Anomaly Detection,
    as it creates "empty space" in the feature space where anomalies can be detected.
    """

    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        # --- ARCFACE PARAMS ---
        arcface_s: float = 30.0,  # Scale factor: expands the decision boundary
        arcface_m: float = 0.50,  # Margin: penalty added to the angular distance
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient

        # ArcFace parameters
        self.arcface_s = arcface_s
        self.arcface_m = arcface_m

        # Precompute cos(m) and sin(m) for trigonometric identity usage later
        self.cos_m = math.cos(self.arcface_m)
        self.sin_m = math.sin(self.arcface_m)

        # Thresholds for numerical stability handling in edge cases
        self.th = math.cos(math.pi - self.arcface_m)
        self.mm = math.sin(math.pi - self.arcface_m) * self.arcface_m

        # Class weights (mostly to downweight the background/no-object class)
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # Hungarian Matcher finds the optimal assignment between predictions and ground truth
        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        """
        Main loss calculation steps:
        1. Match predictions to targets.
        2. Calculate Mask Loss (Dice + BCE).
        3. Calculate Class Loss (ArcFace).
        """
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        # Find optimal assignment
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        """Calculates Binary Cross Entropy and Dice Loss for the segmentation masks."""
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        # Normalize loss by the number of masks across all GPUs (for distributed training)
        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        """Aggregates all partial losses into a single scalar for backpropagation."""
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total

    def loss_labels(self, class_queries_logits, class_labels, indices):
        """
        Computes the ArcFace Loss.

        Standard Softmax: L = -log( e^Wy / sum(e^Wj) )
        ArcFace: L = -log( e^(s * cos(theta + m)) / sum(...) )
        """

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(class_labels, indices)])

        # Initialize target tensor with "no-object" class
        target_classes = torch.full(
            class_queries_logits.shape[:2],
            self.num_labels,  # no-object index
            dtype=torch.int64,
            device=class_queries_logits.device,
        )
        # Fill in the matched positions with actual class indices
        target_classes[idx] = target_classes_o

        # -------------------------------
        # ARCFACE LOGIC
        # -------------------------------

        # 1. Separate "Real" Classes vs "No-Object"
        # We generally only apply ArcFace margins to the semantic classes (cars, roads, etc.),
        # leaving the background/no-object class un-normalized or handled normally.
        class_logits = class_queries_logits[..., :-1]  # [B, Q, C]
        no_obj_logit = class_queries_logits[..., -1:]  # [B, Q, 1]

        # 2. L2 Normalization (Embedding & Weights)
        # By normalizing the logits, the dot product becomes equivalent to Cosine Similarity.
        # W * x = |W| * |x| * cos(theta). If |W|=1 and |x|=1, then W*x = cos(theta).
        eps = 1e-6
        norm = torch.norm(class_logits, p=2, dim=-1, keepdim=True)
        cosine = class_logits / torch.clamp(norm, min=eps)

        # 3. Clamp for numerical stability (acos domain is [-1, 1])
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)

        # 4. Create One-Hot mask for Ground Truth
        # We only add the margin penalty (m) to the target class logit.
        has_object_mask = target_classes < self.num_labels
        one_hot = torch.zeros_like(cosine)

        if target_classes_o.numel() > 0:
            one_hot[idx[0], idx[1], target_classes_o] = 1.0

        # 5. Apply Angular Margin: cos(theta + m)
        # Using trig identity: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.arcface_m > 0.0:
            # Replace the target logit (cosine) with the penalized logit (phi)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine

        # 6. Re-Scale (s)
        # Since we normalized everything to 1, gradients would be too small.
        # We multiply by 's' (e.g. 30.0) to bring magnitudes back to a range suitable for Softmax.
        output = output * self.arcface_s

        # 7. Reassemble with No-Object
        # Concatenate the ArcFace-processed class logits with the raw no-object logit.
        logits = torch.cat([output, no_obj_logit], dim=-1)

        # 8. Standard Cross Entropy on the modified logits
        loss_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="mean",
        )

        return {"loss_cross_entropy": loss_ce}

    def _get_src_permutation_idx(self, indices):
        """Helper to create batch indices for scatter operations."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
