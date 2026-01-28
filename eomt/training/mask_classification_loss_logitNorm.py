# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
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


class MaskClassificationLoss(Mask2FormerLoss):
    """
    Standard Mask2Former Loss with a critical modification for Anomaly Detection:
    Logit Normalization with Temperature Scaling (Safe Cosine Similarity).

    This replaces the standard dot-product logits with cosine similarity,
    which bounds the logits and makes the classification space more compact,
    improving OOD (Out-of-Distribution) detection capabilities.
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

        # Set weight for the 'no-object' class (usually low, e.g., 0.1)
        # to prevent the model from trivially predicting 'background' everywhere.
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # Hungarian Matcher: Solves the assignment problem between
        # N predicted masks and M ground truth objects.
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
        Computes the matching and the losses.
        Args:
            masks_queries_logits: [batch, num_queries, height, width]
            targets: List of dicts with 'masks' and 'labels'
            class_queries_logits: [batch, num_queries, num_classes + 1]
        """
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        # 1. Match Predictions to Ground Truth
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        # 2. Compute Losses based on the optimal matching
        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        """Computes Dice and Binary Cross Entropy loss for masks."""
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        # Normalize by the total number of masks across all GPUs
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
        """Weighted sum of all individual losses."""
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

        return loss_total  # type: ignore

    def loss_labels(self, class_queries_logits, class_labels, indices):
        """
        Computes Classification Loss with Cosine Normalization.

        Standard linear classification: Logits = W @ x
        Here: Logits = (W @ x) / (|W| * |x|) / temperature

        This forces the model to learn angular separability rather than magnitude,
        which helps clustering known classes tightly and detecting unknown anomalies.
        """

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(class_labels, indices)])

        # Initialize targets with 'no-object' class
        target_classes = torch.full(
            class_queries_logits.shape[:2],
            self.num_labels,  # no-object index
            dtype=torch.int64,
            device=class_queries_logits.device,
        )
        target_classes[idx] = target_classes_o

        # -------------------------------
        # LOGIT NORMALIZATION (SAFE)
        # -------------------------------
        tau = 0.04  # Temperature parameter: controls the sharpness of the distribution
        eps = 1e-6  # Epsilon for numerical stability

        # 1. Separate real classes [B, Q, C] from no-object [B, Q, 1]
        # We normalize ONLY the real class logits to enforcing cosine similarity.
        # The 'no-object' logit is often left un-normalized or handled differently
        # to allow the model to easily reject background.
        class_logits = class_queries_logits[..., :-1]  # [B, Q, C]
        no_obj_logit = class_queries_logits[..., -1:]  # [B, Q, 1]

        # 2. L2 Normalization on the feature dimension
        norm = torch.norm(class_logits, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps)

        # 3. Apply Temperature Scaling
        # Higher temperature -> softer distribution
        # Lower temperature -> sharper distribution (closer to argmax)
        class_logits = (class_logits / norm) / tau

        # 4. Re-concatenate with the raw 'no-object' logit
        logits = torch.cat([class_logits, no_obj_logit], dim=-1)

        # 5. Ensure Float32 for stability (Crucial for Mixed Precision Training)
        logits = logits.float()
        # -------------------------------

        loss_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="mean",
        )

        return {"loss_cross_entropy": loss_ce}

    def _get_src_permutation_idx(self, indices):
        """Helper to create index tensors for batch operations."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
