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
        arcface_s: float = 30.0,  # Scale factor, we multiply logits
        arcface_m: float = 0.50,  # Angular margin, add this for improving class separation
        # --- OUTLIER LOSS PARAMS ---
        outlier_temperature: float = 1.0,  # Temperature for softmax in outlier loss
        outlier_loss_weight: float = 1.0,  # Weight for the outlier loss component
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

        # Outlier loss parameters
        self.outlier_temperature = outlier_temperature
        self.outlier_loss_weight = outlier_loss_weight

        # Precompute cos(m) and sin(m) for efficiency usually,
        # but inside forward is safer for dynamic graphs
        self.cos_m = math.cos(self.arcface_m)
        self.sin_m = math.sin(self.arcface_m)

        # Threshold for numerical stability (Taylor expansion or clamp)
        self.th = math.cos(math.pi - self.arcface_m)
        self.mm = math.sin(math.pi - self.arcface_m) * self.arcface_m

        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

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
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

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
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

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
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "rba" in loss_key:
                # RbA loss is already weighted by outlier_loss_weight in loss_labels
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
        Compute the Cross Entropy Loss with conditional ArcFace + Outlier Loss (RbA):
        - Label 0-19 (In-Distribution): Apply ArcFace. Objective: cos(θ_target) → 1
        - Label 20 (Outliers/Cut-Paste): Apply RbA Loss. Objective: max(prob_0-18) → 0
        """
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(class_labels, indices)])

        # Build the full target classes tensor filled with "no-object"
        target_classes = torch.full(
            class_queries_logits.shape[:2],
            self.num_labels,  #  no-object index
            dtype=torch.int64,
            device=class_queries_logits.device,
        )
        target_classes[idx] = target_classes_o

        # OUTLIER LABEL THRESHOLD
        # Label 20 is the outlier/anomaly class (Cut-Paste)
        # Labels 0-19 are in-distribution classes
        outlier_label = 20

        # Separate In-Distribution and Outlier samples
        # For In-Distribution samples (labels 0-19)
        is_inlier = (target_classes >= 0) & (target_classes < outlier_label)

        # For Outlier samples (label 20)
        is_outlier = target_classes == outlier_label

        # ========================
        # PART 1: IN-DISTRIBUTION SAMPLES (0-19) -> ArcFace
        # ========================
        class_logits = class_queries_logits[..., :-1]  # [B, Q, C]
        no_obj_logit = class_queries_logits[..., -1:]  # [B, Q, 1]

        # 2. Normalization L2 (Simulate Cosine Similarity)
        eps = 1e-6
        norm = torch.norm(class_logits, p=2, dim=-1, keepdim=True)
        cosine = class_logits / torch.clamp(norm, min=eps)

        # 3. Clamp for numerical stability of acos
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)

        # 4. Create One-Hot mask for IN-DISTRIBUTION targets
        one_hot = torch.zeros_like(cosine)

        if target_classes_o.numel() > 0:
            # Apply one-hot only for inlier samples (label < 20)
            # idx[0] = batch indices, idx[1] = query indices (both shape [num_matched])
            batch_idx = idx[0]
            query_idx = idx[1]

            # For each matched sample, assign one-hot if it's an inlier (label < 20)
            # We iterate over matched samples to avoid device mismatch issues
            for i in range(len(target_classes_o)):
                label = target_classes_o[i].item()
                if label < outlier_label:
                    # Only apply one-hot for in-distribution classes (0-19)
                    one_hot[batch_idx[i], query_idx[i], label] = 1.0

        # 5. Compute ArcFace: cos(theta + m)a
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.arcface_m > 0.0:
            output_inlier = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output_inlier = cosine

        # 6. Scaling (s)
        output_inlier = output_inlier * self.arcface_s

        # ========================
        # PART 2: OUTLIER SAMPLES (20) -> RbA Loss (Repelling from Boundaries)
        # ========================
        # RbA Objective: For anomalies, push them away from all in-distribution class centers
        # max(prob_0-19) should be 0 (or minimized) for outliers
        # This is naturally handled by standard CE with label 20, BUT we can add an explicit term

        output_full = output_inlier  # [B, Q, C]
        logits_full = torch.cat([output_full, no_obj_logit], dim=-1)  # [B, Q, C+1]

        # ========================
        # COMPUTE STANDARD CROSS ENTROPY
        # ========================
        loss_ce = F.cross_entropy(
            logits_full.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="mean",
        )

        # ========================
        # ADD EXPLICIT OUTLIER REPULSION (Optional but Recommended)
        # ========================
        # For outlier samples: maximize distance from in-distribution classes
        # This is the RbA (Repelling away from Boundaries) component

        loss_rba = self._compute_rba_loss(
            logits_full, target_classes, is_outlier, outlier_label
        )

        # Combine losses
        total_loss = loss_ce + self.outlier_loss_weight * loss_rba

        return {"loss_cross_entropy": total_loss, "loss_rba": loss_rba}

    def _compute_rba_loss(self, logits, target_classes, is_outlier, outlier_label):
        """
        Compute Repelling from Boundaries (RbA) Loss for outlier samples.

        Objective: For outliers (label 20), minimize max(prob_0-19)
        This ensures the model doesn't confuse outliers with in-distribution classes.

        Args:
            logits: [B, Q, C+1] Full logits (including no-object)
            target_classes: [B, Q] Target labels
            is_outlier: [B, Q] Boolean mask for outlier positions
            outlier_label: int, the outlier class index (20)

        Returns:
            loss_rba: scalar tensor
        """
        # Extract logits for in-distribution classes only [B, Q, C]
        inlier_logits = logits[..., :-1]  # Exclude no-object class

        # For outlier samples: compute the maximum probability across in-distribution classes
        # We want max(softmax(inlier_logits)) to be as low as possible

        # Compute softmax over in-distribution classes only (with temperature)
        # Note: We apply softmax per position, getting probabilities for each class
        inlier_probs = F.softmax(inlier_logits / self.outlier_temperature, dim=-1)

        # For each position, get the max probability
        max_inlier_prob, _ = torch.max(inlier_probs, dim=-1)  # [B, Q]

        # Only compute loss for outlier samples
        outlier_max_probs = max_inlier_prob[is_outlier]

        if outlier_max_probs.numel() == 0:
            # No outlier samples in this batch
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # RbA Loss: minimize max probability on in-distribution classes for outliers
        # Using negative log to create a loss that decreases when max_prob decreases
        # loss_rba = -log(1 - max_prob) ≈ max_prob (for small max_prob)
        # Or simpler: loss_rba = max_prob (we want to minimize this)

        loss_rba = outlier_max_probs.mean()

        return loss_rba

    # Function to get source permutation indices
    def _get_src_permutation_idx(self, indices):
        # indices: List[Tuple[src_idx, tgt_idx]]
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
