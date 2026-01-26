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

class MaskClassificationLoss(Mask2FormerLoss):
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
        Compute the Cross Entropy Loss with ArcFace implementation on class logits.
        """
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(class_labels, indices)]
        )

        # Build the full target classes tensor filled with "no-object"
        target_classes = torch.full(
            class_queries_logits.shape[:2],
            self.num_labels,  #  no-object index
            dtype=torch.int64,
            device=class_queries_logits.device,
        )
        target_classes[idx] = target_classes_o

        # -------------------------------
        # ARCFACE IMPLEMENTATION
        # -------------------------------
        # 1. Separate real classes [B, Q, K] and no-object [B, Q, 1]
        # ArcFace is applied only to real classes. Leaving no-object "raw".
        
        class_logits = class_queries_logits[..., :-1]   # [B, Q, C]
        no_obj_logit = class_queries_logits[..., -1:]   # [B, Q, 1]
        
        # 2. Normalization L2 (Simulate Cosine Similarity)
        #  in standard Softmax logits are computed as: logits = W * x * cosine_theta
        #  normalizing dividing by ||x|| (feature norm) we get cosine similarity.
        eps = 1e-6
        norm = torch.norm(class_logits, p=2, dim=-1, keepdim=True)
        cosine = class_logits / torch.clamp(norm, min=eps)

        # 3. Clamp for numerical stability of acos
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps) # between -1 and 1

        # 4. Create One-Hot mask for targets
        #    We want to apply margin m ONLY to the true class index (Ground Truth).
        #    Note: we must ignore targets that are "no-object" (index == self.num_labels)
        
        # [B, Q] -> True where there is an object assigned
        has_object_mask = target_classes < self.num_labels
        
        # One hot encoding only for positions where there is an object
        # Shape: [B, Q, C]
        one_hot = torch.zeros_like(cosine) 
        
        # Scatter 1.0 in the target class positions, but only for rows that have an object
        # We use idx (batch_idx, src_idx) calculated earlier by the Hungarian Matcher
        if target_classes_o.numel() > 0:
            # target_classes_o contains only valid class indices (no-object filtered out)
            one_hot[idx[0], idx[1], target_classes_o] = 1.0

        # 5. Compute ArcFace: cos(theta + m)
        #    Formula: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  
        
        # If theta + m > pi, the function decreases and then increases, ruining optimization.
        if self.arcface_m > 0.0:    # if margin is set
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # apply phi (cos(theta + m)) only to true class
        else:
            output = cosine

        # 6. Scaling (s)
        output = output * self.arcface_s

        # 7. Recomposition with no-object (not normalized)
        # [B, Q, C + 1], containing modified logits for real classes and raw no-object logit
        logits = torch.cat([output, no_obj_logit], dim=-1)

        # 8. Cross Entropy
        loss_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="mean",
        )

        return {"loss_cross_entropy": loss_ce}

    # Function to get source permutation indices
    def _get_src_permutation_idx(self, indices):
        # indices: List[Tuple[src_idx, tgt_idx]]
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx