# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    """
    Lightning Module specifically designed for Semantic Segmentation tasks.

    It extends the base LightningModule by configuring it for:
    1. Mask-based classification (Mask2Former approach).
    2. Semantic segmentation metrics (mIoU).
    3. Sliding window inference during evaluation.
    """

    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,  # <--- CRITICAL: Receives the pre-configured loss (e.g., ArcFace) from YAML
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        """
        Args:
            network: The neural network architecture (e.g., EoMT/Mask2Former).
            criterion: The loss function module. Passing this as an argument allows
                       us to easily swap between standard CrossEntropy and ArcFace
                       via the configuration file without changing code.
            img_size: Input resolution [H, W].
            mask_thresh: Probability threshold for generating binary masks during inference.
            overlap_thresh: Threshold for handling overlapping masks.
        """
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            criterion=criterion,  # Register the injected loss function
        )

        # Save hyperparameters for checkpointing, excluding large objects/modules
        self.save_hyperparameters(ignore=["_class_path", "network", "criterion"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        # Initialize specific metrics for Semantic Segmentation (mIoU)
        # We initialize metrics for every decoder block if deep supervision is enabled.
        self.init_metrics_semantic(
            ignore_idx,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        """
        Performs evaluation (Validation/Test) using Sliding Window Inference.

        Since semantic segmentation images (e.g., Cityscapes) are often larger
        than the training crop size, we:
        1. Crop the large image into overlapping windows.
        2. Run the model on each window.
        3. Stitch the predictions back together (averaging overlaps).
        """

        imgs, targets = batch
        img_sizes = [img.shape[-2:] for img in imgs]

        # 1. Generate Crops
        crops, origins = self.window_imgs_semantic(imgs)

        # 2. Forward Pass on Crops
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        # Prepare targets for metric calculation
        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        # 3. Process outputs from each decoder layer (Deep Supervision)
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            # Upsample mask logits to crop size
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # Convert to per-pixel probabilities
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)

            # Stitch crops back to original image size
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            # Update mIoU metrics
            self.update_metrics_semantic(logits, targets, i)

            # Visualize the first batch (useful for debugging in WandB)
            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        """Aggregates validation metrics at the end of the epoch."""
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        """Logs the final summary of validation metrics."""
        self._on_eval_end_semantic("val")
