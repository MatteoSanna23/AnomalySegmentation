# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional, cast
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality, MeanAveragePrecision
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _Color,
    _get_color_areas,
    _calculate_iou,
)
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad
import logging

from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    """
    Main PyTorch Lightning module controlling the training loop, validation,
    optimization, and logging.
    """

    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float = 1e-4,
        llrd: float = 0.9,
        llrd_l2_enabled: bool = False,
        lr_mult: float = 1.0,
        weight_decay: float = 0.0,
        poly_power: float = 0.9,
        warmup_steps: tuple[int, int] = (1000, 1000),
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=True,
        criterion: Optional[nn.Module] = None,
        ignore_idx: int = 255,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        # Parameters for Mask2Former-style attention mask annealing
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps

        # Optimization hyperparameters
        self.lr = lr
        self.llrd = llrd  # Layer-wise Learning Rate Decay factor
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.llrd_l2_enabled = llrd_l2_enabled
        self.criterion = criterion
        self.ignore_idx = ignore_idx

        self.strict_loading = False

        # Initialize IoU metrics for semantic segmentation
        self.init_metrics_semantic(
            ignore_idx,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

        # --- CHECKPOINT LOADING LOGIC ---
        # Handles complex scenarios like Delta Weights or partial loading
        if delta_weights and ckpt_path:
            logging.info("Delta weights mode")
            # Initialize non-encoder weights to zero (for additive fine-tuning)
            self._zero_init_outside_encoder(skip_class_head=not load_ckpt_class_head)
            current_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}

            # Filter out class head if we are changing the number of classes
            if not load_ckpt_class_head:
                current_state_dict = {
                    k: v
                    for k, v in current_state_dict.items()
                    if "class_head" not in k and "class_predictor" not in k
                }

            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            combined_state_dict = self._add_state_dicts(current_state_dict, ckpt)
            incompatible_keys = self.load_state_dict(combined_state_dict, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)
        elif ckpt_path:
            # Standard checkpoint loading
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            incompatible_keys = self.load_state_dict(ckpt, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)

        # --- FINAL PARAMETER PURIFICATION ---
        # Some pretrained ViT checkpoints (like DINOv2) might contain ComplexFloat parameters
        # related to RoPE or FFTs. We convert them to standard Float32 to ensure
        # stability during training and optimizer steps.
        for name, module in self.network.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if torch.is_complex(param) or "complex" in str(param.dtype):
                    print(f"Purifying complex parameter in memory: {name}.{param_name}")
                    new_data = (
                        param.data.real.detach().clone().float()
                    )  # Keep only the real part
                    new_param = nn.Parameter(new_data)
                    new_param.requires_grad = param.requires_grad
                    setattr(module, param_name, new_param)
                else:
                    param.data = param.data.float()
        # -----------------------------------------

        self.log = torch.compiler.disable(self.log)  # type: ignore

        if self.criterion is not None:
            logging.info(
                f"{bold_green}Using loss function: {self.criterion.__class__.__name__}{reset}"
            )

    def configure_optimizers(self):
        """
        Sets up the optimizer and scheduler.
        Crucial logic:
        1. Freezes the encoder (Backbone) to support LoRA/Transfer learning.
        2. Applies Layer-wise Learning Rate Decay (LLRD) to backbone layers if they are unfrozen.
        """

        # 1. Force freeze the encoder backbone
        for param in self.network.encoder.parameters():
            param.requires_grad = False

        encoder_param_names = {
            n for n, _ in self.network.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.network.encoder.backbone.blocks)

        # Identify which blocks are involved in L2 regularization/decay
        l2_blocks = torch.arange(
            backbone_blocks - self.network.num_blocks, backbone_blocks
        ).tolist()

        # 2. Iterate through all parameters to assign specific learning rates
        for name, param in reversed(list(self.named_parameters())):
            # Skip frozen parameters (like the base weights in LoRA)
            if not param.requires_grad:
                continue

            lr = self.lr

            # Handle Backbone Parameters (if any are unfrozen)
            if name.replace("network.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")
                is_block = False
                block_i = backbone_blocks

                # Determine block index for LLRD
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True

                # Apply geometric decay to learning rate based on depth
                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)
                elif (is_block or block_i == 0) and self.lr_mult != 1.0:
                    lr *= self.lr_mult

                if "backbone.norm" in name:
                    lr = self.lr

                if (
                    is_block
                    and (block_i in l2_blocks)
                    and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0))
                ):
                    lr = self.lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                # Handle Head/Decoder/LoRA Parameters
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups

        # 3. Initialize Optimizer
        # foreach=False avoids issues with complex tensors if any remain
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay, foreach=False)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs):
        """Standard forward pass with input normalization."""
        x = imgs / 255.0
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """
        Executes one training step.
        Calculates loss for the final output and auxiliary losses for intermediate blocks.
        """
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        # Iterate over outputs from all decoder blocks (deep supervision)
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        """
        Shared evaluation logic for Validation and Testing.
        Performs sliding window inference to handle high-resolution images.
        """

        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]

        # 1. Create crops (sliding window)
        crops, origins = self.window_imgs_semantic(imgs)

        # 2. Run inference on crops
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        # 3. Stitch predictions back to full image size
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            # Resize mask logits to match the crop size
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # Convert mask/class logits to per-pixel probabilities
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)

            # Reconstruct full image logits from crops
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            # Update metrics (IoU)
            self.update_metrics_semantic(logits, targets, i)

            # Visualize the first batch for debugging
            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        """
        Mask2Former strategy: Gradually transition from global attention (mask=1)
        to local masked attention (mask=0) based on training progress.
        """

        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        """Updates the attention mask probability at the end of every step."""
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def init_metrics_semantic(self, ignore_idx, num_blocks):
        """Initializes IoU metrics for each decoder block."""
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )

    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList(
            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
        )

    def init_metrics_panoptic(self, thing_classes, stuff_classes, num_blocks):
        self.metrics = nn.ModuleList(
            [
                PanopticQuality(
                    thing_classes,
                    stuff_classes + [self.num_classes],
                    return_sq_and_rq=True,
                    return_per_class=True,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    @torch.compiler.disable
    def update_metrics_instance(
        self,
        preds: list[dict],
        targets: list[dict],
        block_idx,
    ):
        self.metrics[block_idx].update(preds, targets)

    @torch.compiler.disable
    def update_metrics_panoptic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        is_crowds: list[torch.Tensor],
        block_idx,
    ):
        """Complex metric calculation for Panoptic Quality (PQ)."""
        for i in range(len(preds)):
            metric = self.metrics[block_idx]
            flatten_pred = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                preds[i][None, ...],
                metric.void_color,
                metric.allow_unknown_preds_category,
            )[0]
            flatten_target = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                targets[i][None, ...],
                metric.void_color,
                True,
            )[0]

            pred_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_pred)
            )
            target_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_target)
            )
            intersection_matrix = torch.transpose(
                torch.stack((flatten_pred, flatten_target), -1), -1, -2
            )
            intersection_areas = cast(
                dict[tuple[_Color, _Color], torch.Tensor],
                _get_color_areas(intersection_matrix),
            )

            pred_segment_matched = set()
            target_segment_matched = set()
            for pred_color, target_color in intersection_areas:
                if is_crowds[i][target_color[1]]:
                    continue
                if target_color == metric.void_color:
                    continue
                if pred_color[0] != target_color[0]:
                    continue
                iou = _calculate_iou(
                    pred_color,
                    target_color,
                    pred_areas,
                    target_areas,
                    intersection_areas,
                    metric.void_color,
                )
                continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
                if iou > 0.5:
                    pred_segment_matched.add(pred_color)
                    target_segment_matched.add(target_color)
                    metric.iou_sum[continuous_id] += iou
                    metric.true_positives[continuous_id] += 1

            false_negative_colors = set(target_areas) - target_segment_matched
            false_positive_colors = set(pred_areas) - pred_segment_matched

            false_negative_colors.discard(metric.void_color)
            false_positive_colors.discard(metric.void_color)

            for target_color in list(false_negative_colors):
                void_target_area = intersection_areas.get(
                    (metric.void_color, target_color), 0
                )
                if void_target_area / target_areas[target_color] > 0.5:
                    false_negative_colors.discard(target_color)

            crowd_by_cat_id = {}
            for false_negative_color in false_negative_colors:
                if is_crowds[i][false_negative_color[1]]:
                    crowd_by_cat_id[false_negative_color[0]] = false_negative_color[1]
                    continue

                continuous_id = metric.cat_id_to_continuous_id[false_negative_color[0]]
                metric.false_negatives[continuous_id] += 1

            for pred_color in list(false_positive_colors):
                pred_void_crowd_area = intersection_areas.get(
                    (pred_color, metric.void_color), 0
                )

                if pred_color[0] in crowd_by_cat_id:
                    crowd_color = (pred_color[0], crowd_by_cat_id[pred_color[0]])
                    pred_void_crowd_area += intersection_areas.get(
                        (pred_color, crowd_color), 0
                    )

                if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
                    false_positive_colors.discard(pred_color)

            for false_positive_color in false_positive_colors:
                continuous_id = metric.cat_id_to_continuous_id[false_positive_color[0]]
                metric.false_positives[continuous_id] += 1

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_semantic(self, log_prefix, log_per_class=False):
        """Aggregates and logs Semantic Segmentation metrics (mIoU)."""
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx, iou in enumerate(iou_per_class):
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                        iou,
                    )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                iou_all,
            )

    def _on_eval_epoch_end_instance(self, log_prefix):
        """Aggregates and logs Instance Segmentation metrics (mAP)."""
        for i, metric in enumerate(self.metrics):  # type: ignore
            results = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            self.log(
                f"metrics/{log_prefix}_ap_all{block_postfix}",
                results["map"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_small_all{block_postfix}",
                results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_medium_all{block_postfix}",
                results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_large_all{block_postfix}",
                results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_50_all{block_postfix}",
                results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_75_all{block_postfix}",
                results["map_75"],
            )

    def _on_eval_epoch_end_panoptic(self, log_prefix, log_per_class=False):
        """Aggregates and logs Panoptic Quality metrics."""
        for i, metric in enumerate(self.metrics):  # type: ignore
            result = metric.compute()[:-1]
            metric.reset()

            pq, sq, rq = result[:, 0], result[:, 1], result[:, 2]

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx in range(len(pq)):
                    self.log(
                        f"metrics/{log_prefix}_pq_class_{class_idx}{block_postfix}",
                        pq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_sq_class_{class_idx}{block_postfix}",
                        sq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_rq_class_{class_idx}{block_postfix}",
                        rq[class_idx],
                    )

            self.log(
                f"metrics/{log_prefix}_pq_all{block_postfix}",
                pq.mean(),
            )
            self.log(f"metrics/{log_prefix}_sq_all{block_postfix}", sq.mean())
            self.log(f"metrics/{log_prefix}_rq_all{block_postfix}", rq.mean())

            num_things = len(metric.things)
            pq_things, sq_things, rq_things = (
                result[:num_things, 0],
                result[:num_things, 1],
                result[:num_things, 2],
            )
            pq_stuff, sq_stuff, rq_stuff = (
                result[num_things:, 0],
                result[num_things:, 1],
                result[num_things:, 2],
            )

            self.log(
                f"metrics/{log_prefix}_pq_things{block_postfix}",
                pq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_things{block_postfix}",
                sq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_things{block_postfix}",
                rq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_pq_stuff{block_postfix}",
                pq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_stuff{block_postfix}",
                sq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_stuff{block_postfix}",
                rq_stuff.mean(),
            )

    def _on_eval_end_semantic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_instance(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_all'] * 100:.1f} | "
                f"mAP Small: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_small_all'] * 100:.1f} | "
                f"mAP Medium: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_medium_all'] * 100:.1f} | "
                f"mAP Large: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_large_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_panoptic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}PQ All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_all'] * 100:.1f} | "
                f"PQ Things: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_things'] * 100:.1f} | "
                f"PQ Stuff: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_stuff'] * 100:.1f}{reset}"
            )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
    ):
        """
        Visualizes the segmentation results (Input, Ground Truth, Prediction)
        and logs them to Weights & Biases (WandB).
        """
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        # Plot Input Image
        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        # Handle Ignore Index (usually paint it black)
        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        # Plot Ground Truth
        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        # Plot Prediction
        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        # Create Legend
        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        # Save to buffer and log
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        """Helper to compute scaling factor ensuring image fits within self.img_size."""
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        """
        Splits a large image into overlapping crops (windows) for inference.
        Necessary when image resolution > model input resolution.
        """
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            # Resize image to target scale
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            # Calculate number of crops
            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            # Create crops
            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        """
        Recombines predictions from overlapping crops into the original image space.
        Uses a 'count' buffer to average predictions in overlapping regions.
        """
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        """
        Converts Mask2Former outputs (mask logits + class logits) into standard
        per-pixel semantic segmentation maps via Einstein summation.
        """
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        """Converts mask-based targets into per-pixel semantic labels."""
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]):
        factor = min(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        transformed_imgs = []

        for img in imgs:
            new_h, new_w = self.scale_img_size_instance_panoptic(img.shape[-2:])

            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
            )

            pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
            pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
            padding = [0, 0, pad_w, pad_h]

            padded_img = pad(resized_img, padding)

            transformed_imgs.append(padded_img)

        return torch.stack(transformed_imgs)

    @torch.compiler.disable
    def revert_resize_and_pad_logits_instance_panoptic(
        self, transformed_logits, img_sizes
    ):
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = interpolate(
                logits_i[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            logits.append(logits_i)

        return logits

    def to_per_pixel_preds_panoptic(
        self, mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh
    ):
        """Converts model outputs to Panoptic Segmentation format."""
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        preds_list = []

        for i in range(len(mask_logits_list)):
            preds = -torch.ones(
                (*mask_logits_list[i].shape[-2:], 2),
                dtype=torch.long,
                device=class_logits.device,
            )
            preds[:, :, 0] = self.num_classes

            keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
            if not keep.any():
                preds_list.append(preds)
                continue

            masks = mask_logits_list[i].sigmoid()
            segments = -torch.ones(
                *masks.shape[-2:],
                dtype=torch.long,
                device=class_logits.device,
            )

            mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
            stuff_segment_ids, segment_id = {}, 0
            segment_and_class_ids = []

            for k, class_id in enumerate(classes[i][keep].tolist()):
                orig_mask = masks[keep][k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < overlap_thresh
                ):
                    continue

                if class_id in stuff_classes:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    else:
                        stuff_segment_ids[class_id] = segment_id

                segments[final_mask] = segment_id
                segment_and_class_ids.append((segment_id, class_id))

                segment_id += 1

            for segment_id, class_id in segment_and_class_ids:
                segment_mask = segments == segment_id
                preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
                preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

            preds_list.append(preds)

        return preds_list

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_panoptic(targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = -torch.ones(
                (*target["masks"].shape[-2:], 2),
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[:, :, 0] = torch.where(
                    mask, target["labels"][i], per_pixel_target[:, :, 0]
                )

                per_pixel_target[:, :, 1] = torch.where(
                    mask,
                    torch.tensor(i, device=target["masks"].device),
                    per_pixel_target[:, :, 1],
                )

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def _zero_init_outside_encoder(
        self, encoder_prefix="network.encoder.", skip_class_head=False
    ):
        """Zero-initializes weights outside the backbone (typically for delta training)."""
        with torch.no_grad():
            total, zeroed = 0, 0
            for name, p in self.named_parameters():
                total += p.numel()
                if not name.startswith(encoder_prefix):
                    if skip_class_head and (
                        "class_head" in name or "class_predictor" in name
                    ):
                        continue
                    p.zero_()
                    zeroed += p.numel()
            msg = f"Zeroed {zeroed:,} / {total:,} parameters (everything not under '{encoder_prefix}'"
            if skip_class_head:
                msg += ", skipping class head"
            msg += ")"
            logging.info(msg)

    def _add_state_dicts(self, state_dict1, state_dict2):
        summed = {}
        for k in state_dict1.keys():
            if k not in state_dict2:
                raise KeyError(f"Key {k} not found in second state_dict")

            if state_dict1[k].shape != state_dict2[k].shape:
                raise ValueError(
                    f"Shape mismatch at {k}: "
                    f"{state_dict1[k].shape} vs {state_dict2[k].shape}"
                )

            summed[k] = state_dict1[k] + state_dict2[k]

        return summed

    def _load_ckpt(self, ckpt_path, load_ckpt_class_head):
        """
        Loads a checkpoint with advanced handling for:
        1. Complex float conversion.
        2. Resolution mismatch (Position Embedding interpolation).
        3. Class head filtering.
        """

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # --- CONVERT TO REAL FLOAT32 ---
        # Handles complex-valued tensors from FFTs in specific pre-trained models
        ckpt = {
            k: v.real.float() if torch.is_complex(v) else v.float()
            for k, v in ckpt.items()
        }
        # --------------------------------

        # --- INTERPOLATE POS_EMBED IF SIZE DIFFERS ---
        # Critical when fine-tuning at a resolution different from pre-training.
        # We perform bicubic interpolation on the positional embedding grid.
        pos_embed_key = "network.encoder.backbone.pos_embed"
        if pos_embed_key in ckpt:
            ckpt_pos_embed = ckpt[pos_embed_key]
            current_pos_embed = self.state_dict().get(pos_embed_key)
            if (
                current_pos_embed is not None
                and ckpt_pos_embed.shape != current_pos_embed.shape
            ):
                logging.info(
                    f"Interpolating pos_embed from {ckpt_pos_embed.shape} to {current_pos_embed.shape}"
                )
                # pos_embed shape: [1, num_patches, embed_dim]
                ckpt_num_patches = ckpt_pos_embed.shape[1]
                current_num_patches = current_pos_embed.shape[1]
                embed_dim = ckpt_pos_embed.shape[2]

                # Calculate grid sizes (assume square grid)
                ckpt_grid_size = int(ckpt_num_patches**0.5)
                current_grid_size = int(current_num_patches**0.5)

                # Reshape to 2D grid for spatial interpolation
                ckpt_pos_embed = ckpt_pos_embed.reshape(
                    1, ckpt_grid_size, ckpt_grid_size, embed_dim
                )
                ckpt_pos_embed = ckpt_pos_embed.permute(
                    0, 3, 1, 2
                )  # [1, embed_dim, H, W] for interpolation

                # Bicubic interpolation to new grid size
                interpolated = torch.nn.functional.interpolate(
                    ckpt_pos_embed,
                    size=(current_grid_size, current_grid_size),
                    mode="bicubic",
                    align_corners=False,
                )

                # Reshape back to [1, num_patches, embed_dim]
                interpolated = interpolated.permute(0, 2, 3, 1)  # [1, H, W, embed_dim]
                interpolated = interpolated.reshape(1, current_num_patches, embed_dim)

                ckpt[pos_embed_key] = interpolated
                logging.info(f"pos_embed interpolated successfully")
        # ----------------------------------------------

        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
        if not load_ckpt_class_head:
            ckpt = {
                k: v
                for k, v in ckpt.items()
                if "class_head" not in k and "class_predictor" not in k
            }
        logging.info(f"Loaded {len(ckpt)} keys")
        return ckpt

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        """Helper to check strict loading compliance."""
        if incompatible_keys.missing_keys:
            if not load_ckpt_class_head:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if (
                        "class_head" not in key
                        and "class_predictor" not in key
                        and "lora_a" not in key
                        and "lora_b" not in key
                        and "criterion" not in key
                    )
                ]
            else:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if "lora_a" not in key
                    and "lora_b" not in key
                    and "criterion" not in key
                ]
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
        if incompatible_keys.unexpected_keys:
            raise ValueError(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
