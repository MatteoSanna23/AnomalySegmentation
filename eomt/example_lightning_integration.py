#!/usr/bin/env python3
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Example: Integrating LoRA with Lightning CLI
# ---------------------------------------------------------------

"""
Complete example showing how to integrate LoRA with PyTorch Lightning CLI.

Usage:
    # With LoRA enabled
    python main.py fit --config config_with_lora.yaml

    # Without LoRA
    python main.py fit --config config_baseline.yaml
"""

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.eomt import EoMT
from models.vit import ViT
from models.lora_integration import LoRAConfig, print_lora_summary


class EoMTLightningModule(LightningModule):
    """Lightning module with LoRA support.

    This module wraps EoMT and handles training/validation/testing loops
    while supporting LoRA configuration.

    Example YAML config:
        model:
          class_path: example_lightning_integration.EoMTLightningModule
          init_args:
            img_size: [640, 640]
            num_classes: 19
            num_q: 100
            lr: 0.0001
            enable_lora: true
            lora_rank: 8
            lora_alpha: 16
    """

    def __init__(
        self,
        img_size: tuple = (640, 640),
        num_classes: int = 19,
        num_q: int = 100,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        enable_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
        lora_freeze_base: bool = True,
    ):
        """Initialize the Lightning module.

        Args:
            img_size: Input image size
            num_classes: Number of semantic classes
            num_q: Number of learnable query tokens
            backbone_name: Vision Transformer backbone
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            enable_lora: Whether to enable LoRA
            lora_rank: Rank of LoRA matrices
            lora_alpha: Scaling factor for LoRA
            lora_dropout: Dropout in LoRA layers
            lora_target_modules: Modules to apply LoRA to
            lora_freeze_base: Whether to freeze base model
        """
        super().__init__()
        self.save_hyperparameters()

        # Build LoRA config if enabled
        lora_config = None
        if enable_lora:
            lora_target_modules = lora_target_modules or ["class_head", "mask_head"]
            lora_config = LoRAConfig(
                enabled=True,
                rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                freeze_base_model=lora_freeze_base,
            )

        # Build model
        vit_encoder = ViT(
            img_size=img_size,
            backbone_name=backbone_name,
        )

        self.model = EoMT(
            encoder=vit_encoder,
            num_classes=num_classes,
            num_q=num_q,
            lora_config=lora_config,
        )

        # Print LoRA summary if enabled
        if enable_lora:
            print_lora_summary(self.model, lora_config)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch  # Adjust based on your data loader

        mask_logits, class_logits = self(images)

        # Compute loss (example - adjust to your loss function)
        loss = self._compute_loss(mask_logits, class_logits, masks)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch

        with torch.no_grad():
            mask_logits, class_logits = self(images)

        loss = self._compute_loss(mask_logits, class_logits, masks)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        images, masks = batch

        with torch.no_grad():
            mask_logits, class_logits = self(images)

        loss = self._compute_loss(mask_logits, class_logits, masks)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Important: With LoRA and freeze_base_model=True, only LoRA parameters
        will be trainable. The optimizer will automatically only update those.
        """
        # Get only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        print(
            f"\nConfiguring optimizer for {len(trainable_params)} trainable parameters"
        )

        optimizer = AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _compute_loss(self, mask_logits, class_logits, masks):
        """Compute training loss.

        This is a placeholder - implement your actual loss function.
        """
        # Example: Simple MSE loss
        # Adjust this based on your task
        loss = torch.tensor(0.0, device=self.device)

        # Process multi-layer outputs
        for mask_pred in mask_logits:
            # Resize to match target
            mask_pred_resized = torch.nn.functional.interpolate(
                mask_pred,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = loss + torch.nn.functional.mse_loss(mask_pred_resized, masks)

        return loss / len(mask_logits)


# Example YAML configuration with LoRA
EXAMPLE_CONFIG_LORA = """
# config_with_lora.yaml

# Model with LoRA
model:
  class_path: example_lightning_integration.EoMTLightningModule
  init_args:
    img_size: [640, 640]
    num_classes: 19
    num_q: 100
    backbone_name: vit_large_patch14_reg4_dinov2
    lr: 0.0001
    weight_decay: 0.01
    enable_lora: true
    lora_rank: 8
    lora_alpha: 16
    lora_dropout: 0.1
    lora_target_modules: ["class_head", "mask_head"]
    lora_freeze_base: true

# Data
data:
  class_path: datasets.lightning_data_module.LightningDataModule
  init_args:
    # Your data config here
    batch_size: 4

# Trainer
trainer:
  max_epochs: 100
  devices: [0]  # GPU 0
  accelerator: gpu
  precision: 16-mixed  # Faster with mixed precision
  log_every_n_steps: 10
  val_check_interval: 0.25

# Logger
logger:
  class_path: lightning.pytorch.loggers.WandbLogger
  init_args:
    project: anomaly-segmentation
    name: eomt-with-lora
"""

# Example YAML configuration without LoRA (baseline)
EXAMPLE_CONFIG_BASELINE = """
# config_baseline.yaml

# Model without LoRA
model:
  class_path: example_lightning_integration.EoMTLightningModule
  init_args:
    img_size: [640, 640]
    num_classes: 19
    num_q: 100
    backbone_name: vit_large_patch14_reg4_dinov2
    lr: 0.0001
    weight_decay: 0.01
    enable_lora: false  # Disable LoRA

# Data
data:
  class_path: datasets.lightning_data_module.LightningDataModule
  init_args:
    batch_size: 1  # Smaller batch size for full training

# Trainer
trainer:
  max_epochs: 100
  devices: [0]
  accelerator: gpu
  precision: 16-mixed
  log_every_n_steps: 10

# Logger
logger:
  class_path: lightning.pytorch.loggers.WandbLogger
  init_args:
    project: anomaly-segmentation
    name: eomt-baseline
"""

if __name__ == "__main__":
    print("Lightning Integration Example")
    print("=" * 70)
    print(f"\nConfiguration with LoRA:\n{EXAMPLE_CONFIG_LORA}")
    print(f"\nConfiguration baseline (no LoRA):\n{EXAMPLE_CONFIG_BASELINE}")
    print("\nTo use, save as config_with_lora.yaml and run:")
    print("  python main.py fit --config config_with_lora.yaml")
