# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# LoRA integration utilities for Vision Transformers
# ---------------------------------------------------------------

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from models.lora import LoRALinear, replace_linear_with_lora, count_lora_parameters


class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        enabled: Whether to enable LoRA
        rank: Rank of LoRA matrices
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability
        target_modules: List of module patterns to apply LoRA to
        freeze_base_model: Whether to freeze base model weights
    """

    def __init__(
        self,
        enabled: bool = True,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list[str]] = None,
        freeze_base_model: bool = False,
    ):
        self.enabled = enabled
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "class_head",
            "mask_head",
        ]
        self.freeze_base_model = freeze_base_model

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "freeze_base_model": self.freeze_base_model,
        }


def apply_lora_to_vit(
    model: nn.Module,
    config: LoRAConfig,
) -> None:
    """Apply LoRA to Vision Transformer model.

    Args:
        model: The ViT or EoMT model to adapt
        config: LoRA configuration
    """

    if not config.enabled:
        return

    # Apply LoRA to specified modules
    for target_module in config.target_modules:
        parts = target_module.split(".")
        current = model

        # Navigate to the target module
        try:
            for part in parts[:-1]:
                current = getattr(current, part)

            target = getattr(current, parts[-1])

            # Apply LoRA to all Linear layers in the target
            _apply_lora_to_module(
                target,
                rank=config.rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
        except AttributeError:
            print(f"Warning: Could not find module {target_module}")

    # Freeze base model if requested
    if config.freeze_base_model:
        _freeze_base_model(model)


def _apply_lora_to_module(
    module: nn.Module,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> None:
    """Recursively apply LoRA to all Linear layers in a module."""

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with LoRA
            lora_linear = LoRALinear(
                child.in_features,
                child.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

            # Copy original weights and freeze them
            lora_linear.weight.data.copy_(child.weight.data)
            lora_linear.weight.requires_grad = False

            if child.bias is not None and lora_linear.bias is not None:
                lora_linear.bias.data.copy_(child.bias.data)
                lora_linear.bias.requires_grad = False

            setattr(module, name, lora_linear)
        else:
            # Recursively apply to children
            _apply_lora_to_module(
                child,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )


def _freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def get_lora_stats(model: nn.Module) -> Dict[str, Any]:
    """Get statistics about LoRA parameters in the model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with parameter statistics
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            lora_params += param.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_params": lora_params,
        "efficiency": (lora_params / total_params * 100) if total_params > 0 else 0,
    }


def print_lora_summary(model: nn.Module, config: LoRAConfig) -> None:
    """Print a summary of LoRA configuration and statistics.

    Args:
        model: The model with LoRA
        config: LoRA configuration
    """

    stats = get_lora_stats(model)

    print("\n" + "=" * 60)
    print("LoRA Configuration Summary")
    print("=" * 60)
    print(f"Enabled: {config.enabled}")
    print(f"Rank: {config.rank}")
    print(f"Alpha: {config.lora_alpha}")
    print(f"Dropout: {config.lora_dropout}")
    print(f"Target Modules: {config.target_modules}")
    print(f"Freeze Base Model: {config.freeze_base_model}")
    print("-" * 60)
    print(f"Total Parameters: {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")
    print(f"LoRA Parameters: {stats['lora_params']:,}")
    print(f"Training Efficiency: {stats['efficiency']:.2f}%")
    print("=" * 60 + "\n")
