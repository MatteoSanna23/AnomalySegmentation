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
    """
    Configuration object for Low-Rank Adaptation (LoRA).

    LoRA allows fine-tuning large models by freezing the pre-trained weights
    and injecting trainable rank-decomposition matrices.
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
        """
        Args:
            enabled: Master switch to enable/disable LoRA.
            rank: The dimension of the low-rank matrices (r). Lower = fewer params.
            lora_alpha: Scaling factor for LoRA weights. Higher = stronger adaptation.
            lora_dropout: Dropout probability applied to LoRA inputs.
            target_modules: List of module names to replace (e.g., "qkv" in Attention).
            freeze_base_model: If True, freezes the backbone and trains only LoRA + Heads.
        """
        self.enabled = enabled
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Default targets for standard ViT architectures (Query-Key-Value and Projections)
        self.target_modules = target_modules or ["qkv", "proj", "fc1", "fc2"]
        self.freeze_base_model = freeze_base_model

    def to_dict(self) -> Dict[str, Any]:
        """Helper to serialize config for logging."""
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
    """
    Injects LoRA layers into the Vision Transformer architecture.

    It traverses the model, finds Linear layers matching `target_modules`,
    and replaces them with `LoRALinear` wrappers containing the low-rank matrices.
    """

    if not config.enabled:
        return

    print(f"Applying LoRA to targets: {config.target_modules}")
    applied_count = 0

    # Iterate over ALL modules in the network to find targets
    for name, module in model.named_modules():
        # Inspect direct children of the current module
        for child_name, child in module.named_children():
            # Check if the child matches a target name (e.g., 'qkv') and is a Linear layer
            if child_name in config.target_modules and isinstance(child, nn.Linear):

                # Create the LoRA wrapper layer
                lora_linear = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=config.rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                )

                # Copy original pre-trained weights to the new layer
                # We freeze these immediately because LoRA only trains the side-matrices
                lora_linear.weight.data.copy_(child.weight.data)
                lora_linear.weight.requires_grad = False

                if child.bias is not None and lora_linear.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)
                    lora_linear.bias.requires_grad = False

                # Replace the original layer with the LoRA-augmented one
                setattr(module, child_name, lora_linear)
                applied_count += 1
                # Debug: print(f"  -> LoRA injected in: {name}.{child_name}")

    if applied_count == 0:
        print("WARNING: No LoRA modules applied! Check 'target_modules' in config.")
    else:
        print(f"Successfully applied LoRA to {applied_count} layers.")

    # Freeze the main backbone weights if requested to save memory/compute
    if config.freeze_base_model:
        _freeze_base_model(model)


def _freeze_base_model(model: nn.Module) -> None:
    """
    Freezes the base model parameters but keeps specific parts trainable.

    We keep trainable:
    1. 'lora_': The new low-rank adapter matrices.
    2. 'head': The classification/segmentation head (must adapt to new classes).
    3. 'norm': Normalization layers (LayerNorm), often crucial for stability.
    """
    modules_to_keep = ["head", "norm"]

    print(f"Freezing base model. Keeping trainable: LoRA layers + {modules_to_keep}")

    for name, param in model.named_parameters():
        # 1. If it's a LoRA parameter -> Trainable
        if "lora_" in name:
            param.requires_grad = True

        # 2. If it's part of the output head or normalization -> Trainable
        elif any(k in name for k in modules_to_keep):
            param.requires_grad = True

        # 3. Everything else (backbone weights) -> Frozen
        else:
            param.requires_grad = False


def get_lora_stats(model: nn.Module) -> Dict[str, Any]:
    """Calculates parameter counts to measure LoRA efficiency."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count specific LoRA parameters
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
    """Prints a summary table of the model's parameter efficiency."""
    stats = get_lora_stats(model)
    print("\n" + "=" * 60)
    print("LoRA Configuration Summary")
    print("=" * 60)
    print(f"Enabled: {config.enabled}")
    print(f"Rank: {config.rank}")
    print(f"Target Modules: {config.target_modules}")
    print("-" * 60)
    print(f"Total Parameters: {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")
    print(f"LoRA Parameters: {stats['lora_params']:,}")
    print(f"Training Efficiency: {stats['efficiency']:.2f}%")
    print("=" * 60 + "\n")
