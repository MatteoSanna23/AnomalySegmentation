#!/usr/bin/env python3
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Example: Using LoRA with EoMT model
# ---------------------------------------------------------------

"""
Example script showing how to use LoRA with the EoMT model.

This demonstrates:
1. Creating a LoRA configuration
2. Applying LoRA to the model
3. Fine-tuning with LoRA
4. Getting statistics about LoRA parameters
"""

import torch
import torch.nn as nn
from models.lora_integration import LoRAConfig, apply_lora_to_vit, print_lora_summary
from models.lora import count_lora_parameters


def create_dummy_lora_config() -> LoRAConfig:
    """Create a LoRA configuration for head-based adaptation.

    This configuration applies LoRA only to the class_head and mask_head
    of the EoMT model, freezing the backbone encoder.
    """
    return LoRAConfig(
        enabled=True,
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["class_head", "mask_head"],
        freeze_base_model=True,
    )


def create_lora_config_full_model() -> LoRAConfig:
    """Create a LoRA configuration for full model adaptation.

    This configuration applies LoRA more broadly across the model.
    """
    return LoRAConfig(
        enabled=True,
        rank=16,
        lora_alpha=32,
        lora_dropout=0.15,
        target_modules=[
            "class_head",
            "mask_head",
            "upscale",
        ],
        freeze_base_model=True,
    )


def example_basic_usage(model: nn.Module) -> None:
    """Example: Basic LoRA usage.

    Args:
        model: An EoMT model instance
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic LoRA Usage")
    print("=" * 70)

    # Create LoRA configuration
    lora_config = create_dummy_lora_config()

    # Apply LoRA to model
    apply_lora_to_vit(model, lora_config)

    # Print summary
    print_lora_summary(model, lora_config)

    # Get parameter counts
    lora_stats = count_lora_parameters(model)
    print(f"LoRA Parameters: {lora_stats['lora']:,}")
    print(f"Non-LoRA Trainable Parameters: {lora_stats['non_lora']:,}")
    print(f"Total Trainable: {lora_stats['total_trainable']:,}")


def example_with_config_file(model: nn.Module) -> None:
    """Example: Using LoRA with configuration from file/dict.

    Args:
        model: An EoMT model instance
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: LoRA from Configuration Dictionary")
    print("=" * 70)

    # Configuration as dictionary (useful for YAML configs)
    lora_dict = {
        "enabled": True,
        "rank": 12,
        "lora_alpha": 24,
        "lora_dropout": 0.1,
        "target_modules": ["class_head", "mask_head"],
        "freeze_base_model": True,
    }

    # Create config from dictionary
    lora_config = LoRAConfig(**lora_dict)

    # Apply to model
    apply_lora_to_vit(model, lora_config)

    # Print summary
    print_lora_summary(model, lora_config)


def example_parameter_comparison(model: nn.Module) -> None:
    """Example: Compare parameters before and after LoRA.

    Args:
        model: An EoMT model instance
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Parameter Comparison")
    print("=" * 70)

    # Count original parameters
    original_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    original_total = sum(p.numel() for p in model.parameters())

    print(f"Original Trainable Parameters: {original_trainable:,}")
    print(f"Original Total Parameters: {original_total:,}")

    # Apply LoRA
    lora_config = create_lora_config_full_model()
    apply_lora_to_vit(model, lora_config)

    # Count with LoRA
    lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in model.parameters())

    print(f"\nWith LoRA Trainable Parameters: {lora_trainable:,}")
    print(f"With LoRA Total Parameters: {lora_total:,}")
    print(f"\nAdded Parameters: {lora_trainable - original_trainable:,}")
    print(f"Parameter Reduction: {(1 - lora_trainable/original_total)*100:.1f}%")


def example_training_loop(model: nn.Module, num_steps: int = 5) -> None:
    """Example: Simple training loop with LoRA.

    Args:
        model: An EoMT model instance
        num_steps: Number of training steps
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Training Loop with LoRA")
    print("=" * 70)

    # Apply LoRA
    lora_config = create_dummy_lora_config()
    apply_lora_to_vit(model, lora_config)

    # Verify only LoRA parameters are trainable
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")

    # Create optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    model.train()
    dummy_loss = 0.0

    print(f"\nRunning {num_steps} training steps...")
    for step in range(num_steps):
        # Dummy forward pass
        for param in model.parameters():
            if param.requires_grad:
                dummy_loss = param.sum() * 0.0001

        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

        if (step + 1) % 2 == 0:
            print(f"  Step {step + 1}/{num_steps}, Loss: {dummy_loss:.6f}")

    print("Training completed!")


def example_merge_lora_weights(model: nn.Module) -> None:
    """Example: Merge LoRA weights back into original weights.

    This shows how to combine the base model with LoRA adaptations.

    Args:
        model: An EoMT model instance
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Merging LoRA Weights")
    print("=" * 70)

    # Apply LoRA
    lora_config = create_dummy_lora_config()
    apply_lora_to_vit(model, lora_config)

    print("LoRA applied. Now showing how to merge weights...")
    print("\nFor LoRA Linear layers:")
    print("  original_weight = weight + (alpha/r) * B @ A")

    # Example of how to merge (implementation depends on your use case)
    lora_count = sum(1 for n, _ in model.named_modules() if "LoRA" in str(type(_)))
    print(f"\nTotal LoRA modules in model: {lora_count}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LoRA Integration Examples")
    print("=" * 70)

    # This is a demonstration script
    # In practice, you would:
    # 1. Load your actual model
    # 2. Apply LoRA configuration from your training config
    # 3. Fine-tune on your specific task

    print("\nTo use in your training:")
    print("1. Create a LoRAConfig in your config file")
    print("2. Pass it to EoMT during initialization")
    print("3. Train normally - only LoRA parameters will be updated")
    print("\nExample:")
    print(
        """
    lora_config = LoRAConfig(
        enabled=True,
        rank=8,
        lora_alpha=16,
        target_modules=["class_head", "mask_head"],
        freeze_base_model=True,
    )
    
    model = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=num_q,
        lora_config=lora_config
    )
    """
    )

    print("\n" + "=" * 70)
