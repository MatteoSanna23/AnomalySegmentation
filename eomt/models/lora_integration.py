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
    """Configuration for LoRA adaptation."""

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
            "qkv", "proj", "fc1", "fc2"
        ]
        self.freeze_base_model = freeze_base_model

    def to_dict(self) -> Dict[str, Any]:
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
    """Applica LoRA cercando i moduli target in tutto il modello."""

    if not config.enabled:
        return

    print(f"Applying LoRA to targets: {config.target_modules}")
    applied_count = 0

    # Iteriamo su TUTTI i moduli del network per trovare quelli che matchano i target
    for name, module in model.named_modules():
        # Controlliamo i figli diretti di questo modulo
        for child_name, child in module.named_children():
            if child_name in config.target_modules and isinstance(child, nn.Linear):
                # Trovato un target! (es. è un Linear e si chiama 'qkv')
                
                # Creiamo il layer LoRA
                lora_linear = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=config.rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                )

                # Copiamo i pesi originali e li congeliamo
                lora_linear.weight.data.copy_(child.weight.data)
                lora_linear.weight.requires_grad = False

                if child.bias is not None and lora_linear.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)
                    lora_linear.bias.requires_grad = False

                # Sostituiamo il layer originale con quello LoRA
                setattr(module, child_name, lora_linear)
                applied_count += 1
                # print(f"  -> LoRA injected in: {name}.{child_name}")

    if applied_count == 0:
        print("WARNING: Nessun modulo LoRA è stato applicato! Verifica 'target_modules' nel config.")
    else:
        print(f"Successfully applied LoRA to {applied_count} layers.")

    # Freeze base model if requested
    if config.freeze_base_model:
        _freeze_base_model(model)


def _freeze_base_model(model: nn.Module) -> None:
    """
    Congela il modello base ma MANTIENE SBLOCCATE:
    1. Le parti LoRA
    2. Le teste di classificazione ('head')
    3. I layer di normalizzazione ('norm') - Opzionale ma consigliato
    """
    modules_to_keep = ["head", "norm"] 

    print(f"Freezing base model. Keeping trainable: LoRA layers + {modules_to_keep}")

    for name, param in model.named_parameters():
        # 1. Se è LoRA -> Trainable
        if "lora_" in name:
            param.requires_grad = True
        
        # 2. Se è una testa o norm -> Trainable
        elif any(k in name for k in modules_to_keep):
            param.requires_grad = True
            
        # 3. Tutto il resto -> Frozen
        else:
            param.requires_grad = False

def get_lora_stats(model: nn.Module) -> Dict[str, Any]:
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