# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# LoRA (Low-Rank Adaptation) configuration and utilities
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)


def apply_lora_to_model(
    model: nn.Module,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Applica LoRA al modello EoMT.
    
    LoRA aggiunge matrici low-rank (A, B) ai layer di attenzione del ViT.
    Invece di fine-tuning completo, insegniamo solo adattamenti a basso rango.
    
    Args:
        model: Modello EoMT da adattare
        lora_rank: Dimensione della decomposizione low-rank (default: 16)
        lora_alpha: Scaling factor (default: 32)
        lora_dropout: Dropout rate per i layer LoRA (default: 0.05)
        target_modules: Lista di moduli da applicare LoRA (default: Attention e projection)
    
    Returns:
        Model with LoRA applied (via PEFT)
    """
    
    if target_modules is None:
        # Target i layer di attenzione del ViT (dove conta più il fine-tuning)
        target_modules = ["qkv", "proj", "q_proj", "v_proj"]
    
    logger.info(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
    logger.info(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",  # Generic type, non critical
        modules_to_save=None,  # Niente moduli extra da salvare
    )
    
    model_with_lora = get_peft_model(model, lora_config)
    
    # Stampa statistiche
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    
    logger.info(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model_with_lora


def freeze_encoder_except_lora(model: nn.Module) -> None:
    """
    Assicura che l'encoder sia congelato TRANNE i moduli LoRA.
    PEFT fa questo automaticamente, ma questa funzione è un doppio check.
    
    Args:
        model: Modello con LoRA applicato
    """
    # Se il modello ha LoRA, la libreria PEFT già congela tutto tranne LoRA
    if hasattr(model, "peft_config"):
        logger.info("Model has PEFT LoRA - encoder is already frozen except LoRA modules")
        return
    
    # Doppio check: assicuriamo che tutti i parametri non-LoRA siano congelati
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Stampa un report dettagliato dei parametri allenabili vs freezati.
    """
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    print("\n" + "="*80)
    print("TRAINABLE PARAMETERS (LoRA)")
    print("="*80)
    total_trainable = 0
    for name, count in sorted(trainable_params):
        print(f"  {name}: {count:,}")
        total_trainable += count
    
    print(f"\nTotal trainable: {total_trainable:,}")
    print("\n" + "="*80)
    print("FROZEN PARAMETERS (Pre-trained weights)")
    print("="*80)
    total_frozen = sum(c for _, c in frozen_params)
    print(f"Total frozen: {total_frozen:,}")
    
    print("\n" + "="*80)
    print(f"SUMMARY: {100*total_trainable/(total_trainable+total_frozen):.2f}% trainable")
    print("="*80 + "\n")
