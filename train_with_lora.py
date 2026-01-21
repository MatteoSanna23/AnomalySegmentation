#!/usr/bin/env python3
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Training script with LoRA support
# Usage: python train_with_lora.py --use_lora --lora_rank 16 --lora_alpha 32
# ---------------------------------------------------------------

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# Aggiungi path
sys.path.insert(0, str(Path(__file__).parent))

from eomt.training.mask_classification_semantic import MaskClassificationSemantic
from eomt.datasets.lightning_data_module import LightningDataModule
from eomt.models.eomt import EoMT

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_argument_parser():
    """Crea parser per argomenti da linea di comando"""
    parser = argparse.ArgumentParser(
        description="Train EoMT model with optional LoRA fine-tuning"
    )
    
    # --- Parametri Dataset ---
    parser.add_argument("--dataset_root", type=str, default="/teamspace/studios/this_studio/Cityscapes",
                        help="Path to Cityscapes dataset")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512],
                        help="Input image size (height width)")
    
    # --- Parametri Modello ---
    parser.add_argument("--ckpt_path", type=str, default="/teamspace/studios/this_studio/epoch_106-step_19902_eomt.ckpt",
                        help="Path to pretrained checkpoint")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of semantic classes (Cityscapes: 19)")
    
    # --- Parametri Training ---
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for optimizer")
    
    # --- LoRA Specific ---
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r parameter)")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    
    # --- Logging & Checkpointing ---
    parser.add_argument("--output_dir", type=str, default="/teamspace/studios/this_studio/outputs",
                        help="Directory for saving checkpoints and logs")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="anomaly-segmentation",
                        help="W&B project name")
    
    return parser


def main(args):
    """Main training function"""
    
    # --- Setup ---
    logger.info("="*80)
    logger.info("ANOMALY SEGMENTATION - TRAINING WITH LoRA")
    logger.info("="*80)
    
    # Crea directory di output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    log_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    
    # --- Logger Setup ---
    loggers = [TensorBoardLogger(str(log_dir))]
    
    if args.use_wandb:
        loggers.append(WandbLogger(
            project=args.wandb_project,
            name=f"lora_rank{args.lora_rank}" if args.use_lora else "baseline",
            log_model=True
        ))
    
    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch-{epoch:02d}-val_loss-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]
    
    # --- Data Module ---
    logger.info(f"Loading Cityscapes from {args.dataset_root}")
    data_module = LightningDataModule(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
    )
    
    # --- Model Setup ---
    # NOTA: In produzione, caricheresti il vero modello EoMT da config
    # Qui è un template che mostra come passare i parametri LoRA
    logger.info("Creating EoMT model...")
    
    # Carica checkpoint preaddestrato
    # network = EoMT(...)  # Inizializza con encoder
    # Per adesso, assumiamo che il modello sia caricato da checkpoint
    
    # --- Training Module con LoRA ---
    logger.info("Creating training module with LoRA..." if args.use_lora else "Creating training module...")
    
    training_module = MaskClassificationSemantic(
        network=None,  # Placeholder - sarà caricato da checkpoint
        img_size=tuple(args.img_size),
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=True,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ckpt_path=args.ckpt_path,
        # --- LoRA Parameters ---
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # --- Trainer Setup ---
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
        val_check_interval=0.5,  # Valida ogni 50% dell'epoch
    )
    
    # --- Training ---
    logger.info("Starting training...")
    trainer.fit(training_module, data_module)
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Checkpoints saved to {ckpt_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    logger.info(f"Configuration: {vars(args)}")
    
    main(args)
