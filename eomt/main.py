# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import jsonargparse._typehints as _t
from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
import os
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
os.environ["TORCH_LOGS"] = "-dynamo"

_orig_single = _t.raise_unexpected_value

def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)

_orig_union = _t.raise_union_unexpected_value

def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)

_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union

def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch

class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings("ignore")
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")
        parser.link_arguments("data.init_args.num_classes", "model.init_args.num_classes")
        parser.link_arguments("data.init_args.num_classes", "model.init_args.network.init_args.num_classes")
        parser.link_arguments("data.init_args.stuff_classes", "model.init_args.stuff_classes")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.encoder.init_args.img_size")
        # Rimosso il link automatico per ckpt_path per evitare conflitti nel caricamento manuale
        # parser.link_arguments("model.init_args.ckpt_path", "model.init_args.network.init_args.encoder.init_args.ckpt_path")

    def fit(self, model, **kwargs):
        # ------------------------------------------------------------------
        # FIX PER IL FINE-TUNING + RESTORE EPOCH
        # ------------------------------------------------------------------
        ckpt_path = kwargs.get('ckpt_path')
        
        # Se stiamo usando il nostro checkpoint pulito (senza ottimizzatore)
        if ckpt_path and "clean.ckpt" in str(ckpt_path):
            print(f"\n🔄 MODE: FINE-TUNING (Detected 'clean.ckpt')")
            print(f"   Caricamento pesi manuale da: {ckpt_path}")
            
            # 1. Carichiamo il file
            checkpoint = torch.load(ckpt_path, map_location=model.device)
            
            # 2. Iniettiamo i pesi nel modello (strict=False permette flessibilità per LoRA)
            keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"   ✅ Pesi caricati! (Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)})")
            
            # 3. Importante: Rimuoviamo ckpt_path dai kwargs
            #    Questo impedisce al Trainer di provare a fare il "Resume" automatico
            #    e permette di inizializzare un nuovo ottimizzatore fresco.
            kwargs['ckpt_path'] = None

            # 4. TRUCCO PER L'EPOCA: Impostiamo manualmente l'inizio
            #    Diciamo a Lightning che 106 epoche sono già state completate.
            #    Così il contatore partirà da 107 (Epoch 106 terminata -> Inizio Epoch 107).
            self.trainer.fit_loop.epoch_progress.current.completed = 106
            
            # (Opzionale) Ripristina lo step globale se vuoi continuità nei grafici
            # self.trainer.global_step = 19902 
            
            print(f"   ✅ Epoca forzata a: {self.trainer.current_epoch} (Start Epoch 107)")
        # ------------------------------------------------------------------

        gitignore_path = ".gitignore"
        if os.path.exists(gitignore_path):
            is_gitignored = parse_gitignore(gitignore_path)
        else:
            def is_gitignored(path):
                return False  # Nessun file viene escluso

        if hasattr(self.trainer.logger.experiment, "log_code"):
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)

def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )

if __name__ == "__main__":
    cli_main()