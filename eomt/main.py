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

# Suppress PyTorch FX warnings for DINOv3/ViT models to reduce console noise
os.environ["TORCH_LOGS"] = "-dynamo"

# --- MONKEY PATCHING SECTION ---
# These overrides are necessary to fix specific type-hinting issues
# in the underlying library (jsonargparse) that can cause crashes with complex configurations.
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


# --- CUSTOM VALIDATION LOGIC ---
# This function modifies how Lightning decides *when* to run validation.
# It is specifically patched to handle "Infinite Datasets" (iterable datasets)
# where the concept of an "epoch" is defined by steps rather than dataset size.
def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch

    # Check validation at the end of the data stream
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    # Check validation if early stopping conditions are met
    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    # Standard periodic validation check
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
            # Check based on global steps if epoch-based checking is disabled
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    """
    The Command Line Interface manager.
    It automatically parses the YAML config file and instantiates the Model and Data modules.
    """

    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        # Optimize matrix multiplication for modern GPUs (e.g., A100/H100)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings("ignore")
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        """
        Links parameters between Data and Model sections in the YAML.
        Example: If 'num_classes' is set in Data, it automatically updates Model config.
        """
        parser.add_argument("--compile_disabled", action="store_true")
        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )
        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )
        # Note: ckpt_path linking is disabled to allow manual control in fit()

    def fit(self, model, **kwargs):
        """
        Main entry point for training.
        """
        # Handle logging of source code to WandB (ignoring gitignored files)
        gitignore_path = ".gitignore"
        if os.path.exists(gitignore_path):
            is_gitignored = parse_gitignore(gitignore_path)
        else:

            def is_gitignored(path):
                return False

        if hasattr(self.trainer.logger.experiment, "log_code"):
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        # Apply the validation loop patch defined above
        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        # Torch Compile for speedup (unless disabled via flag)
        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)


def cli_main():
    """
    Initializes the CLI with default trainer settings.
    """
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",  # Use Mixed Precision (FP16/FP32)
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "devices": 1,
            # Gradient Clipping prevents exploding gradients during training
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
