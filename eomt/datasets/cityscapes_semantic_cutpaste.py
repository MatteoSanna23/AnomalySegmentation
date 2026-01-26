# Cityscapes Semantic Dataset with Cut-Paste Augmentation
# ------------------------------------------------------
# This module defines a Lightning DataModule for semantic segmentation training
# with synthetic anomaly augmentation (Cut-Paste strategy, now offline only).
#
# Key features:
# - Extends the standard Cityscapes dataset for semantic segmentation
# - Supports anomaly class (label ID 254) in segmentation masks
# - Designed for use with EoMT training pipeline
# - Uses pre-generated (offline) augmented data, no runtime augmentation


# Standard library imports
from pathlib import Path  # For filesystem path manipulations
from typing import Union  # For type hinting

# PyTorch and torchvision imports
import torch  # PyTorch core
from torch.utils.data import (
    DataLoader,
    Dataset as TorchDataset,
)  # Data loading utilities
from torchvision.datasets import (
    Cityscapes,
)  # Cityscapes dataset class (not directly used)
from torchvision import tv_tensors  # For mask tensor types
from torchvision.transforms.v2 import functional as F  # Image transform functions
from PIL import Image  # For image loading

# Local project imports
from datasets.lightning_data_module import (
    LightningDataModule,
)  # Base Lightning DataModule
from datasets.dataset import Dataset  # Custom dataset wrapper
from datasets.transforms import Transforms  # Custom transform pipeline


# =============================================================================
# LIGHTNING DATA MODULE
# =============================================================================


class CityscapesSemanticCutPaste(
    LightningDataModule
):  # Main DataModule for Cityscapes with anomaly support
    """
    Lightning DataModule for Cityscapes with Cut-Paste anomaly augmentation.

    This class wraps the standard Cityscapes dataset and adds the ability to
    paste Out-of-Distribution (OOD) objects from COCO onto training images,
    creating synthetic anomalies for anomaly segmentation training.

    The validation set remains unchanged (no cut-paste) to evaluate
    standard segmentation performance on clean images.
    """

    def __init__(
        self,
        path: str,  # Path to the pre-generated (offline) Cityscapes dataset
        original_cityscapes_path: str = None,  # Path to original zips for validation (optional)
        num_workers: int = 4,  # Number of DataLoader workers
        batch_size: int = 16,  # Batch size for training
        img_size: tuple[int, int] = (1024, 1024),  # Target image size (H, W)
        num_classes: int = 20,  # Number of classes (19 Cityscapes + 1 anomaly)
        color_jitter_enabled: bool = True,  # Enable color jitter augmentation
        scale_range: tuple = (0.5, 2.0),  # Scale range for random resize
        check_empty_targets: bool = True,  # Skip samples with empty masks
    ) -> None:
        """
        Initialize the Cityscapes Cut-Paste DataModule.

        Args:
            path: Path to Cityscapes dataset (folder with zip files).
            coco_ood_path: Path to downloaded COCO OOD subset.
            num_workers: Number of DataLoader workers.
            batch_size: Training batch size.
            img_size: Target image size (H, W) for training.
            num_classes: Number of classes (19 Cityscapes + 1 anomaly = 20).
            color_jitter_enabled: Enable color jitter augmentation.
            scale_range: Scale range for random resize augmentation.
            check_empty_targets: Check for and skip empty target masks.
            cutpaste_enabled: Enable cut-paste augmentation.
            cutpaste_probability: Probability of applying cut-paste (0-1).
            cutpaste_min_objects: Minimum OOD objects to paste per image.
            cutpaste_max_objects: Maximum OOD objects to paste per image.
            cutpaste_scale_range: Scale range for OOD object resizing.
            cutpaste_blend_mode: Blending mode ("paste", "alpha_feather", "poisson").
            cutpaste_feather_radius: Blur radius for feathered edges.
        """
        # Initialize parent LightningDataModule
        # Initialize parent LightningDataModule with basic dataset settings
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(
            ignore=["_class_path"]
        )  # Save hyperparameters for checkpointing

        # Initialize image transforms (color jitter, random scale, etc.)
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        target, anomaly_label_id: int = 254, **kwargs
    ):  # Parses mask into per-class binary masks
        """
        Parse segmentation mask into per-class masks, including anomaly class.

        This method extends the standard Cityscapes target parsing to handle
        the synthetic anomaly class (label ID 254) introduced by Cut-Paste.

        Args:
            target: Segmentation mask tensor [1, H, W] with label IDs.
            anomaly_label_id: Pixel value used to mark anomaly regions (default: 254).
            **kwargs: Additional arguments (unused, for compatibility).

        Returns:
            tuple: (masks, labels, is_crowd)
                - masks: List of binary masks, one per class present in image
                - labels: List of train_ids corresponding to each mask
                - is_crowd: List of False values (no crowd annotations)
        """
        masks, labels = [], []  # List to accumulate binary masks and their class labels

        # Iterate over all unique label IDs in the mask
        for label_id in target[0].unique():
            # If the label is the anomaly class (254), treat as anomaly
            if label_id == anomaly_label_id:
                masks.append(target[0] == label_id)  # Binary mask for anomaly pixels
                labels.append(
                    19
                )  # Assign train_id=19 for anomaly (Cityscapes uses 0-18)
                continue

            # Find the corresponding Cityscapes class for this label ID
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            # Skip unknown or ignored classes
            if cls is None or cls.ignore_in_eval:
                continue

            # Add binary mask and train_id for this class
            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        # Return masks, labels, and is_crowd (always False for Cityscapes)
        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Setup method: prepares datasets for training and validation
        train_transforms = self._create_train_transforms()  # Get training transforms

        # Common dataset keyword arguments
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",  # Image file extension
            "target_suffix": ".png",  # Mask file extension
            "img_stem_suffix": "leftImg8bit",  # Image filename stem
            "target_stem_suffix": "gtFine_labelIds",  # Mask filename stem
            "zip_path": Path(
                self.path, "leftImg8bit_trainvaltest.zip"
            ),  # Path to images zip
            "target_zip_path": Path(
                self.path, "gtFine_trainvaltest.zip"
            ),  # Path to masks zip
            "target_parser": lambda t, **kw: self.target_parser(
                t, anomaly_label_id=254, **kw
            ),  # Function to parse masks
            "check_empty_targets": self.check_empty_targets,  # Skip empty masks if True
        }

        # Training dataset: uses DatasetWithCutPaste (no runtime augmentation, just a wrapper)
        self.cityscapes_train_dataset = DatasetWithCutPaste(
            transforms=train_transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )

        # Validation dataset: standard Dataset (no augmentation)
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        return self  # Return self for chaining

    def _create_train_transforms(self):  # Returns the configured training transforms
        """
        Get transforms for training data.

        Returns:
            Transforms: Configured transform pipeline (color jitter, scale, crop).
        """
        return self.transforms

    def train_dataloader(self):  # Returns DataLoader for training set
        """
        Create DataLoader for training set.

        Returns:
            DataLoader: Shuffled training loader with cut-paste augmentation.
        """
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,  # Randomize sample order each epoch
            drop_last=True,  # Drop incomplete final batch for stable training
            collate_fn=self.train_collate,  # Custom collation for masks/labels
            **self.dataloader_kwargs,  # batch_size, num_workers, etc.
        )

    def val_dataloader(self):  # Returns DataLoader for validation set
        """
        Create DataLoader for validation set.

        Returns:
            DataLoader: Sequential validation loader (no augmentation).
        """
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,  # Evaluation-specific collation
            **self.dataloader_kwargs,
        )


# =============================================================================
# CUSTOM DATASET WITH CUT-PASTE AUGMENTATION
# =============================================================================


class DatasetWithCutPaste(Dataset):
    """
    Dataset wrapper for compatibility. No cut-paste augmentation applied.
    """

    def __init__(self, **kwargs):
        # Initialize parent Dataset with all arguments
        super().__init__(**kwargs)

    def __getitem__(self, index: int):
        # Loads a sample (image, mask), applies transforms, parses mask
        img, target = self._load_item(index)  # Load raw image and mask
        if self.transforms is not None:
            img, target = self.transforms(img, target)  # Apply transforms if present
        masks, labels, is_crowd = self.target_parser(
            target
        )  # Parse mask into binary masks and labels
        return img, masks, labels, is_crowd  # Return sample for training

    def _load_item(self, index: int):
        # Loads raw image and target mask from zip files (no transforms)
        import torch
        from PIL import Image
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as F

        # Get file info for this sample (image and mask)
        img_info = self.imgs[index]
        target_info = self.targets[index]

        # Load image from zip: open file, convert to RGB, to tensor, normalize
        with self._get_zip().open(img_info.filename) as img_file:
            img = Image.open(img_file).convert("RGB")  # Ensure RGB format
            img = F.to_image(img)  # Convert PIL image to tensor
            img = F.to_dtype(img, torch.float32, scale=True)  # Normalize to [0, 1]

        # Load target mask from zip: open file, convert to tensor mask
        target_zip = self._get_target_zip()
        with target_zip.open(target_info) as target_file:
            target = Image.open(target_file)  # Load as PIL image
            target = tv_tensors.Mask(F.pil_to_tensor(target))  # Convert to Mask tensor

        return img, target  # Return image and mask

    def _get_zip(self):
        # Returns handle to image zip file (lazy initialization, supports multi-worker)
        import zipfile
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()  # Check if running in multi-worker mode
        if worker_info is None or self.zip is None:
            if self.zip is None:
                self.zip = zipfile.ZipFile(
                    self.zip_path, "r"
                )  # Open zip if not already open
        return self.zip

    def _get_target_zip(self):
        # Returns handle to target/mask zip file (lazy initialization)
        import zipfile
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        if worker_info is None or self.target_zip is None:
            if self.target_zip is None:
                # Use dedicated target zip or fall back to image zip
                target_path = self.target_zip_path or self.zip_path
                self.target_zip = zipfile.ZipFile(target_path, "r")
        return self.target_zip
