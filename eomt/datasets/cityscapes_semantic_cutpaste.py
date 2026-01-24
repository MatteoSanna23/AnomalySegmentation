"""
Cityscapes Semantic Dataset with Cut-Paste Augmentation
========================================================

This module provides a Lightning DataModule for training semantic segmentation
models with synthetic anomaly augmentation using the Cut-Paste strategy.

Key Features:
- Extends standard Cityscapes semantic segmentation dataset
- Integrates Cut-Paste augmentation to paste OOD objects from COCO
- Supports anomaly class (label ID 254) in segmentation masks
- Compatible with EoMT training pipeline

Architecture:
    CityscapesSemanticCutPaste (LightningDataModule)
        ├── DatasetWithCutPaste (training) - applies cut-paste before transforms
        └── Dataset (validation) - standard Cityscapes without augmentation

Usage:
    datamodule = CityscapesSemanticCutPaste(
        path="/path/to/cityscapes",
        coco_ood_path="/path/to/coco_ood",
        cutpaste_probability=0.5,
    )
"""

from pathlib import Path
from typing import Union, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms
from cutpaste import CutPasteAugmentation


# =============================================================================
# LIGHTNING DATA MODULE
# =============================================================================


class CityscapesSemanticCutPaste(LightningDataModule):
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
        path: str,
        coco_ood_path: str,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled: bool = True,
        scale_range: tuple = (0.5, 2.0),
        check_empty_targets: bool = True,
        # Cut-Paste params
        cutpaste_enabled: bool = True,
        cutpaste_probability: float = 0.5,
        cutpaste_min_objects: int = 1,
        cutpaste_max_objects: int = 3,
        cutpaste_scale_range: tuple = (0.5, 1.5),
        cutpaste_blend_mode: str = "alpha_feather",
        cutpaste_feather_radius: int = 5,
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
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])  # Save for checkpointing

        # Store cut-paste configuration
        self.coco_ood_path = coco_ood_path
        self.cutpaste_enabled = cutpaste_enabled
        self.cutpaste_probability = cutpaste_probability
        self.cutpaste_min_objects = cutpaste_min_objects
        self.cutpaste_max_objects = cutpaste_max_objects
        self.cutpaste_scale_range = cutpaste_scale_range
        self.cutpaste_blend_mode = cutpaste_blend_mode
        self.cutpaste_feather_radius = cutpaste_feather_radius

        # Initialize image transforms (color jitter, random scale, etc.)
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

        # Initialize Cut-Paste augmentation module
        self.cutpaste_aug = None
        if cutpaste_enabled and coco_ood_path:
            try:
                self.cutpaste_aug = CutPasteAugmentation(
                    coco_ood_path=coco_ood_path,
                    apply_probability=cutpaste_probability,
                    min_objects=cutpaste_min_objects,
                    max_objects=cutpaste_max_objects,
                    scale_range=cutpaste_scale_range,
                    blend_mode=cutpaste_blend_mode,
                    feather_radius=cutpaste_feather_radius,
                )
                print(f"Cut-Paste augmentation enabled (prob={cutpaste_probability})")
            except Exception as e:
                print(f"Warning: Could not initialize Cut-Paste: {e}")
                self.cutpaste_aug = None

    @staticmethod
    def target_parser(target, anomaly_label_id: int = 254, **kwargs):
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
        masks, labels = [], []  # Accumulate masks and their class labels

        # Iterate over unique label IDs present in the mask
        for label_id in target[0].unique():
            # Handle anomaly class (synthetic OOD objects from Cut-Paste)
            if label_id == anomaly_label_id:
                masks.append(target[0] == label_id)  # Binary mask for anomaly pixels
                labels.append(19)  # Anomaly uses train_id=19 (Cityscapes uses 0-18)
                continue

            # Find corresponding Cityscapes class by label ID
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            # Skip unknown classes or classes ignored during evaluation
            if cls is None or cls.ignore_in_eval:
                continue

            # Add binary mask and train_id for this class
            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        # Return masks, labels, and is_crowd flag (always False for Cityscapes)
        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Crea wrapper per transforms che include cut-paste
        train_transforms = self._create_train_transforms()

        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": lambda t, **kw: self.target_parser(
                t, anomaly_label_id=CutPasteAugmentation.ANOMALY_LABEL_ID, **kw
            ),
            "check_empty_targets": self.check_empty_targets,
        }

        self.cityscapes_train_dataset = DatasetWithCutPaste(
            transforms=train_transforms,
            cutpaste_aug=self.cutpaste_aug,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )

        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        return self

    def _create_train_transforms(self):
        """
        Get transforms for training data.

        Returns:
            Transforms: Configured transform pipeline (color jitter, scale, crop).
        """
        return self.transforms

    def train_dataloader(self):
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

    def val_dataloader(self):
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
    Dataset wrapper that applies Cut-Paste augmentation before standard transforms.

    This class extends the base Dataset to intercept image loading and inject
    Cut-Paste augmentation BEFORE other transforms (like color jitter, scaling).
    This order is important because Cut-Paste works on raw images.

    Processing Pipeline:
        1. Load raw image and target from zip
        2. Apply Cut-Paste augmentation (paste OOD objects, update mask)
        3. Apply standard transforms (color jitter, scale, crop)
        4. Parse target into per-class masks
    """

    def __init__(self, cutpaste_aug: Optional[CutPasteAugmentation] = None, **kwargs):
        """
        Initialize the dataset with optional Cut-Paste augmentation.

        Args:
            cutpaste_aug: CutPasteAugmentation instance (or None to disable).
            **kwargs: Arguments passed to parent Dataset class.
        """
        super().__init__(**kwargs)  # Initialize parent dataset
        self.cutpaste_aug = cutpaste_aug  # Store augmentation module

    def __getitem__(self, index: int):
        """
        Get a training sample with Cut-Paste augmentation applied.

        Overrides parent method to inject Cut-Paste before standard transforms.

        Args:
            index: Sample index in the dataset.

        Returns:
            tuple: (image, masks, labels, is_crowd) ready for training.
        """
        # Step 1: Load raw image and segmentation mask from zip
        img, target = self._load_item(index)

        # Step 2: Apply Cut-Paste augmentation (if enabled)
        if self.cutpaste_aug is not None:
            img, target = self.cutpaste_aug(img, target)

        # Step 3: Apply standard transforms (color jitter, random scale, crop)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Step 4: Parse target mask into per-class binary masks
        masks, labels, is_crowd = self.target_parser(target)

        return img, masks, labels, is_crowd

    def _load_item(self, index: int):
        """
        Load raw image and target mask without any transforms.

        Reads directly from zip files for efficient storage.

        Args:
            index: Sample index in the dataset.

        Returns:
            tuple: (image_tensor, target_mask) as raw tensors.
        """
        import torch
        from PIL import Image
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as F

        # Get file info for this sample
        img_info = self.imgs[index]
        target_info = self.targets[index]

        # Load image from zip: convert to RGB, then to normalized float tensor
        with self._get_zip().open(img_info.filename) as img_file:
            img = Image.open(img_file).convert("RGB")  # Ensure RGB format
            img = F.to_image(img)  # Convert PIL to tensor
            img = F.to_dtype(img, torch.float32, scale=True)  # Normalize to [0, 1]

        # Load target mask from zip: convert to Mask tensor
        target_zip = self._get_target_zip()
        with target_zip.open(target_info) as target_file:
            target = Image.open(target_file)  # Load as PIL
            target = tv_tensors.Mask(F.pil_to_tensor(target))  # Convert to Mask

        return img, target

    def _get_zip(self):
        """
        Get handle to image zip file (lazy initialization).

        Uses lazy loading to avoid opening zip until first access.
        Handles multi-worker data loading correctly.

        Returns:
            ZipFile: Open handle to image zip file.
        """
        import zipfile
        from torch.utils.data import get_worker_info

        # Check if we need to open the zip (first access or new worker)
        worker_info = get_worker_info()
        if worker_info is None or self.zip is None:
            if self.zip is None:
                self.zip = zipfile.ZipFile(self.zip_path, "r")
        return self.zip

    def _get_target_zip(self):
        """
        Get handle to target/label zip file (lazy initialization).

        Similar to _get_zip but for segmentation masks.
        Falls back to image zip if target_zip_path is not set.

        Returns:
            ZipFile: Open handle to target zip file.
        """
        import zipfile
        from torch.utils.data import get_worker_info

        # Check if we need to open the zip
        worker_info = get_worker_info()
        if worker_info is None or self.target_zip is None:
            if self.target_zip is None:
                # Use dedicated target zip or fall back to image zip
                target_path = self.target_zip_path or self.zip_path
                self.target_zip = zipfile.ZipFile(target_path, "r")
        return self.target_zip
