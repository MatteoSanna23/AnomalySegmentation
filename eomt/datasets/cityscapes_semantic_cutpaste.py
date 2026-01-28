# -----------------------------------------------------------------------------
# Pre-generated Cut-Paste Cityscapes Dataset Module
# Reads from filesystem folders instead of compressed zip archives
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.datasets import Cityscapes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms


# The specific label ID assigned to anomalies in the pre-generated masks
ANOMALY_LABEL_ID = 254


class CityscapesSemanticCutPaste(LightningDataModule):
    """
    LightningDataModule for the Cityscapes dataset with pre-applied Cut-Paste anomalies.

    This module is designed to read from a standard directory structure where
    images have already been processed and saved, rather than generating anomalies
    on-the-fly or reading from zip files.
    """

    def __init__(
        self,
        path: str,  # Path to the pre-generated dataset root
        original_cityscapes_path: str = None,  # Optional path to original zips for validation
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 20,  # 19 Standard Cityscapes classes + 1 Anomaly class
        color_jitter_enabled: bool = True,
        scale_range: tuple = (0.5, 2.0),
        check_empty_targets: bool = True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # Fallback to the main path if a specific original path is not provided
        self.original_cityscapes_path = original_cityscapes_path or path

        # Initialize data augmentation transforms
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        """
        Parses the raw segmentation mask into binary masks and class labels.

        This function iterates through unique values in the mask tensor.
        It specifically checks for the `ANOMALY_LABEL_ID`. If found, it assigns
        class index 19 (the anomaly class). Standard Cityscapes classes are
        mapped to their respective training IDs.

        Args:
            target: The raw ground truth mask tensor.

        Returns:
            masks: List of boolean tensors representing object masks.
            labels: List of integer class IDs.
            is_crowd: List of booleans indicating crowd annotations.
        """
        masks, labels = [], []

        for label_id in target[0].unique():
            # Handle the specific anomaly class
            if label_id == ANOMALY_LABEL_ID:
                masks.append(target[0] == label_id)
                labels.append(19)  # Assign fixed index 19 for anomaly
                continue

            # Handle standard Cityscapes classes
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            # Skip undefined classes or those marked 'ignore_in_eval'
            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        """
        Prepares the datasets for training and validation.

        For the training set, it initializes a `CutPasteFolderDataset` pointing
        to the directory containing the pre-augmented images.

        For the validation set, it checks if the folder structure exists. If it does,
        it uses the folder dataset. If not, it falls back to loading from the
        original Cityscapes zip files using the standard `Dataset` class.
        """
        # Initialize training dataset from the pre-generated folder
        self.train_dataset = CutPasteFolderDataset(
            root_path=Path(self.path),
            split="train",
            transforms=self.transforms,
            target_parser=self.target_parser,
        )

        # Determine the source for the validation dataset
        val_path = Path(self.original_cityscapes_path)

        # Check if validation data exists in uncompressed folder format
        if (Path(self.path) / "leftImg8bit" / "val").exists():
            self.val_dataset = CutPasteFolderDataset(
                root_path=Path(self.path),
                split="val",
                transforms=None,  # Validation does not require augmentation
                target_parser=self.target_parser,
            )
        else:
            # Fallback to reading from original Zip archives
            self.val_dataset = Dataset(
                img_suffix=".png",
                target_suffix=".png",
                img_stem_suffix="leftImg8bit",
                target_stem_suffix="gtFine_labelIds",
                zip_path=val_path / "leftImg8bit_trainvaltest.zip",
                target_zip_path=val_path / "gtFine_trainvaltest.zip",
                target_parser=self.target_parser,
                check_empty_targets=self.check_empty_targets,
                img_folder_path_in_zip=Path("./leftImg8bit/val"),
                target_folder_path_in_zip=Path("./gtFine/val"),
            )

        return self

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        Enables shuffling and drops the last incomplete batch to maintain shape consistency.
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        Uses the specific evaluation collate function.
        """
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class CutPasteFolderDataset(TorchDataset):
    """
    A custom PyTorch Dataset that reads images and masks directly from a directory structure.

    Expected file structure:
        root/leftImg8bit/{split}/{city}/*_leftImg8bit.png
        root/gtFine/{split}/{city}/*_gtFine_labelIds.png
    """

    def __init__(
        self,
        root_path: Path,
        split: str = "train",
        transforms=None,
        target_parser=None,
    ):
        self.root_path = root_path
        self.split = split
        self.transforms = transforms
        self.target_parser = target_parser

        # Locate all images in the specified split directory
        img_dir = root_path / "leftImg8bit" / split
        self.samples = []

        # Iterate through city directories to find image-mask pairs
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                # Construct the corresponding mask path based on the image filename
                base_name = img_path.stem.replace("_leftImg8bit", "")
                mask_path = (
                    root_path
                    / "gtFine"
                    / split
                    / city_dir.name
                    / f"{base_name}_gtFine_labelIds.png"
                )

                # Only add the sample if the corresponding mask exists
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        print(f"CutPasteFolderDataset: Found {len(self.samples)} images in {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Retrieves a single sample (image and target) from the dataset.

        Steps:
        1. Loads the RGB image and Label mask from disk.
        2. Ensures the mask dimensions match the image dimensions.
        3. Parses the mask to extract instance masks and class labels.
        4. Handles cases where the mask contains no valid objects.
        5. Applies defined transformations (augmentation).
        """
        img_path, mask_path = self.samples[index]

        # Load image as RGB tensor
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        # Load mask as Long tensor
        mask = tv_tensors.Mask(Image.open(mask_path), dtype=torch.long)

        # Resize mask if it does not match image dimensions (e.g., due to previous resizing)
        if img.shape[-2:] != mask.shape[-2:]:
            mask = F.resize(
                mask,
                list(img.shape[-2:]),
                interpolation=F.InterpolationMode.NEAREST,
            )

        # Parse the raw mask into separate instance masks and labels
        masks, labels, is_crowd = self.target_parser(mask)

        # Handle edge case: Empty target (no recognized classes in the image)
        if len(masks) == 0:
            h, w = img.shape[-2:]
            masks = [torch.zeros((h, w), dtype=torch.bool)]
            labels = [255]  # 255 is the standard ignore index
            is_crowd = [False]

        # Construct the target dictionary
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        # Apply data augmentation transforms if defined
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
