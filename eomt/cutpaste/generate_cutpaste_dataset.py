"""
Offline Cut-Paste Dataset Generator
====================================

This script generates a pre-augmented version of the Cityscapes dataset
with Cut-Paste anomalies already applied and saved to disk.

Why Offline Generation?
    - Faster training: no augmentation overhead during training
    - Consistent augmentation: same augmented images across runs
    - Easy inspection: can visually verify augmented images before training
    - Storage tradeoff: requires more disk space but faster iteration

Output Structure:
    output_path/
    ├── leftImg8bit/
    │   └── train/
    │       └── {city}/
    │           └── {city}_XXXXXX_YYYYYY_leftImg8bit.png
    ├── gtFine/
    │   └── train/
    │       └── {city}/
    │           └── {city}_XXXXXX_YYYYYY_gtFine_labelIds.png
    └── metadata_train.json  # Statistics and file list

Usage:
    python generate_cutpaste_dataset.py \\
        --cityscapes_path /path/to/cityscapes \\
        --coco_ood_path /path/to/coco_ood \\
        --output_path /path/to/output \\
        --cutpaste_ratio 0.5
"""

import os
import json
import random
import zipfile
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from tqdm import tqdm
import cv2


# =============================================================================
# CUT-PASTE GENERATOR CLASS
# =============================================================================


class CutPasteGenerator:
    """
    Generates augmented images with OOD objects pasted onto them.

    This is a simplified version of CutPasteAugmentation optimized for
    offline batch processing. Loads COCO OOD objects and pastes them
    onto Cityscapes images with configurable blending.
    """

    ANOMALY_LABEL_ID = 254  # Pixel value for anomaly regions in masks

    def __init__(
        self,
        coco_ood_path: str,
        min_objects: int = 1,
        max_objects: int = 3,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        blend_mode: str = "alpha_feather",
        feather_radius: int = 5,
        seed: int = 42,
    ):
        """
        Initialize the Cut-Paste generator.

        Args:
            coco_ood_path: Path to downloaded COCO OOD subset.
            min_objects: Minimum number of objects to paste per image.
            max_objects: Maximum number of objects to paste per image.
            scale_range: Random scale factor range (min, max).
            blend_mode: Blending mode ("paste" or "alpha_feather").
            feather_radius: Gaussian blur sigma for feathered edges.
            seed: Random seed for reproducibility.
        """
        self.coco_ood_path = Path(coco_ood_path)
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.blend_mode = blend_mode
        self.feather_radius = feather_radius

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Load COCO OOD metadata (list of available objects)
        metadata_path = self.coco_ood_path / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.objects = self.metadata.get("objects", [])
        print(f"Loaded {len(self.objects)} OOD objects")

    def _load_object(self, obj_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and mask for a specific OOD object.

        Args:
            obj_info: Dictionary with object metadata (filename, bbox).

        Returns:
            tuple: (cropped_image, cropped_mask) as numpy arrays.
        """
        # Load full image containing the object
        img_path = self.coco_ood_path / "images" / obj_info["image_filename"]
        img = np.array(Image.open(img_path).convert("RGB"))

        # Load corresponding binary mask
        mask_path = self.coco_ood_path / "masks" / obj_info["mask_filename"]
        mask = np.array(Image.open(mask_path).convert("L"))

        # Crop to bounding box to get just the object
        x1, y1, x2, y2 = obj_info["bbox"]
        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _scale_object(
        self,
        obj_crop: np.ndarray,
        mask_crop: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize object with random scale, capped to max 40% of image size.

        Args:
            obj_crop: Object image crop [H, W, 3].
            mask_crop: Object mask crop [H, W].
            target_h: Target image height (to compute max size).
            target_w: Target image width (to compute max size).

        Returns:
            tuple: (scaled_image, scaled_mask).
        """
        h, w = obj_crop.shape[:2]
        scale = random.uniform(*self.scale_range)  # Random scale factor

        # Cap maximum size to 40% of target image dimensions
        max_h = int(target_h * 0.4)
        max_w = int(target_w * 0.4)

        # Calculate new dimensions with scale
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Clamp to maximum allowed size (preserve aspect ratio)
        if new_h > max_h:
            ratio = max_h / new_h
            new_h, new_w = max_h, int(new_w * ratio)
        if new_w > max_w:
            ratio = max_w / new_w
            new_w, new_h = max_w, int(new_h * ratio)

        # Ensure minimum size of 32x32 pixels
        new_h = max(new_h, 32)
        new_w = max(new_w, 32)

        # Resize image (bilinear) and mask (nearest neighbor)
        obj_scaled = cv2.resize(
            obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        mask_scaled = cv2.resize(
            mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        return obj_scaled, mask_scaled

    def _create_feathered_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create mask with soft feathered edges using erode + blur technique.

        This prevents the model from learning to detect anomalies based on
        artificial hard edges from the pasting process.

        Args:
            mask: Binary mask [H, W] with values 0-255.

        Returns:
            np.ndarray: Feathered mask [H, W] with float values 0-1.
        """
        # Normalize mask to float [0, 1]
        mask_float = mask.astype(np.float32) / 255.0

        if self.feather_radius > 0:
            # Step 1: Erode mask to create solid core
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(
                (mask_float * 255).astype(np.uint8), kernel, iterations=2
            )
            # Step 2: Blur original mask for soft edges
            blurred = cv2.GaussianBlur(mask_float, (0, 0), self.feather_radius)
            # Step 3: Combine - solid core with soft transition
            mask_float = np.maximum(eroded.astype(np.float32) / 255.0, blurred)

        return mask_float

    def apply_cutpaste(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Cut-Paste augmentation to an image.

        Randomly selects OOD objects and pastes them at random locations.
        Updates the segmentation mask to mark pasted regions as anomaly.

        Args:
            image: RGB image as numpy array [H, W, 3].
            target_mask: Segmentation mask as numpy array [H, W].

        Returns:
            tuple: (modified_image, modified_mask) with anomalies added.
        """
        img = image.copy()  # Don't modify original
        mask = target_mask.copy()

        # Randomly select number of objects to paste
        num_objects = random.randint(self.min_objects, self.max_objects)
        selected = random.sample(self.objects, min(num_objects, len(self.objects)))

        img_h, img_w = img.shape[:2]

        # Paste each selected object
        for obj_info in selected:
            try:
                # Load and scale the object
                obj_crop, obj_mask = self._load_object(obj_info)
                obj_crop, obj_mask = self._scale_object(
                    obj_crop, obj_mask, img_h, img_w
                )

                obj_h, obj_w = obj_crop.shape[:2]

                # Choose random position for pasting
                pos_y = random.randint(0, max(0, img_h - obj_h))
                pos_x = random.randint(0, max(0, img_w - obj_w))

                # Calculate valid region (handle edge cases)
                y1, x1 = max(0, pos_y), max(0, pos_x)
                y2, x2 = min(img_h, pos_y + obj_h), min(img_w, pos_x + obj_w)

                # Calculate corresponding region in object crop
                obj_y1, obj_x1 = y1 - pos_y, x1 - pos_x
                obj_y2, obj_x2 = obj_y1 + (y2 - y1), obj_x1 + (x2 - x1)

                # Extract regions for blending
                obj_region = obj_crop[obj_y1:obj_y2, obj_x1:obj_x2]
                mask_region = obj_mask[obj_y1:obj_y2, obj_x1:obj_x2]
                img_region = img[y1:y2, x1:x2]

                # Create alpha mask for blending
                if self.blend_mode == "alpha_feather":
                    alpha = self._create_feathered_mask(mask_region)[..., np.newaxis]
                else:
                    # Simple binary mask (hard edges)
                    alpha = (mask_region > 127).astype(np.float32)[..., np.newaxis]

                # Alpha blend: output = obj * alpha + background * (1 - alpha)
                blended = (obj_region * alpha + img_region * (1 - alpha)).astype(
                    np.uint8
                )
                img[y1:y2, x1:x2] = blended

                # Update segmentation mask: mark object pixels as anomaly
                binary_mask = mask_region > 127
                mask[y1:y2, x1:x2][binary_mask] = self.ANOMALY_LABEL_ID

            except Exception as e:
                continue  # Skip failed objects, continue with others

        return img, mask


# =============================================================================
# DATASET GENERATION FUNCTION
# =============================================================================


def generate_dataset(
    cityscapes_path: str,
    coco_ood_path: str,
    output_path: str,
    split: str = "train",
    cutpaste_ratio: float = 0.5,
    min_objects: int = 1,
    max_objects: int = 3,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    blend_mode: str = "alpha_feather",
    seed: int = 42,
):
    """
    Generate Cityscapes dataset with pre-applied Cut-Paste augmentation.

    Reads from Cityscapes zip files, applies Cut-Paste to a subset of images,
    and saves the augmented dataset to disk in Cityscapes-compatible format.

    Args:
        cityscapes_path: Path to folder containing Cityscapes zip files.
        coco_ood_path: Path to downloaded COCO OOD subset.
        output_path: Output directory for augmented dataset.
        split: Dataset split to process ("train" or "val").
        cutpaste_ratio: Fraction of images to augment (0.0 to 1.0).
        min_objects: Minimum objects to paste per augmented image.
        max_objects: Maximum objects to paste per augmented image.
        scale_range: Random scale factor range for objects.
        blend_mode: Blending mode for pasting.
        seed: Random seed for reproducibility.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    cityscapes_path = Path(cityscapes_path)
    output_path = Path(output_path)

    # Create output directory structure (mirrors Cityscapes format)
    output_images = output_path / "leftImg8bit" / split
    output_masks = output_path / "gtFine" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    # Initialize Cut-Paste generator
    generator = CutPasteGenerator(
        coco_ood_path=coco_ood_path,
        min_objects=min_objects,
        max_objects=max_objects,
        scale_range=scale_range,
        blend_mode=blend_mode,
        seed=seed,
    )

    # Open Cityscapes zip files
    img_zip_path = cityscapes_path / "leftImg8bit_trainvaltest.zip"
    mask_zip_path = cityscapes_path / "gtFine_trainvaltest.zip"

    print(f"Opening {img_zip_path}...")
    img_zip = zipfile.ZipFile(img_zip_path, "r")
    mask_zip = zipfile.ZipFile(mask_zip_path, "r")

    # Find all images in the specified split
    img_prefix = f"leftImg8bit/{split}/"
    img_files = [
        f
        for f in img_zip.namelist()
        if f.startswith(img_prefix) and f.endswith("_leftImg8bit.png")
    ]

    print(f"Found {len(img_files)} images in {split}")

    # Track statistics
    stats = {"total": 0, "with_cutpaste": 0, "original": 0}
    metadata = []

    # Process each image
    for img_path in tqdm(img_files, desc=f"Generating {split}"):
        # Parse path: leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        parts = img_path.split("/")
        city = parts[2]  # e.g., "aachen"
        filename = parts[3]  # e.g., "aachen_000000_000019_leftImg8bit.png"
        base_name = filename.replace("_leftImg8bit.png", "")

        # Construct corresponding mask path
        mask_path = f"gtFine/{split}/{city}/{base_name}_gtFine_labelIds.png"

        # Create city subdirectory in output
        (output_images / city).mkdir(exist_ok=True)
        (output_masks / city).mkdir(exist_ok=True)

        # Load image and mask from zip
        with img_zip.open(img_path) as f:
            img = np.array(Image.open(f).convert("RGB"))

        with mask_zip.open(mask_path) as f:
            mask = np.array(Image.open(f))

        # Randomly decide whether to apply Cut-Paste
        apply_cutpaste = random.random() < cutpaste_ratio

        if apply_cutpaste:
            img_out, mask_out = generator.apply_cutpaste(img, mask)
            stats["with_cutpaste"] += 1
        else:
            # Keep original image (no augmentation)
            img_out, mask_out = img, mask
            stats["original"] += 1

        stats["total"] += 1

        # Save augmented/original image and mask
        out_img_path = output_images / city / filename
        out_mask_path = output_masks / city / f"{base_name}_gtFine_labelIds.png"

        Image.fromarray(img_out).save(out_img_path)
        Image.fromarray(mask_out).save(out_mask_path)

        # Record metadata for this image
        metadata.append(
            {
                "image": str(out_img_path.relative_to(output_path)),
                "mask": str(out_mask_path.relative_to(output_path)),
                "has_anomaly": apply_cutpaste,
                "city": city,
            }
        )

    # Close zip files
    img_zip.close()
    mask_zip.close()

    # Save metadata JSON with statistics and file list
    meta_path = output_path / f"metadata_{split}.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "split": split,
                "stats": stats,
                "cutpaste_ratio": cutpaste_ratio,
                "anomaly_label_id": CutPasteGenerator.ANOMALY_LABEL_ID,
                "files": metadata,
            },
            f,
            indent=2,
        )

    # Print summary
    print(f"\nDataset generated in {output_path}")
    print(f"  Total images: {stats['total']}")
    print(f"  With Cut-Paste: {stats['with_cutpaste']}")
    print(f"  Original (unchanged): {stats['original']}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Cityscapes dataset with Cut-Paste augmentation"
    )
    parser.add_argument(
        "--cityscapes_path",
        type=str,
        required=True,
        help="Path to folder containing Cityscapes zip files",
    )
    parser.add_argument(
        "--coco_ood_path",
        type=str,
        required=True,
        help="Path to downloaded COCO OOD subset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for augmented dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (train/val)",
    )
    parser.add_argument(
        "--cutpaste_ratio",
        type=float,
        default=0.5,
        help="Fraction of images to augment (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min_objects",
        type=int,
        default=1,
        help="Minimum number of OOD objects to paste",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=3,
        help="Maximum number of OOD objects to paste",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Run dataset generation with provided arguments
    generate_dataset(
        cityscapes_path=args.cityscapes_path,
        coco_ood_path=args.coco_ood_path,
        output_path=args.output_path,
        split=args.split,
        cutpaste_ratio=args.cutpaste_ratio,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        seed=args.seed,
    )
