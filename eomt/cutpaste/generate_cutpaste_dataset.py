"""
Offline Cut-Paste Dataset Generator
====================================
This script creates a permanent, pre-augmented version of the Cityscapes dataset.
Instead of pasting objects "on the fly" during training (which slows down the GPU),
we generate a fixed dataset where anomalies are already pasted into the images.

Key benefits:
1. Faster Training Loop (removes CPU bottleneck during training).
2. Visual Debugging (allows inspection of generated anomalies before training).
3. Reproducibility (guarantees the same augmented data across different runs).
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
    Handles the logic of selecting an anomaly object, resizing it,
    and blending it seamlessly into a background image using alpha compositing.
    """

    # The specific pixel value (class ID) assigned to the anomaly mask.
    # The loss function will look for this ID to calculate anomaly detection performance.
    ANOMALY_LABEL_ID = 254

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
        Initialize the generator configuration.

        Args:
            coco_ood_path: Directory containing the OOD objects (images/masks/metadata).
            min_objects: Minimum number of anomalies to paste per image.
            max_objects: Maximum number of anomalies to paste per image.
            scale_range: Tuple (min, max) for random resizing of objects.
            blend_mode: 'alpha_feather' for soft edges, 'paste' for hard edges.
            feather_radius: Gaussian blur radius for edge softening (pixels).
            seed: Random seed for reproducibility.
        """
        self.coco_ood_path = Path(coco_ood_path)
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.blend_mode = blend_mode
        self.feather_radius = feather_radius

        # Ensure deterministic randomness for reproducible datasets
        random.seed(seed)
        np.random.seed(seed)

        # Load the index of available anomaly objects
        metadata_path = self.coco_ood_path / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.objects = self.metadata.get("objects", [])
        print(f"Loaded {len(self.objects)} OOD objects")

    def _load_object(self, obj_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads the image file and mask file for a specific anomaly object.

        Returns:
            Tuple of (RGB image array, Binary mask array) cropped to the object's bbox.
        """
        # Load the RGB image of the object (e.g., a bear)
        img_path = self.coco_ood_path / "images" / obj_info["image_filename"]
        img = np.array(Image.open(img_path).convert("RGB"))

        # Load the binary mask (shape of the object)
        mask_path = self.coco_ood_path / "masks" / obj_info["mask_filename"]
        mask = np.array(Image.open(mask_path).convert("L"))

        # Crop the arrays to the bounding box to remove empty space
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
        Resizes the object so it fits naturally in the scene.
        Enforces a hard limit: object cannot be larger than 40% of the scene dimensions.
        """
        h, w = obj_crop.shape[:2]
        scale = random.uniform(*self.scale_range)

        # Cap maximum size to avoid the object covering the entire image
        max_h = int(target_h * 0.4)
        max_w = int(target_w * 0.4)

        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Logic to preserve aspect ratio while respecting max dimensions
        if new_h > max_h:
            ratio = max_h / new_h
            new_h, new_w = max_h, int(new_w * ratio)
        if new_w > max_w:
            ratio = max_w / new_w
            new_w, new_h = max_w, int(new_h * ratio)

        # Prevent objects from becoming too tiny (less than 32px)
        new_h = max(new_h, 32)
        new_w = max(new_w, 32)

        # Resize the RGB image using linear interpolation (smoother)
        obj_scaled = cv2.resize(
            obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        # Resize the mask using nearest neighbor (keeps edges sharp)
        mask_scaled = cv2.resize(
            mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        return obj_scaled, mask_scaled

    def _create_feathered_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Creates a 'soft' transparency mask to blend edges smoothly.

        Why? If we simply paste an object, the edges are pixel-perfect and sharp.
        A neural network might learn to detect "sharp edges" instead of the object itself.
        Feathering blurs the edge to make the blending look more natural.
        """
        # Convert 0-255 integer mask to 0.0-1.0 float
        mask_float = mask.astype(np.float32) / 255.0

        if self.feather_radius > 0:
            # 1. Shrink the mask slightly (erode) to define the solid core
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(
                (mask_float * 255).astype(np.uint8), kernel, iterations=2
            )
            # 2. Blur the original mask to create the soft edge
            blurred = cv2.GaussianBlur(mask_float, (0, 0), self.feather_radius)

            # 3. Combine: Keep the core solid, use the blur for the edges
            mask_float = np.maximum(eroded.astype(np.float32) / 255.0, blurred)

        return mask_float

    def apply_cutpaste(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The main augmentation pipeline.

        1. Pick N random OOD objects.
        2. Resize them.
        3. Pick random (x,y) coordinates on the background.
        4. Blend the object into the image using the alpha mask.
        5. Update the ground truth label to '254' at that location.
        """
        img = image.copy()  # Protect original data
        mask = target_mask.copy()

        num_objects = random.randint(self.min_objects, self.max_objects)
        selected = random.sample(self.objects, min(num_objects, len(self.objects)))

        img_h, img_w = img.shape[:2]

        for obj_info in selected:
            try:
                # Prepare object
                obj_crop, obj_mask = self._load_object(obj_info)
                obj_crop, obj_mask = self._scale_object(
                    obj_crop, obj_mask, img_h, img_w
                )

                obj_h, obj_w = obj_crop.shape[:2]

                # Random positioning
                pos_y = random.randint(0, max(0, img_h - obj_h))
                pos_x = random.randint(0, max(0, img_w - obj_w))

                # Define regions (Crop coordinates)
                y1, x1 = max(0, pos_y), max(0, pos_x)
                y2, x2 = min(img_h, pos_y + obj_h), min(img_w, pos_x + obj_w)

                # Coordinate math to handle if object goes partially off-screen
                obj_y1, obj_x1 = y1 - pos_y, x1 - pos_x
                obj_y2, obj_x2 = obj_y1 + (y2 - y1), obj_x1 + (x2 - x1)

                # Get the slices
                obj_region = obj_crop[obj_y1:obj_y2, obj_x1:obj_x2]
                mask_region = obj_mask[obj_y1:obj_y2, obj_x1:obj_x2]
                img_region = img[y1:y2, x1:x2]

                # Generate Alpha Mask (Soft edges)
                if self.blend_mode == "alpha_feather":
                    alpha = self._create_feathered_mask(mask_region)[..., np.newaxis]
                else:
                    alpha = (mask_region > 127).astype(np.float32)[..., np.newaxis]

                # Perform Alpha Blending:
                # Result = (Object * Alpha) + (Background * (1 - Alpha))
                blended = (obj_region * alpha + img_region * (1 - alpha)).astype(
                    np.uint8
                )
                img[y1:y2, x1:x2] = blended

                # Update the Semantic Mask (Ground Truth)
                # Any pixel belonging to the object is now Class 254 (Anomaly)
                binary_mask = mask_region > 127
                mask[y1:y2, x1:x2][binary_mask] = self.ANOMALY_LABEL_ID

            except Exception as e:
                continue

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
    Orchestrates the batch processing.
    It reads raw Cityscapes ZIPs, augments 50% (default) of them,
    and saves the new dataset structure to disk.
    """
    random.seed(seed)
    np.random.seed(seed)

    cityscapes_path = Path(cityscapes_path)
    output_path = Path(output_path)

    # Set up output directories
    output_images = output_path / "leftImg8bit" / split
    output_masks = output_path / "gtFine" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    generator = CutPasteGenerator(
        coco_ood_path=coco_ood_path,
        min_objects=min_objects,
        max_objects=max_objects,
        scale_range=scale_range,
        blend_mode=blend_mode,
        seed=seed,
    )

    # Directly read ZIP files to avoid extracting the huge Cityscapes dataset manually
    img_zip_path = cityscapes_path / "leftImg8bit_trainvaltest.zip"
    mask_zip_path = cityscapes_path / "gtFine_trainvaltest.zip"

    print(f"Opening {img_zip_path}...")
    img_zip = zipfile.ZipFile(img_zip_path, "r")
    mask_zip = zipfile.ZipFile(mask_zip_path, "r")

    # Filter for the correct split (train/val)
    img_prefix = f"leftImg8bit/{split}/"
    img_files = [
        f
        for f in img_zip.namelist()
        if f.startswith(img_prefix) and f.endswith("_leftImg8bit.png")
    ]

    print(f"Found {len(img_files)} images in {split}")

    stats = {"total": 0, "with_cutpaste": 0, "original": 0}
    metadata = []

    for img_path in tqdm(img_files, desc=f"Generating {split}"):
        # Extract city name and filename from path
        parts = img_path.split("/")
        city = parts[2]
        filename = parts[3]
        base_name = filename.replace("_leftImg8bit.png", "")
        mask_path = f"gtFine/{split}/{city}/{base_name}_gtFine_labelIds.png"

        # Create subfolders for the city
        (output_images / city).mkdir(exist_ok=True)
        (output_masks / city).mkdir(exist_ok=True)

        # Read binary data from ZIP
        with img_zip.open(img_path) as f:
            img = np.array(Image.open(f).convert("RGB"))

        with mask_zip.open(mask_path) as f:
            mask = np.array(Image.open(f))

        # Decide whether to apply Cut-Paste augmentation
        apply_cutpaste = random.random() < cutpaste_ratio

        if apply_cutpaste:
            img_out, mask_out = generator.apply_cutpaste(img, mask)
            stats["with_cutpaste"] += 1
        else:
            img_out, mask_out = img, mask
            stats["original"] += 1

        stats["total"] += 1

        # Save results to the new output folder
        out_img_path = output_images / city / filename
        out_mask_path = output_masks / city / f"{base_name}_gtFine_labelIds.png"

        Image.fromarray(img_out).save(out_img_path)
        Image.fromarray(mask_out).save(out_mask_path)

        metadata.append(
            {
                "image": str(out_img_path.relative_to(output_path)),
                "mask": str(out_mask_path.relative_to(output_path)),
                "has_anomaly": apply_cutpaste,
                "city": city,
            }
        )

    img_zip.close()
    mask_zip.close()

    # Save a JSON manifest of what we created
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
