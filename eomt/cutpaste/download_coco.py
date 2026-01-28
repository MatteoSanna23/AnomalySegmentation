"""
COCO OOD Subset Downloader
==========================
This script creates a dataset of "Out-of-Distribution" (OOD) objects.
It downloads images from the COCO dataset but filters out any object classes
that already exist in Cityscapes (like cars, people, buses).

The result is a collection of objects (bears, pizza, surfboards) that the
segmentation model has never seen, which are perfect for creating synthetic anomalies.
"""

import os
import json
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

# These classes exist in BOTH datasets. We must EXCLUDE them.
# If we paste a "person" onto a Cityscapes road, the model might correctly
# identify it as a person, which defeats the purpose of "anomaly" detection.
CITYSCAPES_OVERLAP_CLASSES = {
    "person",  # Exists in Cityscapes
    "bicycle",  # Exists in Cityscapes
    "car",  # Exists in Cityscapes
    "motorcycle",  # Exists in Cityscapes
    "bus",  # Exists in Cityscapes
    "train",  # Exists in Cityscapes
    "truck",  # Exists in Cityscapes
}

# Mapping of COCO IDs to readable names.
# COCO IDs are sparse (numbers are skipped).
COCO_CLASSES = {
    # ... (IDs 1-10: mostly vehicles/people, often excluded) ...
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    # ... (IDs 11-90: Animals, Food, Indoor items - these are good anomalies) ...
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

# Dynamically build the list of allowed classes.
# Logic: If it's NOT in the overlap list, keep it.
OOD_CLASS_IDS = [
    cat_id
    for cat_id, cat_name in COCO_CLASSES.items()
    if cat_name.lower() not in CITYSCAPES_OVERLAP_CLASSES
]

# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Helper to download large files with a progress bar.
    Uses streaming to keep memory usage low.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


# =============================================================================
# MAIN LOGIC
# =============================================================================


def download_coco_subset(
    output_dir: str,
    split: str = "train2017",
    max_images: Optional[int] = None,
    min_object_area: int = 1000,
    force_download: bool = False,
) -> Path:
    """
    Main driver function to prepare the anomaly dataset.

    Workflow:
    1. Download Metadata -> 2. Filter Objects -> 3. Download Images -> 4. Generate Masks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Standardize output structure
    images_dir = output_path / "images"  # Stores source JPEGs
    masks_dir = output_path / "masks"  # Stores binary PNG masks
    metadata_path = output_path / "metadata.json"  # The index file

    # Check if job is already done
    if metadata_path.exists() and not force_download:
        print(f"COCO subset already exists in {output_path}")
        return output_path

    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # --- STEP 1: Download Annotations (JSON only) ---
    # We download annotations first because they are small compared to images.
    # This allows us to filter *before* downloading gigabytes of image data.
    annotations_url = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    annotations_zip = output_path / "annotations.zip"

    if not annotations_zip.exists() or force_download:
        print("Downloading COCO annotations...")
        download_file(annotations_url, annotations_zip, "Annotations")

    annotations_dir = output_path / "annotations"
    if not annotations_dir.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, "r") as z:
            z.extractall(output_path)

    # --- STEP 2: Filter Annotations ---
    # Load the massive JSON file into memory
    ann_file = annotations_dir / f"instances_{split}.json"
    print(f"Loading {ann_file}...")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    print("Filtering for OOD classes...")
    ood_annotations = []
    image_ids_with_ood = set()

    for ann in tqdm(coco_data["annotations"], desc="Filtering annotations"):
        # CRITICAL FILTER 1: Is the class "Out of Distribution"?
        if ann["category_id"] in OOD_CLASS_IDS:
            # CRITICAL FILTER 2: Is the object big enough to be useful?
            # Tiny objects (like a distant bird) are bad for anomaly training.
            if ann["area"] >= min_object_area:
                # CRITICAL FILTER 3: Is it a valid polygon? (ignore crowds)
                if ann.get("segmentation") and not ann.get("iscrowd", False):
                    ood_annotations.append(ann)
                    image_ids_with_ood.add(ann["image_id"])

    print(f"Found {len(ood_annotations)} valid OOD objects.")

    # --- STEP 3: Limit Dataset Size (Optional) ---
    # If testing, we might only want 500 images, not 50,000.
    if max_images and len(image_ids_with_ood) > max_images:
        import random

        random.seed(42)  # Ensure we get the same subset every time
        image_ids_with_ood = set(random.sample(list(image_ids_with_ood), max_images))
        # Remove annotations for images we just discarded
        ood_annotations = [
            a for a in ood_annotations if a["image_id"] in image_ids_with_ood
        ]

    # Quick lookup for image filenames (id -> metadata)
    image_info_map = {img["id"]: img for img in coco_data["images"]}

    # --- STEP 4: Download Images ---
    # Only download the specific images we selected.
    print("Downloading images...")
    downloaded_images = {}

    for img_id in tqdm(image_ids_with_ood, desc="Downloading images"):
        img_info = image_info_map[img_id]
        img_filename = img_info["file_name"]
        img_url = f"http://images.cocodataset.org/{split}/{img_filename}"
        img_dest = images_dir / img_filename

        if not img_dest.exists():
            try:
                response = requests.get(img_url, timeout=30)
                response.raise_for_status()
                with open(img_dest, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {img_filename}: {e}")
                continue

        # Record successful downloads
        downloaded_images[img_id] = {
            "filename": img_filename,
            "width": img_info["width"],
            "height": img_info["height"],
        }

    # --- STEP 5: Generate Binary Masks ---
    # COCO stores masks as RLE (Run Length Encoding) or Polygons.
    # We need standard PNG images where white = object, black = background.
    print("Creating masks...")
    from pycocotools import mask as coco_mask
    from PIL import Image
    import numpy as np

    objects_metadata = []

    for ann in tqdm(ood_annotations, desc="Creating masks"):
        img_id = ann["image_id"]
        # Skip if image failed to download
        if img_id not in downloaded_images:
            continue

        img_info = downloaded_images[img_id]
        h, w = img_info["height"], img_info["width"]

        try:
            # Decode COCO polygon/RLE to a numpy bitmask
            if isinstance(ann["segmentation"], list):
                rles = coco_mask.frPyObjects(ann["segmentation"], h, w)
                rle = coco_mask.merge(rles)
            else:
                rle = ann["segmentation"]

            mask_array = coco_mask.decode(rle)

            # Save as PNG
            mask_filename = f"{ann['id']}.png"
            mask_path = masks_dir / mask_filename
            Image.fromarray((mask_array * 255).astype(np.uint8)).save(mask_path)

            # Calculate Bounding Box (x1, y1, x2, y2)
            ys, xs = np.where(mask_array > 0)
            if len(xs) == 0:
                continue
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            # Add to final index
            objects_metadata.append(
                {
                    "annotation_id": ann["id"],
                    "image_id": img_id,
                    "image_filename": img_info["filename"],
                    "mask_filename": mask_filename,
                    "category_id": ann["category_id"],
                    "category_name": COCO_CLASSES.get(ann["category_id"], "unknown"),
                    "bbox": bbox,
                    "area": ann["area"],
                }
            )

        except Exception as e:
            print(f"Mask generation error: {e}")
            continue

    # --- STEP 6: Save Index ---
    # Save a JSON file listing all available anomaly objects.
    # The CutPaste module will read this to know what objects it can paste.
    metadata = {
        "split": split,
        "num_objects": len(objects_metadata),
        "objects": objects_metadata,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSuccess! Saved {len(objects_metadata)} OOD objects to {output_path}")

    # Cleanup temporary zip files to save space
    if annotations_zip.exists():
        annotations_zip.unlink()
    if annotations_dir.exists():
        shutil.rmtree(annotations_dir)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download COCO OOD subset")
    parser.add_argument(
        "--output_dir", type=str, default="./data/coco_ood", help="Where to save data"
    )
    parser.add_argument("--split", type=str, default="train2017", help="COCO split")
    parser.add_argument(
        "--max_images", type=int, default=None, help="Limit number of images"
    )
    parser.add_argument(
        "--min_area", type=int, default=1000, help="Min pixels for object"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    args = parser.parse_args()

    download_coco_subset(
        output_dir=args.output_dir,
        split=args.split,
        max_images=args.max_images,
        min_object_area=args.min_area,
        force_download=args.force,
    )
