"""
COCO OOD Subset Downloader
==========================

This script downloads a filtered subset of the COCO dataset containing only
Out-of-Distribution (OOD) objects - objects that don't appear in Cityscapes.

Why do we need this?
- Cityscapes has 19 semantic classes (person, car, road, etc.)
- COCO has 80+ object categories
- We want objects that the model has NEVER seen during normal training
- These become synthetic anomalies when pasted onto Cityscapes images

Excluded classes (present in both COCO and Cityscapes):
- person, bicycle, car, motorcycle, bus, train, truck

Included OOD classes (examples):
- Animals: dog, cat, elephant, giraffe, zebra, bear
- Objects: umbrella, suitcase, chair, bottle, laptop
- Food: banana, pizza, sandwich, apple

Output structure:
    output_dir/
    ├── images/          # Source COCO images (full images)
    ├── masks/           # Binary masks for each object (one per annotation)
    └── metadata.json    # Index of all objects with bbox, category, paths

Usage:
    python -m cutpaste.download_coco --output_dir /path/to/coco_ood --max_images 5000
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

# Classes that exist in BOTH COCO and Cityscapes - must be excluded
# These would create in-distribution objects, not anomalies
CITYSCAPES_OVERLAP_CLASSES = {
    "person",  # Cityscapes: pedestrians
    "bicycle",  # Cityscapes: bicycle
    "car",  # Cityscapes: car
    "motorcycle",  # Cityscapes: motorcycle
    "bus",  # Cityscapes: bus
    "train",  # Cityscapes: train
    "truck",  # Cityscapes: truck
}

# Complete COCO category mapping: ID -> class name
# Note: IDs are not consecutive (some numbers skipped in original COCO)
COCO_CLASSES = {
    # People and vehicles (IDs 1-10) - some will be excluded
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
    # Street objects (IDs 11-15)
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    # Animals (IDs 16-25) - all are OOD for Cityscapes
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
    # Accessories (IDs 27-33)
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    # Sports equipment (IDs 34-43)
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
    # Kitchen items (IDs 44-51)
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    # Food (IDs 52-61) - great OOD objects
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
    # Furniture (IDs 62-70)
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    # Electronics (IDs 72-82)
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
    # Misc objects (IDs 84-90)
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

# Build list of OOD category IDs by filtering out Cityscapes overlaps
# These are the only classes we'll download and use for cut-paste
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
    Download a file from URL with progress bar.

    Uses streaming to handle large files without loading into memory.
    """
    response = requests.get(url, stream=True)  # Stream mode for large files
    response.raise_for_status()  # Raise exception if download fails
    total_size = int(
        response.headers.get("content-length", 0)
    )  # File size for progress

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                f.write(chunk)
                pbar.update(len(chunk))


# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================


def download_coco_subset(
    output_dir: str,
    split: str = "train2017",
    max_images: Optional[int] = None,
    min_object_area: int = 1000,
    force_download: bool = False,
) -> Path:
    """
    Download a filtered COCO subset containing only OOD classes.

    This function:
    1. Downloads COCO annotations (JSON with all object info)
    2. Filters annotations to keep only OOD classes
    3. Downloads only the images containing OOD objects
    4. Creates binary masks for each OOD object
    5. Saves metadata.json with index of all objects

    Args:
        output_dir: Directory to save the subset.
        split: COCO split to use ("train2017" or "val2017").
        max_images: Maximum images to download (None = all available).
        min_object_area: Minimum object area in pixels (filters tiny objects).
        force_download: If True, re-download even if exists.

    Returns:
        Path to the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create output directory

    # Define subdirectories for images and masks
    images_dir = output_path / "images"  # Full COCO images
    masks_dir = output_path / "masks"  # Binary masks per object
    metadata_path = output_path / "metadata.json"  # Object index

    # Skip if already downloaded (unless force=True)
    if metadata_path.exists() and not force_download:
        print(f"COCO subset already exists in {output_path}")
        return output_path

    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # =================================================================
    # STEP 1: DOWNLOAD COCO ANNOTATIONS
    # =================================================================
    # COCO provides annotations separately from images (much smaller download)
    annotations_url = (
        f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    images_url = f"http://images.cocodataset.org/zips/{split}.zip"  # Not used - we download images individually

    # Download annotations ZIP (~240MB for train+val)
    annotations_zip = output_path / "annotations.zip"
    if not annotations_zip.exists() or force_download:
        print("Downloading COCO annotations...")
        download_file(annotations_url, annotations_zip, "Annotations")

    # Extract annotations ZIP
    annotations_dir = output_path / "annotations"
    if not annotations_dir.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, "r") as z:
            z.extractall(output_path)

    # =================================================================
    # STEP 2: LOAD AND FILTER ANNOTATIONS
    # =================================================================
    # Load the main COCO annotations file (instances_train2017.json ~450MB)
    ann_file = annotations_dir / f"instances_{split}.json"
    print(f"Loading {ann_file}...")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)  # Contains "images", "annotations", "categories"

    # Filter annotations to keep only OOD classes
    print("Filtering for OOD classes...")
    ood_annotations = []  # Will store filtered annotations
    image_ids_with_ood = set()  # Track which images have OOD objects

    for ann in tqdm(coco_data["annotations"], desc="Filtering annotations"):
        # Check if category is in our OOD list
        if ann["category_id"] in OOD_CLASS_IDS:
            # Filter out tiny objects (< min_object_area pixels)
            if ann["area"] >= min_object_area:
                # Ensure valid segmentation exists (not crowd annotation)
                if ann.get("segmentation") and not ann.get("iscrowd", False):
                    ood_annotations.append(ann)
                    image_ids_with_ood.add(ann["image_id"])

    print(
        f"Found {len(ood_annotations)} OOD annotations in {len(image_ids_with_ood)} images"
    )

    # =================================================================
    # STEP 3: LIMIT IMAGES (OPTIONAL)
    # =================================================================
    # If max_images specified, randomly sample to limit download size
    if max_images and len(image_ids_with_ood) > max_images:
        import random

        random.seed(42)  # Reproducible sampling
        image_ids_with_ood = set(random.sample(list(image_ids_with_ood), max_images))
        # Keep only annotations for selected images
        ood_annotations = [
            a for a in ood_annotations if a["image_id"] in image_ids_with_ood
        ]

    # Create lookup table: image_id -> image metadata
    image_info_map = {img["id"]: img for img in coco_data["images"]}

    # =================================================================
    # STEP 4: DOWNLOAD IMAGES
    # =================================================================
    # Download only the images that contain OOD objects (not full dataset)
    print("Downloading images...")
    downloaded_images = {}  # Track successfully downloaded images

    for img_id in tqdm(image_ids_with_ood, desc="Downloading images"):
        img_info = image_info_map[img_id]
        img_filename = img_info["file_name"]  # e.g., "000000123456.jpg"
        img_url = f"http://images.cocodataset.org/{split}/{img_filename}"
        img_dest = images_dir / img_filename

        # Skip if already downloaded
        if not img_dest.exists():
            try:
                response = requests.get(img_url, timeout=30)  # 30s timeout
                response.raise_for_status()
                with open(img_dest, "wb") as f:
                    f.write(response.content)  # Save image to disk
            except Exception as e:
                print(f"Error downloading {img_filename}: {e}")
                continue  # Skip this image

        # Store image metadata for later use
        downloaded_images[img_id] = {
            "filename": img_filename,
            "width": img_info["width"],
            "height": img_info["height"],
        }

    # =================================================================
    # STEP 5: CREATE BINARY MASKS FOR EACH OBJECT
    # =================================================================
    # Convert COCO segmentation format to binary PNG masks
    print("Creating masks...")
    from pycocotools import mask as coco_mask  # COCO's mask utilities
    from PIL import Image
    import numpy as np

    objects_metadata = []  # Will store metadata for each object

    for ann in tqdm(ood_annotations, desc="Creating masks"):
        img_id = ann["image_id"]
        if img_id not in downloaded_images:
            continue  # Skip if image download failed

        img_info = downloaded_images[img_id]
        h, w = img_info["height"], img_info["width"]  # Image dimensions

        # Convert COCO segmentation to binary mask array
        try:
            if isinstance(ann["segmentation"], list):
                # Polygon format: list of [x1,y1,x2,y2,...] polygons
                rles = coco_mask.frPyObjects(
                    ann["segmentation"], h, w
                )  # Convert to RLE
                rle = coco_mask.merge(rles)  # Merge multiple polygons
            else:
                # Already in RLE (Run-Length Encoding) format
                rle = ann["segmentation"]

            # Decode RLE to binary numpy array (0=background, 1=object)
            mask_array = coco_mask.decode(rle)

            # Save mask as PNG (0 or 255 values)
            mask_filename = f"{ann['id']}.png"  # Use annotation ID as filename
            mask_path = masks_dir / mask_filename
            Image.fromarray((mask_array * 255).astype(np.uint8)).save(mask_path)

            # Calculate tight bounding box from mask
            ys, xs = np.where(mask_array > 0)  # Find all object pixels
            if len(xs) == 0:
                continue  # Skip empty masks

            # bbox format: [x1, y1, x2, y2] (top-left to bottom-right)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            # Store object metadata for the index
            objects_metadata.append(
                {
                    "annotation_id": ann["id"],  # COCO annotation ID
                    "image_id": img_id,  # Source image ID
                    "image_filename": img_info["filename"],  # Image file
                    "mask_filename": mask_filename,  # Mask file
                    "category_id": ann["category_id"],  # COCO category ID
                    "category_name": COCO_CLASSES.get(ann["category_id"], "unknown"),
                    "bbox": bbox,  # Bounding box [x1, y1, x2, y2]
                    "area": ann["area"],  # Object area in pixels
                }
            )

        except Exception as e:
            print(f"Error creating mask for annotation {ann['id']}: {e}")
            continue

    # =================================================================
    # STEP 6: SAVE METADATA INDEX
    # =================================================================
    # Create JSON index with all object information
    metadata = {
        "split": split,  # COCO split used
        "num_objects": len(objects_metadata),  # Total OOD objects
        "num_images": len(downloaded_images),  # Total images downloaded
        "ood_class_ids": OOD_CLASS_IDS,  # List of OOD category IDs
        "excluded_classes": list(CITYSCAPES_OVERLAP_CLASSES),  # Excluded classes
        "objects": objects_metadata,  # Full object list with paths and bboxes
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)  # Pretty-print JSON

    print(f"\nCOCO subset saved to {output_path}")
    print(f"  - {len(objects_metadata)} OOD objects")
    print(f"  - {len(downloaded_images)} images")

    # =================================================================
    # CLEANUP: Remove temporary files
    # =================================================================
    if annotations_zip.exists():
        annotations_zip.unlink()  # Delete annotations.zip
    if annotations_dir.exists():
        shutil.rmtree(annotations_dir)  # Delete extracted annotations folder

    return output_path


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download COCO OOD subset for Cut-Paste augmentation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/coco_ood",
        help="Output directory for the subset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        help="COCO split (train2017 or val2017)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to download (None = all)",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=1000,
        help="Minimum object area in pixels (filters tiny objects)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )

    args = parser.parse_args()

    download_coco_subset(
        output_dir=args.output_dir,
        split=args.split,
        max_images=args.max_images,
        min_object_area=args.min_area,
        force_download=args.force,
    )
