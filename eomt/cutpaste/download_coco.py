# ---------------------------------------------------------------
# Script per scaricare subset COCO filtrato (solo classi OOD)
# ---------------------------------------------------------------

import os
import json
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# Classi COCO che sovrappongono con Cityscapes (da escludere)
CITYSCAPES_OVERLAP_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
}

# Tutte le classi COCO
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

# Classi OOD da usare per cut-paste (escluse quelle che sovrappongono)
OOD_CLASS_IDS = [
    cat_id for cat_id, cat_name in COCO_CLASSES.items()
    if cat_name.lower() not in CITYSCAPES_OVERLAP_CLASSES
]


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """Scarica un file con progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_coco_subset(
    output_dir: str,
    split: str = "train2017",
    max_images: Optional[int] = None,
    min_object_area: int = 1000,
    force_download: bool = False,
) -> Path:
    """
    Scarica un subset di COCO con solo le classi OOD (non presenti in Cityscapes).
    
    Args:
        output_dir: Directory dove salvare i dati
        split: Split COCO da usare ("train2017" o "val2017")
        max_images: Numero massimo di immagini da scaricare (None = tutte)
        min_object_area: Area minima dell'oggetto in pixel
        force_download: Se True, riscarica anche se esiste già
        
    Returns:
        Path alla directory con il subset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Directory per immagini e maschere
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    metadata_path = output_path / "metadata.json"
    
    if metadata_path.exists() and not force_download:
        print(f"Subset COCO già presente in {output_path}")
        return output_path
    
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    # URL COCO
    annotations_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    images_url = f"http://images.cocodataset.org/zips/{split}.zip"
    
    # Scarica annotations
    annotations_zip = output_path / "annotations.zip"
    if not annotations_zip.exists() or force_download:
        print("Scaricando annotations COCO...")
        download_file(annotations_url, annotations_zip, "Annotations")
    
    # Estrai annotations
    annotations_dir = output_path / "annotations"
    if not annotations_dir.exists():
        print("Estraendo annotations...")
        with zipfile.ZipFile(annotations_zip, "r") as z:
            z.extractall(output_path)
    
    # Carica annotations
    ann_file = annotations_dir / f"instances_{split}.json"
    print(f"Caricando {ann_file}...")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)
    
    # Filtra annotations per classi OOD
    print("Filtrando classi OOD...")
    ood_annotations = []
    image_ids_with_ood = set()
    
    for ann in tqdm(coco_data["annotations"], desc="Filtering annotations"):
        if ann["category_id"] in OOD_CLASS_IDS:
            # Controlla area minima
            if ann["area"] >= min_object_area:
                # Controlla che abbia segmentazione valida
                if ann.get("segmentation") and not ann.get("iscrowd", False):
                    ood_annotations.append(ann)
                    image_ids_with_ood.add(ann["image_id"])
    
    print(f"Trovate {len(ood_annotations)} annotazioni OOD in {len(image_ids_with_ood)} immagini")
    
    # Limita numero immagini se richiesto
    if max_images and len(image_ids_with_ood) > max_images:
        import random
        random.seed(42)
        image_ids_with_ood = set(random.sample(list(image_ids_with_ood), max_images))
        ood_annotations = [a for a in ood_annotations if a["image_id"] in image_ids_with_ood]
    
    # Mappa image_id -> info
    image_info_map = {img["id"]: img for img in coco_data["images"]}
    
    # Scarica immagini necessarie
    print("Scaricando immagini...")
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
                print(f"Errore scaricando {img_filename}: {e}")
                continue
        
        downloaded_images[img_id] = {
            "filename": img_filename,
            "width": img_info["width"],
            "height": img_info["height"],
        }
    
    # Crea maschere per ogni oggetto OOD
    print("Creando maschere...")
    from pycocotools import mask as coco_mask
    from PIL import Image
    import numpy as np
    
    objects_metadata = []
    
    for ann in tqdm(ood_annotations, desc="Creating masks"):
        img_id = ann["image_id"]
        if img_id not in downloaded_images:
            continue
            
        img_info = downloaded_images[img_id]
        h, w = img_info["height"], img_info["width"]
        
        # Crea maschera dalla segmentazione
        try:
            if isinstance(ann["segmentation"], list):
                # Polygon format
                rles = coco_mask.frPyObjects(ann["segmentation"], h, w)
                rle = coco_mask.merge(rles)
            else:
                # RLE format
                rle = ann["segmentation"]
            
            mask_array = coco_mask.decode(rle)
            
            # Salva maschera
            mask_filename = f"{ann['id']}.png"
            mask_path = masks_dir / mask_filename
            Image.fromarray((mask_array * 255).astype(np.uint8)).save(mask_path)
            
            # Calcola bounding box
            ys, xs = np.where(mask_array > 0)
            if len(xs) == 0:
                continue
                
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            
            objects_metadata.append({
                "annotation_id": ann["id"],
                "image_id": img_id,
                "image_filename": img_info["filename"],
                "mask_filename": mask_filename,
                "category_id": ann["category_id"],
                "category_name": COCO_CLASSES.get(ann["category_id"], "unknown"),
                "bbox": bbox,
                "area": ann["area"],
            })
            
        except Exception as e:
            print(f"Errore creando maschera per annotation {ann['id']}: {e}")
            continue
    
    # Salva metadata
    metadata = {
        "split": split,
        "num_objects": len(objects_metadata),
        "num_images": len(downloaded_images),
        "ood_class_ids": OOD_CLASS_IDS,
        "excluded_classes": list(CITYSCAPES_OVERLAP_CLASSES),
        "objects": objects_metadata,
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSubset COCO salvato in {output_path}")
    print(f"  - {len(objects_metadata)} oggetti OOD")
    print(f"  - {len(downloaded_images)} immagini")
    
    # Cleanup
    if annotations_zip.exists():
        annotations_zip.unlink()
    if annotations_dir.exists():
        shutil.rmtree(annotations_dir)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download COCO OOD subset")
    parser.add_argument("--output_dir", type=str, default="./data/coco_ood",
                        help="Directory di output")
    parser.add_argument("--split", type=str, default="train2017",
                        help="Split COCO (train2017 o val2017)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Numero massimo di immagini")
    parser.add_argument("--min_area", type=int, default=1000,
                        help="Area minima oggetto in pixel")
    parser.add_argument("--force", action="store_true",
                        help="Forza ri-download")
    
    args = parser.parse_args()
    
    download_coco_subset(
        output_dir=args.output_dir,
        split=args.split,
        max_images=args.max_images,
        min_object_area=args.min_area,
        force_download=args.force,
    )
