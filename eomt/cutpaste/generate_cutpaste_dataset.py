# ---------------------------------------------------------------
# Script per generare dataset Cityscapes con Cut-Paste pre-applicato
# Genera immagini e maschere offline per training veloce
# ---------------------------------------------------------------

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


class CutPasteGenerator:
    """Genera immagini con oggetti OOD incollati."""

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
        self.coco_ood_path = Path(coco_ood_path)
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.blend_mode = blend_mode
        self.feather_radius = feather_radius

        random.seed(seed)
        np.random.seed(seed)

        # Carica metadata
        metadata_path = self.coco_ood_path / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.objects = self.metadata.get("objects", [])
        print(f"Caricati {len(self.objects)} oggetti OOD")

    def _load_object(self, obj_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Carica immagine e maschera di un oggetto."""
        img_path = self.coco_ood_path / "images" / obj_info["image_filename"]
        img = np.array(Image.open(img_path).convert("RGB"))

        mask_path = self.coco_ood_path / "masks" / obj_info["mask_filename"]
        mask = np.array(Image.open(mask_path).convert("L"))

        x1, y1, x2, y2 = obj_info["bbox"]
        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _scale_object(
        self,
        obj_crop: np.ndarray,
        mask_crop: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ridimensiona oggetto."""
        h, w = obj_crop.shape[:2]
        scale = random.uniform(*self.scale_range)

        max_h = int(target_h * 0.4)
        max_w = int(target_w * 0.4)

        new_h = int(h * scale)
        new_w = int(w * scale)

        if new_h > max_h:
            ratio = max_h / new_h
            new_h, new_w = max_h, int(new_w * ratio)
        if new_w > max_w:
            ratio = max_w / new_w
            new_w, new_h = max_w, int(new_h * ratio)

        new_h = max(new_h, 32)
        new_w = max(new_w, 32)

        obj_scaled = cv2.resize(
            obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        mask_scaled = cv2.resize(
            mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        return obj_scaled, mask_scaled

    def _create_feathered_mask(self, mask: np.ndarray) -> np.ndarray:
        """Crea maschera con bordi sfumati."""
        mask_float = mask.astype(np.float32) / 255.0

        if self.feather_radius > 0:
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(
                (mask_float * 255).astype(np.uint8), kernel, iterations=2
            )
            blurred = cv2.GaussianBlur(mask_float, (0, 0), self.feather_radius)
            mask_float = np.maximum(eroded.astype(np.float32) / 255.0, blurred)

        return mask_float

    def apply_cutpaste(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applica cut-paste a un'immagine.

        Args:
            image: Immagine RGB numpy array [H, W, 3]
            target_mask: Maschera numpy array [H, W]

        Returns:
            (image_modified, mask_modified)
        """
        img = image.copy()
        mask = target_mask.copy()

        num_objects = random.randint(self.min_objects, self.max_objects)
        selected = random.sample(self.objects, min(num_objects, len(self.objects)))

        img_h, img_w = img.shape[:2]

        for obj_info in selected:
            try:
                obj_crop, obj_mask = self._load_object(obj_info)
                obj_crop, obj_mask = self._scale_object(
                    obj_crop, obj_mask, img_h, img_w
                )

                obj_h, obj_w = obj_crop.shape[:2]

                # Posizione random
                pos_y = random.randint(0, max(0, img_h - obj_h))
                pos_x = random.randint(0, max(0, img_w - obj_w))

                # Calcola regione valida
                y1, x1 = max(0, pos_y), max(0, pos_x)
                y2, x2 = min(img_h, pos_y + obj_h), min(img_w, pos_x + obj_w)

                obj_y1, obj_x1 = y1 - pos_y, x1 - pos_x
                obj_y2, obj_x2 = obj_y1 + (y2 - y1), obj_x1 + (x2 - x1)

                obj_region = obj_crop[obj_y1:obj_y2, obj_x1:obj_x2]
                mask_region = obj_mask[obj_y1:obj_y2, obj_x1:obj_x2]
                img_region = img[y1:y2, x1:x2]

                # Blending
                if self.blend_mode == "alpha_feather":
                    alpha = self._create_feathered_mask(mask_region)[..., np.newaxis]
                else:
                    alpha = (mask_region > 127).astype(np.float32)[..., np.newaxis]

                blended = (obj_region * alpha + img_region * (1 - alpha)).astype(
                    np.uint8
                )
                img[y1:y2, x1:x2] = blended

                # Aggiorna maschera
                binary_mask = mask_region > 127
                mask[y1:y2, x1:x2][binary_mask] = self.ANOMALY_LABEL_ID

            except Exception as e:
                continue

        return img, mask


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
    Genera dataset Cityscapes con cut-paste pre-applicato.

    Args:
        cityscapes_path: Path alla cartella con i zip Cityscapes
        coco_ood_path: Path al subset COCO OOD
        output_path: Path dove salvare il nuovo dataset
        split: Split da processare ("train", "val")
        cutpaste_ratio: Percentuale di immagini con cut-paste (0-1)
        min_objects: Min oggetti da incollare
        max_objects: Max oggetti da incollare
        scale_range: Range scala oggetti
        blend_mode: Modalità blending
        seed: Seed per reproducibilità
    """
    random.seed(seed)
    np.random.seed(seed)

    cityscapes_path = Path(cityscapes_path)
    output_path = Path(output_path)

    # Crea cartelle output
    output_images = output_path / "leftImg8bit" / split
    output_masks = output_path / "gtFine" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    # Inizializza generator
    generator = CutPasteGenerator(
        coco_ood_path=coco_ood_path,
        min_objects=min_objects,
        max_objects=max_objects,
        scale_range=scale_range,
        blend_mode=blend_mode,
        seed=seed,
    )

    # Apri zip Cityscapes
    img_zip_path = cityscapes_path / "leftImg8bit_trainvaltest.zip"
    mask_zip_path = cityscapes_path / "gtFine_trainvaltest.zip"

    print(f"Aprendo {img_zip_path}...")
    img_zip = zipfile.ZipFile(img_zip_path, "r")
    mask_zip = zipfile.ZipFile(mask_zip_path, "r")

    # Trova tutte le immagini dello split
    img_prefix = f"leftImg8bit/{split}/"
    img_files = [
        f
        for f in img_zip.namelist()
        if f.startswith(img_prefix) and f.endswith("_leftImg8bit.png")
    ]

    print(f"Trovate {len(img_files)} immagini in {split}")

    stats = {"total": 0, "with_cutpaste": 0, "original": 0}
    metadata = []

    for img_path in tqdm(img_files, desc=f"Generando {split}"):
        # Estrai info
        # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        parts = img_path.split("/")
        city = parts[2]
        filename = parts[3]
        base_name = filename.replace("_leftImg8bit.png", "")

        # Path maschera corrispondente
        mask_path = f"gtFine/{split}/{city}/{base_name}_gtFine_labelIds.png"

        # Crea sottocartella città
        (output_images / city).mkdir(exist_ok=True)
        (output_masks / city).mkdir(exist_ok=True)

        # Carica immagine e maschera
        with img_zip.open(img_path) as f:
            img = np.array(Image.open(f).convert("RGB"))

        with mask_zip.open(mask_path) as f:
            mask = np.array(Image.open(f))

        # Decidi se applicare cut-paste
        apply_cutpaste = random.random() < cutpaste_ratio

        if apply_cutpaste:
            img_out, mask_out = generator.apply_cutpaste(img, mask)
            stats["with_cutpaste"] += 1
        else:
            img_out, mask_out = img, mask
            stats["original"] += 1

        stats["total"] += 1

        # Salva
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

    # Salva metadata
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

    print(f"\nDataset generato in {output_path}")
    print(f"  Totale: {stats['total']}")
    print(f"  Con cut-paste: {stats['with_cutpaste']}")
    print(f"  Originali: {stats['original']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset Cityscapes con Cut-Paste"
    )
    parser.add_argument(
        "--cityscapes_path",
        type=str,
        required=True,
        help="Path alla cartella con i zip Cityscapes",
    )
    parser.add_argument(
        "--coco_ood_path", type=str, required=True, help="Path al subset COCO OOD"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path dove salvare il nuovo dataset",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split da processare (train/val)"
    )
    parser.add_argument(
        "--cutpaste_ratio",
        type=float,
        default=0.5,
        help="Percentuale immagini con cut-paste (0-1)",
    )
    parser.add_argument(
        "--min_objects", type=int, default=1, help="Min oggetti da incollare"
    )
    parser.add_argument(
        "--max_objects", type=int, default=3, help="Max oggetti da incollare"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed per reproducibilità")

    args = parser.parse_args()

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
