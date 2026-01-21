# ---------------------------------------------------------------
# Cut-Paste Augmentation per Anomaly Segmentation
# Incolla oggetti OOD da COCO su immagini Cityscapes
# ---------------------------------------------------------------

import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import torch
from torchvision import tv_tensors
import cv2


class CutPasteAugmentation:
    """
    Augmentazione Cut-Paste per creare anomalie sintetiche.
    
    Prende oggetti OOD dal subset COCO e li incolla su immagini Cityscapes,
    aggiornando la maschera con un ID speciale per l'anomalia.
    """
    
    # ID speciale per i pixel anomali nella maschera
    ANOMALY_LABEL_ID = 254
    
    def __init__(
        self,
        coco_ood_path: str,
        apply_probability: float = 0.5,
        min_objects: int = 1,
        max_objects: int = 3,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        blend_mode: str = "alpha_feather",  # "paste", "alpha_feather", "poisson"
        feather_radius: int = 5,
        avoid_classes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            coco_ood_path: Path al subset COCO OOD scaricato
            apply_probability: Probabilità di applicare cut-paste (0-1)
            min_objects: Numero minimo di oggetti da incollare
            max_objects: Numero massimo di oggetti da incollare
            scale_range: Range di scala per ridimensionare oggetti (min, max)
            blend_mode: Modalità blending ("paste", "alpha_feather", "poisson")
            feather_radius: Raggio sfumatura bordi per alpha_feather
            avoid_classes: Lista di class IDs Cityscapes da evitare per il paste
            seed: Seed per reproducibilità
        """
        self.coco_ood_path = Path(coco_ood_path)
        self.apply_probability = apply_probability
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.blend_mode = blend_mode
        self.feather_radius = feather_radius
        self.avoid_classes = avoid_classes or []
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Carica metadata COCO OOD
        self.metadata = self._load_metadata()
        self.objects = self.metadata.get("objects", [])
        
        if not self.objects:
            raise ValueError(f"Nessun oggetto OOD trovato in {coco_ood_path}")
        
        print(f"CutPaste: caricati {len(self.objects)} oggetti OOD")
    
    def _load_metadata(self) -> dict:
        """Carica metadata del subset COCO OOD."""
        metadata_path = self.coco_ood_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json non trovato in {self.coco_ood_path}. "
                "Esegui prima download_coco.py"
            )
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def _load_object(self, obj_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carica immagine e maschera di un oggetto OOD.
        
        Returns:
            Tuple (object_crop, mask_crop) come numpy arrays
        """
        # Carica immagine sorgente
        img_path = self.coco_ood_path / "images" / obj_info["image_filename"]
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Carica maschera
        mask_path = self.coco_ood_path / "masks" / obj_info["mask_filename"]
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Crop usando bbox
        x1, y1, x2, y2 = obj_info["bbox"]
        obj_crop = img[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        
        return obj_crop, mask_crop
    
    def _scale_object(
        self, 
        obj_crop: np.ndarray, 
        mask_crop: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ridimensiona oggetto con scala random."""
        h, w = obj_crop.shape[:2]
        target_h, target_w = target_size
        
        # Scala random
        scale = random.uniform(*self.scale_range)
        
        # Limita dimensione massima al 40% dell'immagine target
        max_h = int(target_h * 0.4)
        max_w = int(target_w * 0.4)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Clamp dimensioni
        if new_h > max_h:
            ratio = max_h / new_h
            new_h = max_h
            new_w = int(new_w * ratio)
        if new_w > max_w:
            ratio = max_w / new_w
            new_w = max_w
            new_h = int(new_h * ratio)
        
        # Dimensione minima
        new_h = max(new_h, 32)
        new_w = max(new_w, 32)
        
        obj_scaled = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return obj_scaled, mask_scaled
    
    def _find_paste_position(
        self,
        target_mask: np.ndarray,
        obj_h: int,
        obj_w: int,
        max_attempts: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """
        Trova posizione valida per incollare l'oggetto.
        Evita aree con classi specifiche se richiesto.
        
        Returns:
            Tuple (y, x) della posizione top-left, o None se non trovata
        """
        h, w = target_mask.shape[:2]
        
        for _ in range(max_attempts):
            # Posizione random
            y = random.randint(0, max(0, h - obj_h))
            x = random.randint(0, max(0, w - obj_w))
            
            # Controlla se posizione valida
            if self.avoid_classes:
                region = target_mask[y:y+obj_h, x:x+obj_w]
                unique_classes = np.unique(region)
                if any(c in self.avoid_classes for c in unique_classes):
                    continue
            
            return y, x
        
        # Fallback: posizione centrale
        y = (h - obj_h) // 2
        x = (w - obj_w) // 2
        return max(0, y), max(0, x)
    
    def _create_feathered_mask(self, mask: np.ndarray) -> np.ndarray:
        """Crea maschera con bordi sfumati per blending."""
        # Converti a float
        mask_float = mask.astype(np.float32) / 255.0
        
        # Applica blur gaussiano ai bordi
        if self.feather_radius > 0:
            # Erode per creare core solido
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode((mask_float * 255).astype(np.uint8), kernel, iterations=2)
            
            # Blur della maschera originale
            blurred = cv2.GaussianBlur(mask_float, (0, 0), self.feather_radius)
            
            # Combina: core solido + bordi sfumati
            mask_float = np.maximum(eroded.astype(np.float32) / 255.0, blurred)
        
        return mask_float
    
    def _paste_object(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        obj_crop: np.ndarray,
        obj_mask: np.ndarray,
        pos_y: int,
        pos_x: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incolla oggetto sull'immagine e aggiorna maschera.
        
        Returns:
            Tuple (immagine_modificata, maschera_modificata)
        """
        obj_h, obj_w = obj_crop.shape[:2]
        img_h, img_w = image.shape[:2]
        
        # Calcola regione di overlap valida
        y1 = max(0, pos_y)
        x1 = max(0, pos_x)
        y2 = min(img_h, pos_y + obj_h)
        x2 = min(img_w, pos_x + obj_w)
        
        # Offset nell'oggetto
        obj_y1 = y1 - pos_y
        obj_x1 = x1 - pos_x
        obj_y2 = obj_y1 + (y2 - y1)
        obj_x2 = obj_x1 + (x2 - x1)
        
        # Estrai regioni
        obj_region = obj_crop[obj_y1:obj_y2, obj_x1:obj_x2]
        mask_region = obj_mask[obj_y1:obj_y2, obj_x1:obj_x2]
        img_region = image[y1:y2, x1:x2]
        
        if self.blend_mode == "paste":
            # Paste semplice
            alpha = (mask_region > 127).astype(np.float32)[..., np.newaxis]
        elif self.blend_mode == "alpha_feather":
            # Alpha blending con bordi sfumati
            alpha = self._create_feathered_mask(mask_region)[..., np.newaxis]
        elif self.blend_mode == "poisson":
            # Poisson blending (più lento ma più naturale)
            try:
                center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
                mask_bin = (mask_region > 127).astype(np.uint8) * 255
                result = cv2.seamlessClone(
                    obj_region, image, mask_bin, center, cv2.NORMAL_CLONE
                )
                # Aggiorna solo maschera target
                binary_mask = mask_region > 127
                target_mask[y1:y2, x1:x2][binary_mask] = self.ANOMALY_LABEL_ID
                return result, target_mask
            except Exception:
                # Fallback a alpha feather
                alpha = self._create_feathered_mask(mask_region)[..., np.newaxis]
        else:
            alpha = (mask_region > 127).astype(np.float32)[..., np.newaxis]
        
        # Blend
        blended = (obj_region * alpha + img_region * (1 - alpha)).astype(np.uint8)
        image[y1:y2, x1:x2] = blended
        
        # Aggiorna maschera target con ID anomalia
        binary_mask = mask_region > 127
        target_mask[y1:y2, x1:x2][binary_mask] = self.ANOMALY_LABEL_ID
        
        return image, target_mask
    
    def __call__(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applica augmentazione cut-paste.
        
        Args:
            image: Immagine tensor [C, H, W] o [H, W, C]
            target: Maschera tensor [H, W] o [1, H, W]
            
        Returns:
            Tuple (immagine_augmentata, maschera_augmentata)
        """
        # Controlla probabilità
        if random.random() > self.apply_probability:
            return image, target
        
        # Converti a numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                # [C, H, W] -> [H, W, C]
                img_np = image.permute(1, 2, 0).numpy()
            else:
                img_np = image.numpy()
            was_tensor = True
        else:
            img_np = np.array(image)
            was_tensor = False
        
        if isinstance(target, torch.Tensor):
            if target.dim() == 3:
                target_np = target.squeeze(0).numpy()
            else:
                target_np = target.numpy()
        else:
            target_np = np.array(target)
        
        # Copia per modifiche
        img_np = img_np.copy()
        target_np = target_np.copy()
        
        # Numero random di oggetti
        num_objects = random.randint(self.min_objects, self.max_objects)
        
        # Seleziona oggetti random
        selected_objects = random.sample(self.objects, min(num_objects, len(self.objects)))
        
        for obj_info in selected_objects:
            try:
                # Carica oggetto
                obj_crop, obj_mask = self._load_object(obj_info)
                
                # Scala
                obj_crop, obj_mask = self._scale_object(
                    obj_crop, obj_mask, 
                    (img_np.shape[0], img_np.shape[1])
                )
                
                # Trova posizione
                pos = self._find_paste_position(
                    target_np, 
                    obj_crop.shape[0], 
                    obj_crop.shape[1]
                )
                
                if pos is None:
                    continue
                
                # Incolla
                img_np, target_np = self._paste_object(
                    img_np, target_np,
                    obj_crop, obj_mask,
                    pos[0], pos[1]
                )
                
            except Exception as e:
                # Skip oggetto problematico
                continue
        
        # Riconverti a tensor
        if was_tensor:
            # [H, W, C] -> [C, H, W]
            image_out = torch.from_numpy(img_np).permute(2, 0, 1)
            target_out = torch.from_numpy(target_np)
            if isinstance(target, tv_tensors.Mask):
                target_out = tv_tensors.Mask(target_out)
            return image_out, target_out
        else:
            return Image.fromarray(img_np), target_np

    def get_anomaly_label_id(self) -> int:
        """Ritorna l'ID usato per etichettare i pixel anomali."""
        return self.ANOMALY_LABEL_ID
