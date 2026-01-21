# ---------------------------------------------------------------
# Cityscapes Semantic Dataset con Cut-Paste Augmentation
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms
from cutpaste import CutPasteAugmentation


class CityscapesSemanticCutPaste(LightningDataModule):
    """
    Dataset Cityscapes con augmentazione Cut-Paste per anomaly segmentation.
    
    Estende CityscapesSemantic aggiungendo la possibilità di incollare
    oggetti OOD da COCO sulle immagini di training.
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
        Args:
            path: Path al dataset Cityscapes (con i file zip)
            coco_ood_path: Path al subset COCO OOD scaricato
            num_workers: Numero di worker per dataloader
            batch_size: Batch size
            img_size: Dimensione immagine (H, W)
            num_classes: Numero classi (19 per Cityscapes + 1 anomalia opzionale)
            color_jitter_enabled: Abilita color jitter
            scale_range: Range scala per augmentation
            check_empty_targets: Controlla target vuoti
            cutpaste_enabled: Abilita cut-paste augmentation
            cutpaste_probability: Probabilità di applicare cut-paste (0-1)
            cutpaste_min_objects: Min oggetti da incollare
            cutpaste_max_objects: Max oggetti da incollare
            cutpaste_scale_range: Range scala oggetti OOD
            cutpaste_blend_mode: Modalità blending (paste, alpha_feather, poisson)
            cutpaste_feather_radius: Raggio sfumatura bordi
        """
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        
        self.coco_ood_path = coco_ood_path
        self.cutpaste_enabled = cutpaste_enabled
        self.cutpaste_probability = cutpaste_probability
        self.cutpaste_min_objects = cutpaste_min_objects
        self.cutpaste_max_objects = cutpaste_max_objects
        self.cutpaste_scale_range = cutpaste_scale_range
        self.cutpaste_blend_mode = cutpaste_blend_mode
        self.cutpaste_feather_radius = cutpaste_feather_radius

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        
        # Inizializza cut-paste augmentation
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
                print(f"Cut-Paste augmentation abilitata (prob={cutpaste_probability})")
            except Exception as e:
                print(f"Warning: impossibile inizializzare Cut-Paste: {e}")
                self.cutpaste_aug = None

    @staticmethod
    def target_parser(target, anomaly_label_id: int = 254, **kwargs):
        """
        Parser delle maschere che gestisce anche la classe anomalia.
        
        Args:
            target: Tensor della maschera
            anomaly_label_id: ID usato per i pixel anomali
        """
        masks, labels = [], []

        for label_id in target[0].unique():
            # Gestisci classe anomalia
            if label_id == anomaly_label_id:
                masks.append(target[0] == label_id)
                # Usa train_id speciale per anomalia (es. 19 se Cityscapes ha 0-18)
                labels.append(19)  # Classe anomalia
                continue
            
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

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
                t, 
                anomaly_label_id=CutPasteAugmentation.ANOMALY_LABEL_ID,
                **kw
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
        """Crea transforms per training."""
        return self.transforms

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class DatasetWithCutPaste(Dataset):
    """
    Estensione di Dataset che applica Cut-Paste augmentation.
    """
    
    def __init__(
        self,
        cutpaste_aug: Optional[CutPasteAugmentation] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cutpaste_aug = cutpaste_aug
    
    def __getitem__(self, index: int):
        """Override per applicare cut-paste prima delle altre transforms."""
        # Ottieni immagine e target dal parent
        img, target = self._load_item(index)
        
        # Applica cut-paste se abilitato
        if self.cutpaste_aug is not None:
            img, target = self.cutpaste_aug(img, target)
        
        # Applica transforms standard
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        # Parse target
        masks, labels, is_crowd = self.target_parser(target)
        
        return img, masks, labels, is_crowd
    
    def _load_item(self, index: int):
        """Carica immagine e target raw senza transforms."""
        import torch
        from PIL import Image
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as F
        
        img_info = self.imgs[index]
        target_info = self.targets[index]
        
        # Carica immagine
        with self._get_zip().open(img_info.filename) as img_file:
            img = Image.open(img_file).convert("RGB")
            img = F.to_image(img)
            img = F.to_dtype(img, torch.float32, scale=True)
        
        # Carica target
        target_zip = self._get_target_zip()
        with target_zip.open(target_info) as target_file:
            target = Image.open(target_file)
            target = tv_tensors.Mask(F.pil_to_tensor(target))
        
        return img, target
    
    def _get_zip(self):
        """Ottieni handle al file zip."""
        import zipfile
        from torch.utils.data import get_worker_info
        
        worker_info = get_worker_info()
        if worker_info is None or self.zip is None:
            if self.zip is None:
                self.zip = zipfile.ZipFile(self.zip_path, "r")
        return self.zip
    
    def _get_target_zip(self):
        """Ottieni handle al file zip dei target."""
        import zipfile
        from torch.utils.data import get_worker_info
        
        worker_info = get_worker_info()
        if worker_info is None or self.target_zip is None:
            if self.target_zip is None:
                target_path = self.target_zip_path or self.zip_path
                self.target_zip = zipfile.ZipFile(target_path, "r")
        return self.target_zip
