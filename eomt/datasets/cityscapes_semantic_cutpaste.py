# ---------------------------------------------------------------
# Dataset Cityscapes con Cut-Paste pre-generato
# Legge da cartella invece che da zip
# ---------------------------------------------------------------

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


# ID anomalia usato nel dataset pre-generato
ANOMALY_LABEL_ID = 254


class CityscapesSemanticCutPaste(LightningDataModule):
    """
    Dataset Cityscapes con Cut-Paste pre-generato.

    Legge da una cartella con immagini già processate.
    """

    def __init__(
        self,
        path: str,  # Path al dataset pre-generato
        original_cityscapes_path: str = None,  # Path ai zip originali per validation
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 20,  # 19 + anomalia
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

        self.original_cityscapes_path = original_cityscapes_path or path

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        """Parser maschere con supporto anomalia."""
        masks, labels = [], []

        for label_id in target[0].unique():
            # Classe anomalia
            if label_id == ANOMALY_LABEL_ID:
                masks.append(target[0] == label_id)
                labels.append(19)
                continue

            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Dataset training da cartella pre-generata
        self.train_dataset = CutPasteFolderDataset(
            root_path=Path(self.path),
            split="train",
            transforms=self.transforms,
            target_parser=self.target_parser,
        )

        # Dataset validation dagli zip originali (senza cut-paste)
        val_path = Path(self.original_cityscapes_path)

        # Se la cartella val esiste nel dataset pre-generato, usala
        if (Path(self.path) / "leftImg8bit" / "val").exists():
            self.val_dataset = CutPasteFolderDataset(
                root_path=Path(self.path),
                split="val",
                transforms=None,  # No augmentation per validation
                target_parser=self.target_parser,
            )
        else:
            # Altrimenti usa i zip originali
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
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class CutPasteFolderDataset(TorchDataset):
    """
    Dataset che legge immagini e maschere da cartella.
    Struttura attesa:
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

        # Trova tutte le immagini
        img_dir = root_path / "leftImg8bit" / split
        self.samples = []

        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                base_name = img_path.stem.replace("_leftImg8bit", "")
                mask_path = (
                    root_path
                    / "gtFine"
                    / split
                    / city_dir.name
                    / f"{base_name}_gtFine_labelIds.png"
                )

                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        print(f"CutPasteFolderDataset: trovate {len(self.samples)} immagini in {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, mask_path = self.samples[index]

        # Carica immagine
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        # Carica maschera
        mask = tv_tensors.Mask(Image.open(mask_path), dtype=torch.long)

        # Resize se necessario
        if img.shape[-2:] != mask.shape[-2:]:
            mask = F.resize(
                mask,
                list(img.shape[-2:]),
                interpolation=F.InterpolationMode.NEAREST,
            )

        # Parse target
        masks, labels, is_crowd = self.target_parser(mask)

        # Gestisci caso vuoto
        if len(masks) == 0:
            h, w = img.shape[-2:]
            masks = [torch.zeros((h, w), dtype=torch.bool)]
            labels = [255]
            is_crowd = [False]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        # Applica transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target