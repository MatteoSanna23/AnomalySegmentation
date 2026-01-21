# ---------------------------------------------------------------
# Cut-Paste Augmentation for Anomaly Segmentation
# ---------------------------------------------------------------

from .cutpaste_augmentation import CutPasteAugmentation
from .download_coco import download_coco_subset

__all__ = ["CutPasteAugmentation", "download_coco_subset"]
