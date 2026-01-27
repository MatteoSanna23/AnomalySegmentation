# ---------------------------------------------------------------
# Cut-Paste Augmentation for Anomaly Segmentation
# ---------------------------------------------------------------

from .generate_cutpaste_dataset import CutPasteGenerator
from .download_coco import download_coco_subset

__all__ = ["CutPasteGenerator", "download_coco_subset"]
