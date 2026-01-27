# =============================================================================
# Evaluation Script for EoMT Combined (LoRA + ArcFace + CutPaste)
# =============================================================================
# This script evaluates anomaly detection performance for a model trained
# with 20 classes (19 Cityscapes + 1 explicit Anomaly class).
# =============================================================================

import os
import cv2
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
import sys
import math

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
eomt_package_dir = osp.join(project_root, "eomt")

if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_package_dir not in sys.path:
    sys.path.append(eomt_package_dir)

# --- IMPORTS ---
from models.eomt import EoMT
from models.vit import ViT
from models.lora_integration import LoRAConfig

from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- REPRODUCIBILITY ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CONFIGURATION
NUM_CLASSES = 20  # 19 Cityscapes + 1 Anomaly
PATCH_SIZE = 16
IMG_SIZE = (640, 640)  # Match training resolution for accuracy

# --- TRANSFORMS ---
input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

target_transform = Compose(
    [
        Resize(IMG_SIZE, Image.NEAREST),
    ]
)


def load_combined_model(checkpoint_path, device):
    # 1. Initialize LoRA Config
    lora_config = LoRAConfig(
        enabled=True,
        rank=8,
        lora_alpha=16,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        freeze_base_model=True,
    )

    # 2. Initialize ViT Encoder
    encoder = ViT(
        img_size=IMG_SIZE, backbone_name="vit_base_patch14_reg4_dinov2", patch_size=16
    )

    # 3. Initialize EoMT
    model = EoMT(
        num_q=100,
        num_blocks=3,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        lora_config=lora_config,
    ).to(device)

    # 4. Load State Dict
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]

    # Clean keys (remove "network." prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("network."):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load Results: {msg}")
    model.eval()
    return model


def evaluate_anomaly(model, dataset_path, device):
    images = sorted(glob.glob(osp.join(dataset_path, "leftImg8bit/val/*/*.png")))
    # Adjust ground truth path logic based on your dataset structure
    # Standard: segmentation is in gtFine/val/*/*_gtFine_labelTrainIds.png

    anomaly_scores = []
    gt_masks = []

    print(f"Starting evaluation on {len(images)} images...")

    with torch.no_grad():
        for i, img_path in enumerate(images):
            # Load and transform image
            img = Image.open(img_path).convert("RGB")
            input_tensor = input_transform(img).unsqueeze(0).to(device)

            # Inference
            # EoMT returns (mask_logits_list, class_logits_list)
            mask_logits, class_logits = model(input_tensor)

            # Take last block outputs
            m_logits = mask_logits[-1]  # [B, Q, H, W]
            c_logits = class_logits[
                -1
            ]  # [B, Q, C+1] (C=20, so 21 total with no-object)

            # Map back to per-pixel
            # Probabilities for class 19 (Anomaly)
            c_probs = F.softmax(c_logits, dim=-1)  # [1, 100, 21]
            anomaly_query_probs = c_probs[
                0, :, 19
            ]  # [100] Prob of being anomaly for each query

            m_probs = torch.sigmoid(m_logits[0])  # [100, H, W]

            # Weighted average of masks based on anomaly query probability
            # Score(x,y) = sum_q ( Prob(q=Anomaly) * Prob(Pixel(x,y) belongs to q) )
            anomaly_map = torch.einsum("q,qhw->hw", anomaly_query_probs, m_probs)
            anomaly_map = anomaly_map.cpu().numpy()

            # Load Ground Truth
            gt_path = img_path.replace("leftImg8bit", "gtFine").replace(
                ".png", "_gtFine_labelTrainIds.png"
            )
            if not osp.exists(gt_path):
                continue

            gt = Image.open(gt_path)
            gt = np.array(target_transform(gt))

            # Create binary mask (assumes label 19 is anomaly in GT)
            binary_gt = (gt == 19).astype(np.uint8)

            if np.sum(binary_gt) > 0:  # Only evaluate images with anomalies
                anomaly_scores.append(anomaly_map.flatten())
                gt_masks.append(binary_gt.flatten())

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(images)}")

    # Calculate METRICS
    y_true = np.concatenate(gt_masks)
    y_score = np.concatenate(anomaly_scores)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fpr95 = fpr_at_95_tpr(y_score, y_true).item()

    print("\n" + "=" * 30)
    print(f"FINAL METRICS (Explicit Anomaly Class)")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Avg Precision: {ap:.4f}")
    print(f"FPR@95%TPR: {fpr95:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--data",
        type=str,
        default="/teamspace/studios/this_studio/datasets",
        help="Path to cityscapes",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_combined_model(args.ckpt, device)
    evaluate_anomaly(model, args.data, device)
