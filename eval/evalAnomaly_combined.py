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
        Resize((512, 1024), Image.NEAREST),
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
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Gestisce sia .ckpt (Lightning) che .pth (Pytorch puro)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

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


def get_pixel_scores(pred_logits, pred_masks):
    """
    Function to convert mask-based outputs to pixel-wise maps for metrics.
    Inspirato da evalAnomaly_cut_paste.py per confronto standardizzato.
    """
    # 1. Prepare Mask Probabilities (Clean Noise)
    mask_probs = F.sigmoid(pred_masks)
    mask_probs[mask_probs < 0.5] = 0.0

    # --- PART A: STANDARDIZED METRICS (FAIR COMPARISON 19 CLASSES) ---
    # Slicing solo i primi 19 logits (Cityscapes standard)
    logits_19 = pred_logits[:, :, :19]
    # Softmax solo su 19 classi per simulare comportamento senza anomalia esplicita
    probs_19 = F.softmax(logits_19, dim=-1)

    # Compute Semantic Probabilities (Standardized)
    sem_probs_std = torch.einsum("bqc,bqhw->bchw", probs_19, mask_probs)

    # Compute Logits (Standardized)
    pixel_logits_std = torch.einsum("bqc,bqhw->bchw", logits_19, mask_probs)

    # --- PART B: LEARNED ANOMALY METRIC ---
    # Softmax su tutte le 20 classi
    probs_20 = F.softmax(pred_logits, dim=-1)

    # Estrazione probabilità classe 19 (Anomalia)
    anomaly_prob = probs_20[:, :, 19:20]

    # Heatmap anomalia
    learned_anomaly_map = torch.einsum("bqc,bqhw->bchw", anomaly_prob, mask_probs)

    return sem_probs_std, pixel_logits_std, learned_anomaly_map


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern",
        required=True,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # Inizializzazione liste per metriche
    msp_list, maxLogit_list, entropy_list, rba_list, learned_list = [], [], [], [], []
    ood_gts_list = []

    if not os.path.exists("results_combined.txt"):
        open("results_combined.txt", "w").close()
    file = open("results_combined.txt", "a")

    # Caricamento Modello
    model = load_combined_model(args.ckpt, device)

    # LOOP SUI FILE
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(os.path.expanduser(pattern)))

    print(f"Starting evaluation on {len(input_files)} images...")

    for path in input_files:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(filename, end=" - ", flush=True)

        pil_img = Image.open(path).convert("RGB")
        images = input_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            mask_logits, class_logits = model(images)
            m_logits = mask_logits[-1]  # [B, Q, H, W]
            c_logits = class_logits[-1]  # [B, Q, C+1]

            # Upsample maschere a 512x1024 per confronto con GT
            m_logits_up = F.interpolate(
                m_logits, size=(512, 1024), mode="bilinear", align_corners=False
            )

            # Calcolo score (Logic di cut_paste)
            sem_probs_std, pixel_logits_std, learned_map = get_pixel_scores(
                c_logits, m_logits_up
            )

            sem_probs_np = sem_probs_std.squeeze(0).cpu().numpy()
            pixel_logits_np = pixel_logits_std.squeeze(0).cpu().numpy()
            learned_np = learned_map.squeeze(0).cpu().numpy()

            # Estrazione canali per score
            msp_score = 1 - np.max(sem_probs_np, axis=0)
            maxLogit_score = -np.max(pixel_logits_np, axis=0)
            entropy_score = -np.sum(sem_probs_np * np.log(sem_probs_np + 1e-8), axis=0)
            rba_score = 1 - np.sum(sem_probs_np, axis=0)
            learned_score = learned_np[0, :, :]

        # Gestione Ground Truth
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        if not os.path.exists(pathGT):
            print(f"GT not found for {path}")
            continue

        mask = Image.open(pathGT)
        mask = Resize((512, 1024), Image.NEAREST)(mask)
        ood_gts = np.array(mask)

        # Mapping etichette binary OOD
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            print("No anomaly in GT, skipping.")
            continue

        ood_gts_list.append(ood_gts)
        msp_list.append(msp_score)
        maxLogit_list.append(maxLogit_score)
        entropy_list.append(entropy_score)
        rba_list.append(rba_score)
        learned_list.append(learned_score)
        print("Done")

    # Metriche finali
    def evaluate_metric(score_list, gt_list, method_name):
        if len(score_list) == 0:
            return
        y_true = np.concatenate([gt.flatten() for gt in gt_list])
        y_score = np.concatenate([s.flatten() for s in score_list])

        valid_mask = y_true != 255
        y_true, y_score = y_true[valid_mask], y_score[valid_mask]

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fpr = fpr_at_95_tpr(y_score, y_true).item()

        res_str = f"[{method_name}] ROC AUC: {auc*100:.2f} | AUPRC: {ap*100:.2f} | FPR@95: {fpr*100:.2f}"
        print(res_str)
        file.write(res_str + "\n")

    if len(ood_gts_list) > 0:
        print("\n--- STANDARD METRICS (Normalized on 19 Classes) ---")
        evaluate_metric(msp_list, ood_gts_list, "MSP")
        evaluate_metric(maxLogit_list, ood_gts_list, "MaxLogit")
        evaluate_metric(entropy_list, ood_gts_list, "Entropy")
        evaluate_metric(rba_list, ood_gts_list, "RbA")

        print("\n--- NEW METRIC (Using Learned Anomaly Class) ---")
        evaluate_metric(learned_list, ood_gts_list, "Learned_Anomaly")
    else:
        print("No valid data for evaluation.")

    file.close()
