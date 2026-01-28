# Copyright (c) OpenMMLab. All rights reserved.
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
eomt_package_dir = os.path.join(project_root, "eomt")

if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_package_dir not in sys.path:
    sys.path.append(eomt_package_dir)

from models.eomt import EoMT
from models.vit import ViT

# Import custom OOD metrics
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- REPRODUCIBILITY ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19  # Standard Cityscapes classes (ID)

# Enable deterministic CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- TRANSFORMS ---
input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Standard ImageNet normalization (Required for DINOv2 backbone)
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


# =============================================================================
# UTILITY: RESIZE POSITIONAL EMBEDDINGS
# =============================================================================
def resize_pos_embed(state_dict, model, target_img_size=(512, 1024)):
    """
    Interpolates Vision Transformer positional embeddings to match
    the inference resolution (512x1024) if it differs from training.
    """
    if "encoder.backbone.pos_embed" in state_dict:
        pos_embed_checkpoint = state_dict["encoder.backbone.pos_embed"]
        embedding_dim = pos_embed_checkpoint.shape[-1]

        if hasattr(model.encoder.backbone, "pos_embed"):
            model_pos_embed = model.encoder.backbone.pos_embed
        else:
            model_pos_embed = torch.zeros(
                1,
                (target_img_size[0] // 16) * (target_img_size[1] // 16) + 1,
                embedding_dim,
            )

        num_patches_checkpoint = pos_embed_checkpoint.shape[1]
        grid_size_chk = int(math.sqrt(num_patches_checkpoint))

        # Reshape to 2D grid for spatial interpolation
        pos_embed_reshaped = pos_embed_checkpoint.transpose(1, 2).reshape(
            1, embedding_dim, grid_size_chk, grid_size_chk
        )

        patch_size = 16
        new_h, new_w = (
            target_img_size[0] // patch_size,
            target_img_size[1] // patch_size,
        )

        print(
            f"Resizing pos_embed from {grid_size_chk}x{grid_size_chk} to {new_h}x{new_w}"
        )

        new_pos_embed = F.interpolate(
            pos_embed_reshaped, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        new_pos_embed_spatial = new_pos_embed.flatten(2).transpose(1, 2)

        # Handle CLS Token concatenation
        target_len = model_pos_embed.shape[1]
        if target_len == new_pos_embed_spatial.shape[1] + 1:
            print("Adding CLS token to pos_embed (concatenation)")
            cls_pos_embed = model_pos_embed[:, 0:1, :]
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed_spatial), dim=1)
        else:
            new_pos_embed = new_pos_embed_spatial

        state_dict["encoder.backbone.pos_embed"] = new_pos_embed

    # Resize Attention Mask Probabilities if needed
    if "attn_mask_probs" in state_dict:
        amp = state_dict["attn_mask_probs"]
        if amp.shape[0] != model.attn_mask_probs.shape[0]:
            print(
                f"Adapting attn_mask_probs from {amp.shape} to {model.attn_mask_probs.shape}"
            )
            new_amp = F.interpolate(
                amp.view(1, 1, -1),
                size=model.attn_mask_probs.shape[0],
                mode="linear",
                align_corners=False,
            )
            state_dict["attn_mask_probs"] = new_amp.view(-1)

    return state_dict


# =============================================================================
# UTILITY: EXTRACT RAW LOGITS
# =============================================================================
def get_pixel_scores(pred_logits, pred_masks):
    """
    Extracts pixel-wise maps.
    Crucially returns 'pixel_logits' (unnormalized) to allow
    Temperature Scaling in the main loop.
    """
    # 1. Query probabilities (Softmax over classes)
    query_probs = F.softmax(pred_logits, dim=-1)

    # 2. Mask spatial probabilities (Sigmoid)
    mask_probs = F.sigmoid(pred_masks)

    # Remove the last class (void/no-object)
    valid_query_probs = query_probs[:, :, :NUM_CLASSES]
    valid_query_logits = pred_logits[:, :, :NUM_CLASSES]  # Raw logits

    # 3. Calculate Pixel-wise Semantic Probabilities (T=1.0 reference)
    sem_probs = torch.einsum("bqc,bqhw->bchw", valid_query_probs, mask_probs)

    # 4. Calculate Approximate Pixel-wise LOGITS
    # Weighted sum of logits: sum_q ( Logit(c|q) * P(pixel|q) )
    pixel_logits = torch.einsum("bqc,bqhw->bchw", valid_query_logits, mask_probs)

    return sem_probs, pixel_logits


# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern",
        required=True,
    )
    parser.add_argument(
        "--loadWeights", default="../trained_models/eomt_pretrained.pth"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # --- TEMPERATURE CONFIG ---
    # Define the temperatures to evaluate.
    # T < 1: Sharpens the distribution (increases confidence).
    # T > 1: Smoothens the distribution (decreases confidence, increases entropy).
    target_temps = [0.5, 0.75, 1.0, 1.1]

    # Storage for results per temperature
    msp_results = {t: [] for t in target_temps}
    ood_gts_list = []

    if not os.path.exists("results_eomt_temp.txt"):
        open("results_eomt_temp.txt", "w").close()
    file = open("results_eomt_temp.txt", "a")

    if not os.path.isfile(args.loadWeights):
        print(f"Weights file not found: {args.loadWeights}")
        sys.exit(1)

    print(f"Loading EoMT model weights from: {args.loadWeights}")

    # Model Setup
    encoder = ViT(
        img_size=(512, 1024),
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )
    num_queries = 100
    model = EoMT(encoder=encoder, num_q=num_queries, num_classes=NUM_CLASSES)

    if not args.cpu:
        model = model.cuda()

    # Load and Clean Weights
    checkpoint = torch.load(args.loadWeights, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name.replace("module.", "")
        if name.startswith("model."):
            name = name.replace("model.", "")
        if name.startswith("network."):
            name = name.replace("network.", "")
        new_state_dict[name] = v

    new_state_dict = resize_pos_embed(
        new_state_dict, model, target_img_size=(512, 1024)
    )
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # --- IMAGE PROCESSING ---
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(os.path.expanduser(pattern)))

    print(
        f"Starting inference on {len(input_files)} images with temperatures: {target_temps}"
    )

    for i, path in enumerate(input_files):
        if i % 10 == 0:
            print(f"Processing {i}/{len(input_files)}: {os.path.basename(path)}")

        # 1. Preprocess
        pil_img = Image.open(path).convert("RGB")
        images = input_transform(pil_img).unsqueeze(0)

        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                pred_masks = outputs[0][-1]
                pred_logits = outputs[1][-1]
            elif isinstance(outputs, dict):
                pred_masks = outputs["pred_masks"]
                pred_logits = outputs["pred_logits"]

            pred_masks = F.interpolate(
                pred_masks, size=(512, 1024), mode="bilinear", align_corners=False
            )

            # 2. Get Raw Pixel Logits
            _, pixel_logits = get_pixel_scores(pred_logits, pred_masks)

            # --- TEMPERATURE SCALING LOOP ---

            # We apply scaling here to avoid re-running the heavy model backbone
            for t in target_temps:
                # A. Scale Logits: L_scaled = L / T
                scaled_logits = pixel_logits / t

                # B. Apply Softmax on scaled logits
                scaled_probs = F.softmax(scaled_logits, dim=1)

                # C. Compute MSP (1 - Max Probability)
                # Ideally, OOD pixels will have a lower max probability (higher entropy)
                # after scaling if T is chosen correctly.
                scaled_probs_np = scaled_probs.squeeze(0).cpu().numpy()
                msp_score_t = 1 - np.max(scaled_probs_np, axis=0)

                msp_results[t].append(msp_score_t)

        # 3. Ground Truth Loading
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Label Mapping
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
            # Alignment maintenance: remove scores if image is skipped
            for t in target_temps:
                msp_results[t].pop()
            continue
        else:
            ood_gts_list.append(ood_gts)

        del outputs, pixel_logits, scaled_probs, ood_gts
        torch.cuda.empty_cache()

    file.write("\n")
    print("\n --- RESULTS FOR TEMPERATURE SCALING --- \n")

    # --- METRIC CALCULATION ---
    def evaluate_metric(score_list, gt_list, method_name):
        if len(score_list) == 0:
            print(f"No data to evaluate for {method_name}.")
            return 0.0, 0.0

        ood_gts = np.array(gt_list)
        anomaly_scores = np.array(score_list)

        ood_mask = ood_gts == 1
        ind_mask = ood_gts == 0

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        res_str = (
            f"[{method_name}] AUPRC: {prc_auc*100.0:.2f} | FPR@95: {fpr*100.0:.2f}"
        )
        print(res_str)
        file.write(res_str + "\n")
        return prc_auc, fpr

    # --- PRINT SUMMARY ---

    best_auprc = 0.0
    best_t = 1.0

    for t in target_temps:
        method_name = f"MSP (T={t})"
        auprc, fpr = evaluate_metric(msp_results[t], ood_gts_list, method_name)

        if auprc > best_auprc:
            best_auprc = auprc
            best_t = t

    print(
        f"\n>>> BEST TEMPERATURE FOUND: T={best_t} with AUPRC: {best_auprc*100.0:.2f}"
    )
    file.write(f"Best T: {best_t} (AUPRC: {best_auprc*100.0:.2f})\n")

    file.close()


if __name__ == "__main__":
    main()
