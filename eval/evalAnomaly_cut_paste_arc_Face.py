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
# Dynamically add project root and 'eomt' package to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
eomt_package_dir = os.path.join(project_root, "eomt")

if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_package_dir not in sys.path:
    sys.path.append(eomt_package_dir)

from models.eomt import EoMT
from models.vit import ViT

# Import custom metrics for Out-of-Distribution (OOD) detection
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- REPRODUCIBILITY SETUP ---
# Set seeds to ensure deterministic results across runs
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20  # 19 Standard Cityscapes classes + 1 Learned Anomaly class

# Enable deterministic CuDNN algorithms for consistency
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- DATA TRANSFORMS ---

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Standard ImageNet normalization.
        # CRITICAL: Do not remove this, as the pre-trained backbone (DINOv2/ViT) expects this distribution.
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Resize Ground Truth masks using Nearest Neighbor to preserve integer class labels
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
    Resizes the positional embeddings in a checkpoint to match the current model's resolution.

    Vision Transformers use fixed-size positional embeddings. If the inference resolution
    differs from training (e.g., 512x1024 vs 224x224), we must interpolate the embeddings.
    """
    if "encoder.backbone.pos_embed" in state_dict:
        pos_embed_checkpoint = state_dict[
            "encoder.backbone.pos_embed"
        ]  # Shape: [1, num_patches, embed_dim]
        embedding_dim = pos_embed_checkpoint.shape[-1]

        # Get current model's embedding placeholder
        if hasattr(model.encoder.backbone, "pos_embed"):
            model_pos_embed = model.encoder.backbone.pos_embed
        else:
            # Fallback calculation if attribute is missing
            model_pos_embed = torch.zeros(
                1,
                (target_img_size[0] // 16) * (target_img_size[1] // 16) + 1,
                embedding_dim,
            )

        num_patches_checkpoint = pos_embed_checkpoint.shape[1]

        # Calculate original grid size (assuming square training images, e.g., 64x64 patches)
        grid_size_chk = int(math.sqrt(num_patches_checkpoint))

        # 1. Reshape 1D sequence to 2D spatial grid [1, Dim, H, W] for interpolation
        pos_embed_reshaped = pos_embed_checkpoint.transpose(1, 2).reshape(
            1, embedding_dim, grid_size_chk, grid_size_chk
        )

        # 2. Calculate target grid size based on patch size (16)
        patch_size = 16
        new_h, new_w = (
            target_img_size[0] // patch_size,
            target_img_size[1] // patch_size,
        )

        print(
            f"Resizing pos_embed from {grid_size_chk}x{grid_size_chk} to {new_h}x{new_w}"
        )

        # 3. Perform Bicubic Interpolation
        new_pos_embed = F.interpolate(
            pos_embed_reshaped, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        # 4. Flatten back to 1D sequence [1, N, Dim]
        new_pos_embed_spatial = new_pos_embed.flatten(2).transpose(1, 2)

        # 5. Handle CLS Token
        # If the model expects N+1 tokens (1 CLS + N Patches), we prepend the CLS token
        # from the current model (or checkpoint if available).
        target_len = model_pos_embed.shape[1]
        if target_len == new_pos_embed_spatial.shape[1] + 1:
            print("Adding CLS token to pos_embed (concatenation)")
            cls_pos_embed = model_pos_embed[:, 0:1, :]
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed_spatial), dim=1)
        else:
            new_pos_embed = new_pos_embed_spatial

        state_dict["encoder.backbone.pos_embed"] = new_pos_embed

    # Resize Attention Mask Probabilities (for annealing schedule) if levels differ
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
# UTILITY: COMPUTE PIXEL-WISE ANOMALY MAPS
# =============================================================================
def get_pixel_scores(pred_logits, pred_masks):
    """
    Converts mask transformer outputs (segment scores + binary masks) into dense pixel-wise maps.

    Returns 3 types of maps:
    1. Semantic Probabilities (Standardized on 19 classes)
    2. Logits (Standardized on 19 classes)
    3. Learned Anomaly Map (The specific contribution of our fine-tuning)
    """

    # Clean up binary mask predictions
    mask_probs = F.sigmoid(pred_masks)
    mask_probs[mask_probs < 0.5] = 0.0

    # --- PART A: STANDARDIZED METRICS (19 Classes) ---
    # We purposefully ignore the 20th class here to compare fairly with standard methods
    # (like MSP/MaxLogit) that only know about the 19 Cityscapes classes.
    logits_19 = pred_logits[:, :, :19]

    # Softmax over 19 classes ensures they sum to 1, treating 'anomaly' as uncertainty
    probs_19 = F.softmax(logits_19, dim=-1)

    # Project segment scores to pixel space using Einstein summation
    # b=batch, q=queries, c=classes, h=height, w=width
    sem_probs_std = torch.einsum("bqc,bqhw->bchw", probs_19, mask_probs)
    pixel_logits_std = torch.einsum("bqc,bqhw->bchw", logits_19, mask_probs)

    # --- PART B: LEARNED ANOMALY METRIC (20 Classes) ---
    # Here we use the full power of the model. We softmax over all 20 classes.
    probs_20 = F.softmax(pred_logits, dim=-1)

    # Extract specifically the 20th channel (Index 19) which represents "Anomaly"
    anomaly_prob = probs_20[:, :, 19:20]

    # Project the anomaly score to pixels
    learned_anomaly_map = torch.einsum("bqc,bqhw->bchw", anomaly_prob, mask_probs)

    return sem_probs_std, pixel_logits_std, learned_anomaly_map


# =============================================================================
# MAIN EVALUATION LOOP
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

    # Lists to store scores for final metric calculation
    msp_list = []
    maxLogit_list = []
    entropy_list = []
    rba_list = []
    learned_list = []  # New list for the learned anomaly score
    ood_gts_list = []  # Ground Truth labels (0=In-Dist, 1=Anomaly)

    # Initialize results file
    if not os.path.exists("results_eomt.txt"):
        open("results_eomt.txt", "w").close()
    file = open("results_eomt.txt", "a")

    if not os.path.isfile(args.loadWeights):
        print(f"Weights file not found: {args.loadWeights}")
        sys.exit(1)

    print(f"Loading EoMT model weights from: {args.loadWeights}")

    # --- MODEL INITIALIZATION ---
    encoder = ViT(
        img_size=(512, 1024),
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )
    # Using 100 queries as per standard Mask2Former/EoMT config
    num_queries = 100
    model = EoMT(encoder=encoder, num_q=num_queries, num_classes=NUM_CLASSES)

    if not args.cpu:
        model = model.cuda()

    # --- WEIGHT LOADING & CLEANING ---
    checkpoint = torch.load(args.loadWeights, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Remove DDP/Lightning prefixes (module., model., network.) to match model keys
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

    # Resize embeddings to match inference resolution
    new_state_dict = resize_pos_embed(
        new_state_dict, model, target_img_size=(512, 1024)
    )

    # Load state dict (strict=False allows flexibility for minor mismatches)
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(
        f"Weights loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}"
    )

    model.eval()

    # --- PROCESSING IMAGES ---
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(os.path.expanduser(pattern)))

    for path in input_files:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(filename, end=" - ", flush=True)

        # 1. Preprocess Image
        pil_img = Image.open(path).convert("RGB")
        images = input_transform(pil_img).unsqueeze(0)

        if not args.cpu:
            images = images.cuda()

        # 2. Inference
        with torch.no_grad():
            outputs = model(images)

            # Handle different output formats (tuple/list vs dict)
            if isinstance(outputs, (tuple, list)):
                # EoMT usually returns a list of outputs for each decoder layer; take the last one
                pred_masks = outputs[0][-1]
                pred_logits = outputs[1][-1]
            elif isinstance(outputs, dict):
                pred_masks = outputs["pred_masks"]
                pred_logits = outputs["pred_logits"]
            else:
                raise TypeError(f"Unexpected output format: {type(outputs)}")

            # Upsample low-res mask predictions to target resolution
            pred_masks = F.interpolate(
                pred_masks, size=(512, 1024), mode="bilinear", align_corners=False
            )

            # 3. Compute Anomaly Maps
            sem_probs_std, pixel_logits_std, learned_map = get_pixel_scores(
                pred_logits, pred_masks
            )

            # Squeeze batch dimension for numpy conversion
            sem_probs_np = sem_probs_std.squeeze(0).cpu().numpy()
            pixel_logits_np = pixel_logits_std.squeeze(0).cpu().numpy()
            learned_np = learned_map.squeeze(0).cpu().numpy()

            # --- METRIC 1: MSP (Maximum Softmax Probability) ---
            # Anomaly Score = 1 - Max probability of any known class
            msp_score = 1 - np.max(sem_probs_np, axis=0)

            # --- METRIC 2: MaxLogit ---
            # Anomaly Score = - Max logit of any known class (Logits are unnormalized)
            maxLogit_score = -np.max(pixel_logits_np, axis=0)

            # --- METRIC 3: Entropy ---
            # Measures uncertainty of the distribution over known classes
            entropy_score = -np.sum(sem_probs_np * np.log(sem_probs_np + 1e-8), axis=0)

            # --- METRIC 4: RbA (Residual-based Anomaly) ---
            # Usually 1 - sum(probs), but since we renormalized, this might be low. kept for legacy.
            rba_score = 1 - np.sum(sem_probs_np, axis=0)

            # --- METRIC 5: Learned Anomaly ---
            # Direct probability of the anomaly class (Class Index 19)
            learned_score = learned_np[0, :, :]

        # 4. Load Ground Truth (GT) and Normalize Labels
        pathGT = path.replace("images", "labels_masks")
        # Handle dataset-specific extensions
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Normalize diverse dataset labels to Binary: 0 = In-Distribution, 1 = Anomaly
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)  # Ignore background
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)  # Road -> ID
            ood_gts = np.where(
                (ood_gts > 1) & (ood_gts < 201), 1, ood_gts
            )  # Objects -> OOD
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Skip images with no valid anomaly pixels if necessary
        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            msp_list.append(msp_score)
            maxLogit_list.append(maxLogit_score)
            entropy_list.append(entropy_score)
            rba_list.append(rba_score)
            learned_list.append(learned_score)

        # Clear memory
        del outputs, sem_probs_std, msp_score, maxLogit_score, learned_score, ood_gts
        torch.cuda.empty_cache()

    file.write("\n")
    print("\n")

    # --- FINAL METRIC EVALUATION HELPER ---
    def evaluate_metric(score_list, gt_list, method_name):
        """Calculates AUPRC and FPR@95 using flattened arrays."""
        if len(score_list) == 0:
            print(f"No data to evaluate for {method_name}.")
            return

        ood_gts = np.array(gt_list)
        anomaly_scores = np.array(score_list)

        # Flatten arrays based on mask (0 vs 1)
        ood_mask = ood_gts == 1
        ind_mask = ood_gts == 0

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        # Create binary labels for sklearn
        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        # Calculate metrics
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        res_str = (
            f"[{method_name}] AUPRC: {prc_auc*100.0:.2f} | FPR@95: {fpr*100.0:.2f}"
        )
        print(res_str)
        file.write(res_str + "\n")

    # --- PRINT RESULTS ---
    print("--- STANDARD METRICS (Normalized on 19 Classes) ---")
    evaluate_metric(msp_list, ood_gts_list, "MSP")
    evaluate_metric(maxLogit_list, ood_gts_list, "MaxLogit")
    evaluate_metric(entropy_list, ood_gts_list, "Entropy")
    evaluate_metric(rba_list, ood_gts_list, "RbA")

    print("--- NEW METRIC (Using Learned Anomaly Class) ---")
    evaluate_metric(learned_list, ood_gts_list, "Learned_Anomaly")

    file.close()


if __name__ == "__main__":
    main()
