# =============================================================================
# Evaluation Script for EoMT Combined (LoRA + ArcFace + CutPaste)
# =============================================================================
# This script evaluates the anomaly detection performance of a vision model
# trained with 20 classes (19 standard Cityscapes classes + 1 explicit Anomaly class).
# It computes metrics such as ROC AUC, AUPRC, and FPR@95.
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
# Determine the absolute path of the current file and project root
# to correctly import custom modules (eomt, models, etc.).
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
# Set seeds for random number generators to ensure consistent results across runs.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# --- CONFIGURATION ---
NUM_CLASSES = 20  # 19 semantic classes (Cityscapes) + 1 explicit Anomaly class
PATCH_SIZE = 16  # Patch size used by the Vision Transformer (ViT)
IMG_SIZE = (
    512,
    1024,
)  # Fixed input resolution for evaluation to ensure metric consistency

# --- TRANSFORMS ---
# Preprocessing for input images: Resize -> Tensor conversion -> Normalization (ImageNet stats)
input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Preprocessing for ground truth masks: Resize only (Nearest Neighbor to preserve label integers)
target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def get_pixel_scores(pred_logits, pred_masks):
    """
    Converts mask-based model outputs into pixel-wise probability maps for evaluation.

    Args:
        pred_logits: Class logits for each mask [Batch, Num_Queries, Num_Classes]
        pred_masks:  Binary mask predictions [Batch, Num_Queries, H, W]

    Returns:
        sem_probs_std:    Standardized semantic probabilities (19 classes).
        pixel_logits_std: Standardized pixel logits (19 classes).
        learned_anomaly_map: Probability map for the 20th class (Anomaly).
    """
    # 1. Prepare Mask Probabilities
    # Apply sigmoid to convert mask logits to probabilities [0, 1]
    mask_probs = F.sigmoid(pred_masks)
    # Filter weak mask predictions to reduce noise
    mask_probs[mask_probs < 0.5] = 0.0

    # --- PART A: STANDARDIZED METRICS (FAIR COMPARISON 19 CLASSES) ---
    # Slice the first 19 logits (standard Cityscapes classes), ignoring the anomaly class.
    logits_19 = pred_logits[:, :, :19]

    # Apply Softmax only on the 19 classes.
    # This simulates how a standard model would behave without the explicit anomaly head.
    probs_19 = F.softmax(logits_19, dim=-1)

    # Compute Semantic Probabilities (Standardized)
    # Einstein summation combines class probabilities (bqc) with spatial masks (bqhw)
    # Result: [Batch, Classes, Height, Width]
    sem_probs_std = torch.einsum("bqc,bqhw->bchw", probs_19, mask_probs)

    # Compute Logits (Standardized) - used for MaxLogit score
    pixel_logits_std = torch.einsum("bqc,bqhw->bchw", logits_19, mask_probs)

    # --- PART B: LEARNED ANOMALY METRIC ---
    # Apply Softmax across ALL 20 classes (including the anomaly class)
    probs_20 = F.softmax(pred_logits, dim=-1)

    # Extract the probability of the 20th class (Index 19 -> Anomaly)
    anomaly_prob = probs_20[:, :, 19:20]

    # Generate the heatmap specifically for the anomaly class
    learned_anomaly_map = torch.einsum("bqc,bqhw->bchw", anomaly_prob, mask_probs)

    return sem_probs_std, pixel_logits_std, learned_anomaly_map


def resize_pos_embed(state_dict, model, target_img_size=(512, 1024)):
    """
    Resizes the positional embeddings in the checkpoint to match the current model's resolution.

    Vision Transformers (ViT) learn positional embeddings for a specific grid size (e.g., 224x224).
    If we evaluate at a different resolution (e.g., 512x1024), we must interpolate the
    embeddings to fit the new grid dimensions.

    Args:
        state_dict: The dictionary containing loaded weights.
        model: The initialized model instance (source of target configuration).
        target_img_size: The desired resolution (H, W).

    Returns:
        state_dict: The updated dictionary with resized positional embeddings.
    """
    if "encoder.backbone.pos_embed" in state_dict:
        pos_embed_checkpoint = state_dict[
            "encoder.backbone.pos_embed"
        ]  # [1, num_patches, embed_dim]
        embedding_dim = pos_embed_checkpoint.shape[-1]

        # Determine the target embedding shape based on the current model configuration
        if hasattr(model.encoder.backbone, "pos_embed"):
            model_pos_embed = model.encoder.backbone.pos_embed
        else:
            # Fallback calculation if model attribute is missing
            model_pos_embed = torch.zeros(
                1,
                (target_img_size[0] // 16) * (target_img_size[1] // 16),
                embedding_dim,
            )

        # Calculate original grid size from the checkpoint (assuming square training image)
        num_patches_checkpoint = pos_embed_checkpoint.shape[1]
        grid_size_chk = int(math.sqrt(num_patches_checkpoint))

        # Reshape embedding from flat [1, N, Dim] to 2D spatial [1, Dim, Grid, Grid] for interpolation
        pos_embed_reshaped = pos_embed_checkpoint.transpose(1, 2).reshape(
            1, embedding_dim, grid_size_chk, grid_size_chk
        )

        # Calculate new grid dimensions based on patch size (16)
        patch_size = 16
        new_h, new_w = (
            target_img_size[0] // patch_size,
            target_img_size[1] // patch_size,
        )

        print(
            f"Resizing pos_embed from {grid_size_chk}x{grid_size_chk} to {new_h}x{new_w}"
        )

        # Perform bicubic interpolation to resize the grid
        new_pos_embed = F.interpolate(
            pos_embed_reshaped, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        # Flatten back to sequence format: [1, Dim, H, W] -> [1, New_N, Dim]
        new_pos_embed_spatial = new_pos_embed.flatten(2).transpose(1, 2)

        # Handle Class Token (CLS) if present
        # Some ViTs prepend a CLS token to the sequence. We must preserve it and concat it back.
        target_len = model_pos_embed.shape[1]
        if target_len == new_pos_embed_spatial.shape[1] + 1:
            print("Adding CLS token to pos_embed (concatenation)")
            cls_pos_embed = model_pos_embed[:, 0:1, :]
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed_spatial), dim=1)
        else:
            new_pos_embed = new_pos_embed_spatial

        # Update the state dict
        state_dict["encoder.backbone.pos_embed"] = new_pos_embed
    return state_dict


def load_combined_model(checkpoint_path, device):
    """
    Initializes the model architecture, applies LoRA configuration, and loads weights.

    Args:
        checkpoint_path: Path to the .ckpt or .pth file.
        device: 'cuda' or 'cpu'.

    Returns:
        model: The model in evaluation mode.
    """
    # 1. Initialize LoRA (Low-Rank Adaptation) Configuration
    # This enables efficient fine-tuning/inference by freezing the base model
    # and only training low-rank matrices in specific layers (qkv, proj, fc).
    lora_config = LoRAConfig(
        enabled=True,
        rank=16,
        lora_alpha=16,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        freeze_base_model=True,
    )

    # 2. Initialize ViT Encoder (Backbone)
    encoder = ViT(
        img_size=IMG_SIZE, backbone_name="vit_base_patch14_reg4_dinov2", patch_size=16
    )

    # 3. Initialize EoMT (Encoder-only Mask Transformer)
    # This acts as the decoder/head of the architecture.
    model = EoMT(
        num_q=100,  # Number of queries
        num_blocks=3,  # Number of transformer blocks in decoder
        num_classes=NUM_CLASSES,
        encoder=encoder,
        lora_config=lora_config,
    ).to(device)

    # 4. Load State Dict
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state_dict: handles both Lightning checkpoints (.ckpt) and standard PyTorch (.pth)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Clean keys: remove "network." prefix if it exists (common in Lightning modules)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("network."):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v

    # Resize positional embedding to match current resolution
    new_state_dict = resize_pos_embed(new_state_dict, model, target_img_size=IMG_SIZE)

    # Load weights into the model (strict=False allows missing/unexpected keys if compatible)
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load Results: {msg}")

    model.eval()
    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern (e.g., 'dataset/*.jpg')",
        required=True,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    # Set computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # Initialize lists to store metric scores for final evaluation
    msp_list, maxLogit_list, entropy_list, rba_list, learned_list = [], [], [], [], []
    ood_gts_list = []

    # Initialize results file
    if not os.path.exists("results_combined.txt"):
        open("results_combined.txt", "w").close()
    file = open("results_combined.txt", "a")

    # Load the Model
    model = load_combined_model(args.ckpt, device)

    # EXPAND FILE LIST (Globbing)
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(os.path.expanduser(pattern)))

    print(f"Starting evaluation on {len(input_files)} images...")

    # --- MAIN INFERENCE LOOP ---
    for path in input_files:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(filename, end=" - ", flush=True)

        # Load and Preprocess Image
        pil_img = Image.open(path).convert("RGB")
        images = input_transform(pil_img).unsqueeze(0).to(device)

        # Perform Inference
        with torch.no_grad():
            mask_logits, class_logits = model(images)
            m_logits = mask_logits[
                -1
            ]  # Get output from last decoder layer [B, Q, H, W]
            c_logits = class_logits[
                -1
            ]  # Get output from last decoder layer [B, Q, C+1]

            # Calculate Scores (Standard metrics + Learned Anomaly)
            sem_probs_std, pixel_logits_std, learned_map = get_pixel_scores(
                c_logits, m_logits
            )

            # Convert tensors to numpy for metric calculation
            sem_probs_np = sem_probs_std.squeeze(0).cpu().numpy()
            pixel_logits_np = pixel_logits_std.squeeze(0).cpu().numpy()
            learned_np = learned_map.squeeze(0).cpu().numpy()

            # --- CALCULATE ANOMALY SCORES ---
            # MSP (Maximum Softmax Probability): 1 - max probability (High confidence = Low anomaly)
            msp_score = 1 - np.max(sem_probs_np, axis=0)

            # MaxLogit: Negative max logit value
            maxLogit_score = -np.max(pixel_logits_np, axis=0)

            # Entropy: Uncertainty measure (Higher entropy = Higher anomaly probability)
            entropy_score = -np.sum(sem_probs_np * np.log(sem_probs_np + 1e-8), axis=0)

            # RbA (Residual based Anomaly): 1 - sum of probabilities
            rba_score = 1 - np.sum(sem_probs_np, axis=0)

            # Learned Score: The explicit output of the 20th class
            learned_score = learned_np[0, :, :]

        # --- GROUND TRUTH HANDLING (Initial Path Construction) ---
        pathGT = path.replace("images", "labels_masks")
        # Adjust extensions based on specific dataset conventions
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        if not os.path.exists(pathGT):
            print(f"GT not found for {path}")
            continue

        # Load GT Mask
        mask = Image.open(pathGT)
        mask = Resize((512, 1024), Image.NEAREST)(mask)
        ood_gts = np.array(mask)

        # --- GT LABEL MAPPING ---
        # Map dataset-specific labels to binary anomaly mask:
        # 0 = In-Distribution, 1 = Anomaly, 255 = Ignore/Void

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)

        if "LostAndFound" in pathGT:
            # Remap specific labels for LostAndFound dataset
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)  # Ignore background 0
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)  # Safe road
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)  # Anomalies

        if "Streethazard" in pathGT:
            # Remap labels for StreetHazard
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)  # Car mount/Ignore
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)  # Standard classes
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)  # Anomalies

        # --- RESIZING SCORES TO MATCH GT ---
        # Ensure predictions match the Ground Truth resolution (if different)
        gt_shape = ood_gts.shape

        def resize_if_needed(arr):
            """Upsamples the score map if it doesn't match GT dimensions."""
            if arr.shape != gt_shape:
                arr_img = Image.fromarray(arr.astype(np.float32))
                arr_resized = arr_img.resize(
                    (gt_shape[1], gt_shape[0]), resample=Image.BILINEAR
                )
                return np.array(arr_resized)
            return arr

        # Apply resizing to all score maps
        msp_score = resize_if_needed(msp_score)
        maxLogit_score = resize_if_needed(maxLogit_score)
        entropy_score = resize_if_needed(entropy_score)
        rba_score = resize_if_needed(rba_score)
        learned_score = resize_if_needed(learned_score)

        # --- GROUND TRUTH RELOAD AND VERIFICATION ---
        # Note: The code below re-executes the path logic and loading logic done above.
        # This acts as a secondary check or a redundant block to ensure variables are fresh.
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

        # Re-apply label mapping (Identical logic to lines 199-215)
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

        # Skip images that do not contain any anomaly (Optional filter)
        if 1 not in np.unique(ood_gts):
            print("No anomaly in GT, skipping.")
            continue

        # Append results to lists for batch evaluation
        ood_gts_list.append(ood_gts)
        msp_list.append(msp_score)
        maxLogit_list.append(maxLogit_score)
        entropy_list.append(entropy_score)
        rba_list.append(rba_score)
        learned_list.append(learned_score)
        print("Done")

    # --- FINAL METRIC CALCULATION ---
    def evaluate_metric(score_list, gt_list, method_name):
        """
        Calculates and prints AUC, AUPRC, and FPR@95.

        Args:
            score_list: List of numpy arrays containing anomaly scores.
            gt_list: List of numpy arrays containing ground truth masks.
            method_name: Name of the metric (e.g., "MSP", "Entropy").
        """
        if len(score_list) == 0:
            return
        # Flatten all arrays into a single 1D vector for global metric calculation
        y_true = np.concatenate([gt.flatten() for gt in gt_list])
        y_score = np.concatenate([s.flatten() for s in score_list])

        # Filter out void/ignore pixels (Value 255)
        valid_mask = y_true != 255
        y_true, y_score = y_true[valid_mask], y_score[valid_mask]

        # Calculate Metrics
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fpr = fpr_at_95_tpr(
            y_score, y_true
        ).item()  # Custom metric for False Positive Rate

        # Format and Print
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
