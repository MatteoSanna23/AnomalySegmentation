# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet  # Import the ERFNet model architecture
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import (
    fpr_at_95_tpr,
    calc_metrics,
    plot_roc,
    plot_pr,
    plot_barcode,
)  # Custom library for Out-of-Distribution metrics
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# --- REPRODUCIBILITY SETUP ---
# Ensure deterministic behavior for consistent evaluation results
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20  # Cityscapes classes + potential void/anomaly handling
# GPU optimization settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- DATA TRANSFORMS ---
# Standard preprocessing for semantic segmentation
input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize using ImageNet mean/std (standard for pre-trained encoders)
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

# Resize Ground Truth masks without interpolation (Nearest Neighbor) to keep integer labels
target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument("--loadModel", default="erfnet.py")
    parser.add_argument("--subset", default="val")  # Subset to evaluate (train/val)
    parser.add_argument(
        "--datadir", default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Lists to aggregate scores across the dataset
    anomaly_score_list = []
    msp_list = []
    maxLogit_list = []
    entropy_list = []
    ood_gts_list = []

    # Initialize results output file
    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    file = open("results.txt", "a")

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)  # Instantiate the ERFNet model

    if not args.cpu:
        model = torch.nn.DataParallel(
            model
        ).cuda()  # Wrap in DataParallel for multi-GPU support

    def load_my_state_dict(model, state_dict):
        """
        Custom weight loading function.
        Handles the mismatch between DataParallel state dicts (starting with 'module.')
        and standard model state dicts.
        """
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # If weights were saved with DataParallel, strip the 'module.' prefix
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    # Load weights into the model
    model = load_my_state_dict(
        model, torch.load(weightspath, map_location=lambda storage, loc: storage)
    )
    print("Model and weights LOADED successfully")
    model.eval()

    # --- INFERENCE LOOP ---
    # Iterate over all images specified in the input arguments
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        filename = os.path.splitext(os.path.basename(path))[0]

        # Preprocess image: Load -> Resize -> ToTensor -> Batch Dim -> GPU

        images = (
            input_transform((Image.open(path).convert("RGB")))
            .unsqueeze(0)
            .float()
            .cuda()
        )

        # Forward pass to get logits (unnormalized scores)
        with torch.no_grad():
            logits = model(images)

        # --- POST-PROCESSING CONFIG ---
        # Gaussian Blur helps smooth out noisy pixel predictions,
        # improving area-based metrics like AUPRC/FPR95.
        DO_BLUR = True
        KERNEL_SIZE = (21, 21)

        #

        # -- METRIC 1: MSP (Maximum Softmax Probability) --
        # Idea: If the max probability is low, the model is uncertain (likely anomaly).
        # Score = 1 - max(softmax)
        softmax = F.softmax(logits, dim=1)
        msp_score = 1 - np.max(softmax.squeeze(0).data.cpu().numpy(), axis=0)

        # -- METRIC 2: MaxLogit --
        # Idea: Logits carry more information about magnitude than softmax.
        # Score = -max(logits) (Negated so higher score = anomaly)
        maxLogit_score = -np.max(logits.squeeze(0).data.cpu().numpy(), axis=0)

        # -- METRIC 3: Entropy --
        # Idea: High entropy implies the distribution is flat (uncertainty).
        # Score = -Sum(p * log(p))
        prob_cpu = softmax.squeeze(0).data.cpu().numpy()
        entropy_score = -np.sum(prob_cpu * np.log(prob_cpu + 1e-8), axis=0)

        # --- APPLY SMOOTHING ---
        if DO_BLUR:
            msp_score = cv2.GaussianBlur(msp_score, KERNEL_SIZE, 0)
            maxLogit_score = cv2.GaussianBlur(maxLogit_score, KERNEL_SIZE, 0)
            entropy_score = cv2.GaussianBlur(entropy_score, KERNEL_SIZE, 0)

        # --- GROUND TRUTH LOADING & MAPPING ---
        # Derive GT path from image path (e.g., .../images/X.jpg -> .../labels_masks/X.png)
        pathGT = path.replace("images", "labels_masks")
        # Handle specific dataset naming conventions
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # --- DATASET-SPECIFIC LABEL MAPPING ---
        # Convert diverse labels into a binary mask:
        # 0 = In-Distribution, 1 = Anomaly, 255 = Ignore/Void

        if "RoadAnomaly" in pathGT:
            # Class 2 is anomaly
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            # 0: Void, 1: Road (ID), >1: Obstacles (Anomaly)
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if "Streethazard" in pathGT:
            # 14: Void (Ego-vehicle), <20: ID, 255: Anomaly
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Skip images containing only background/void (no valid anomaly pixels to evaluate)
        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            msp_list.append(msp_score)
            maxLogit_list.append(maxLogit_score)
            entropy_list.append(entropy_score)

        # Memory cleanup
        del (
            logits,
            softmax,
            msp_score,
            maxLogit_score,
            entropy_score,
            ood_gts,
            mask,
        )
        torch.cuda.empty_cache()

    file.write("\n")
    print("\n")

    # --- EVALUATION FUNCTION ---
    def evaluate_metric(score_list, gt_list, method_name):
        """
        Calculates AUPRC (Area Under Precision-Recall Curve) and FPR@95.
        Flattens the entire dataset prediction maps into a single array for metric calculation.
        """

        if len(score_list) == 0 or len(gt_list) == 0:
            print(f"No data to evaluate for {method_name}.")
            file.write(f"No data to evaluate for {method_name}.\n")
            return

        ood_gts = np.array(gt_list)
        anomaly_scores = np.array(score_list)

        # Create binary masks for valid pixels
        ood_mask = ood_gts == 1  # Anomaly pixels
        ind_mask = ood_gts == 0  # In-distribution pixels

        # Flatten arrays
        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        # Create labels
        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        # Concatenate for sklearn
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        # Calculate Metrics
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f"\n[{method_name}] AUPRC: {prc_auc*100.0:.2f} | FPR@95: {fpr*100.0:.2f}")
        file.write(
            f"[{method_name}] AUPRC: {prc_auc*100.0:.2f}   FPR@95: {fpr*100.0:.2f}\n"
        )

    # Calculate metrics for all methods
    evaluate_metric(msp_list, ood_gts_list, "MSP")
    evaluate_metric(maxLogit_list, ood_gts_list, "MaxLogit")
    evaluate_metric(entropy_list, ood_gts_list, "Entropy")
    file.close()


if __name__ == "__main__":
    main()
