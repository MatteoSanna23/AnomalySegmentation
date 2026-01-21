# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet  # import the ERFNet model
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import (
    fpr_at_95_tpr,
    calc_metrics,
    plot_roc,
    plot_pr,
    plot_barcode,
)  # load OOD metrics functions
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

# resizing ground truth masks to match model output size
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
    parser.add_argument(
        "--subset", default="val"
    )  # can be val or train (must have labels)
    parser.add_argument(
        "--datadir", default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    anomaly_score_list = []
    msp_list = []
    maxLogit_list = []
    entropy_list = []
    ood_gts_list = []

    # if results.txt does not exist, create it
    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    file = open("results.txt", "a")
    modelpath = args.loadDir + args.loadModel  # path to model .py file
    weightspath = args.loadDir + args.loadWeights  # path to model weights .pth file

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)  # create model instance

    if not args.cpu:
        model = torch.nn.DataParallel(
            model
        ).cuda()  # parallelize model on multiple GPUs

    def load_my_state_dict(model, state_dict):  # custom function to load model weights
        own_state = model.state_dict()  # get model's state dict (parameters)
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(
                        param
                    )  # remove module. if model was trained in DataParallel
                else:
                    print(name, " not loaded")  # if
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(
        model, torch.load(weightspath, map_location=lambda storage, loc: storage)
    )
    print("Model and weights LOADED successfully")
    model.eval()

    #! -- LOOP THROUGH IMAGES --
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        filename = os.path.splitext(os.path.basename(path))[0]
        # print(filename, end=' - ', flush=True)
        images = (
            input_transform((Image.open(path).convert("RGB")))
            .unsqueeze(0)
            .float()
            .cuda()
        )  # transofrm + add batch dimension + to GPU
        with torch.no_grad():
            logits = model(images)  # forward pass

        # --- BLUR SETTING ---
        # Un kernel (21,21) aiuta a rimuovere il rumore dei singoli pixel
        DO_BLUR = True
        KERNEL_SIZE = (21, 21)

        # -- COMPUTATION 1 : MSP --
        softmax = F.softmax(logits, dim=1)
        msp_score = 1 - np.max(softmax.squeeze(0).data.cpu().numpy(), axis=0)

        # -- COMPUTATION 2 : MaxLogit --
        maxLogit_score = -np.max(logits.squeeze(0).data.cpu().numpy(), axis=0)

        # -- COMPUTATION 3 : Entropy --
        prob_cpu = softmax.squeeze(0).data.cpu().numpy()
        entropy_score = -np.sum(prob_cpu * np.log(prob_cpu + 1e-8), axis=0)

        # --- APPLICAZIONE GAUSSIAN BLUR (Post-Processing) ---
        # Questo passaggio è fondamentale per abbassare FPR95
        if DO_BLUR:
            msp_score = cv2.GaussianBlur(msp_score, KERNEL_SIZE, 0)
            maxLogit_score = cv2.GaussianBlur(maxLogit_score, KERNEL_SIZE, 0)
            entropy_score = cv2.GaussianBlur(entropy_score, KERNEL_SIZE, 0)

        #! Management of ground truth
        pathGT = path.replace(
            "images", "labels_masks"
        )  # get ground truth path (path/to/images/img.png -> path/to/labels_masks/img.png)
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)  # mask contains the true OOD labels
        mask = target_transform(mask)  # resize, (implemented above)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where(
                (ood_gts == 2), 1, ood_gts
            )  # map anomaly class to 1 and rest to 0
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)  # map void class to 255
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)  # map background to 0
            ood_gts = np.where(
                (ood_gts > 1) & (ood_gts < 201), 1, ood_gts
            )  # map anomaly classes to 1

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)  # map void class to 255
            ood_gts = np.where(
                (ood_gts < 20), 0, ood_gts
            )  # map background classes to 0
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)  # map anomaly classes to 1

        if 1 not in np.unique(ood_gts):  # skip images without OOD pixels
            continue
        else:
            ood_gts_list.append(ood_gts)  # add ground truth mask to list
            msp_list.append(msp_score)
            maxLogit_list.append(maxLogit_score)
            entropy_list.append(entropy_score)
        del (
            logits,
            softmax,
            msp_score,
            maxLogit_score,
            entropy_score,
            ood_gts,
            mask,
        )  # free some memory
        torch.cuda.empty_cache()

    file.write("\n")
    print("\n")

    #!-- FUNCTION TO EVALUATE METRICS FOR A GIVEN METHOD --
    def evaluate_metric(score_list, gt_list, method_name):
        # security check
        if len(score_list) == 0 or len(gt_list) == 0:
            print(f"No data to evaluate for {method_name}.")
            file.write(f"No data to evaluate for {method_name}.\n")
            return

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

        print(f"\n[{method_name}] AUPRC: {prc_auc*100.0:.2f} | FPR@95: {fpr*100.0:.2f}")
        file.write(
            f"[{method_name}] AUPRC: {prc_auc*100.0:.2f}   FPR@95: {fpr*100.0:.2f}\n"
        )

    # FINAL EVALUATION FOR ALL METHODS
    evaluate_metric(msp_list, ood_gts_list, "MSP")
    evaluate_metric(maxLogit_list, ood_gts_list, "MaxLogit")
    evaluate_metric(entropy_list, ood_gts_list, "Entropy")
    file.close()


if __name__ == "__main__":
    main()
