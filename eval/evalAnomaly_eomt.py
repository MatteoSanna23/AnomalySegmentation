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

current_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
project_root = os.path.dirname(current_dir) # get parent directory
eomt_package_dir = os.path.join(project_root, 'eomt') # path to eomt package

# Add both paths to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_package_dir not in sys.path:
    sys.path.append(eomt_package_dir)
    
from models.eomt import EoMT
from models.vit import ViT

from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- REPRODUCIBILITY ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19 # Cityscapes classes (ecxluding background/void for metrics ID)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- TRANSFORMS ---

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Typical normalization for ViT/Mask2Former (ImageNet mean/std)
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# resizing ground truth masks to match model output size
target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

#! FUNCTION TO RESIZE POSITIONAL EMBEDDINGS AND MASK
def resize_pos_embed(state_dict, model, target_img_size=(512, 1024)):
    if 'encoder.backbone.pos_embed' in state_dict:  # if positional embeddings exist
        pos_embed_checkpoint = state_dict['encoder.backbone.pos_embed'] # [1, number_of_patches, embedding_dim]
        embedding_dim = pos_embed_checkpoint.shape[-1]  # Embedding dimension (768)
        
        # Get model's current positional embeddings
        if hasattr(model.encoder.backbone, 'pos_embed'):
            model_pos_embed = model.encoder.backbone.pos_embed
        else:
            # Fallback (if model structure is different)
            model_pos_embed = torch.zeros(1, (target_img_size[0]//16)*(target_img_size[1]//16)+1, embedding_dim)
        
        num_patches_checkpoint = pos_embed_checkpoint.shape[1] # Number of patches in checkpoint (4096)
        # here we have to resize from 4096 (64x64) to 2048 (32x64)
        grid_size_chk = int(math.sqrt(num_patches_checkpoint)) # return 64
        
        # Reshape to [1, C, H, W]
        pos_embed_reshaped = pos_embed_checkpoint.transpose(1, 2).reshape(1, embedding_dim, grid_size_chk, grid_size_chk)
        
        # Compute new grid size based on target image size (512x1024) and patch size (16x16 -> 32x64)
        patch_size = 16 
        new_h, new_w = target_img_size[0] // patch_size, target_img_size[1] // patch_size
        
        print(f"Resizing pos_embed from {grid_size_chk}x{grid_size_chk} to {new_h}x{new_w}")
        
        # Interpolate to new size
        new_pos_embed = F.interpolate(
            pos_embed_reshaped, size=(new_h, new_w), mode='bicubic', align_corners=False
        )
        
        # Reshape back to [1, number_of_patches, embedding_dim], bc ViT expects that
        new_pos_embed_spatial = new_pos_embed.flatten(2).transpose(1, 2)
        
        # 4. Gestione CLS Token (Concatenazione)
        # Se il modello vuole N+1 token (es. 2049) e noi ne abbiamo N (2048), aggiungiamo il CLS
        target_len = model_pos_embed.shape[1]
        if target_len == new_pos_embed_spatial.shape[1] + 1:
            print("Adding CLS token to pos_embed (concatenation)")
            cls_pos_embed = model_pos_embed[:, 0:1, :] # Prendi il CLS random dal modello
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed_spatial), dim=1)
        else:
            new_pos_embed = new_pos_embed_spatial

        state_dict['encoder.backbone.pos_embed'] = new_pos_embed
    
    # now we have to resize attn_mask_probs if needed
    # if ckpt has saved 3 probs levels but model has 4 levels (due to different input size)
    if 'attn_mask_probs' in state_dict:
        amp = state_dict['attn_mask_probs'] # [num_levels]
        if amp.shape[0] != model.attn_mask_probs.shape[0]:  # if number of levels differ
            print(f"Adapting attn_mask_probs from {amp.shape} to {model.attn_mask_probs.shape}")
            new_amp = F.interpolate(amp.view(1, 1, -1), size=model.attn_mask_probs.shape[0], mode='linear', align_corners=False)
            state_dict['attn_mask_probs'] = new_amp.view(-1)
            # after this, the state_dict will have the correct number of levels
    return state_dict
#! FUNCTION TO GET PIXEL-WISE SCORES FROM MASK-BASED OUTPUTS
def get_pixel_scores(pred_logits, pred_masks):
    """
    Function to convert mask-based outputs to pixel-wise maps for metrics.
    Returns:
        sem_probs: (19, H, W) Softmax probabilities per class
        pixel_logits: (19, H, W) Approximate logits per pixel (for MaxLogit)
    """
    # now we have pred_logits: (B, Q, K+1) and pred_masks: (B, Q, H, W)
    
    # 1. Query probabilities (Softmax over classes)
    query_probs = F.softmax(pred_logits, dim=-1) # (B, Q, K+1)
    
    # 2. Mask spatial probabilities (Sigmoid)
    mask_probs = F.sigmoid(pred_masks) # (B, Q, H, W)
    
    # Remove the last class (void/no-object) if present (assuming K=19+1)
    valid_query_probs = query_probs[:, :, :NUM_CLASSES] # (B, Q, 19)
    valid_query_logits = pred_logits[:, :, :NUM_CLASSES] # (B, Q, 19) - Raw logits
    
    # 3. Calculate Pixel-wise Semantic Probabilities
    # Weighted sum: sum_q ( P(c|q) * P(pixel|q) )
    sem_probs = torch.einsum("bqc,bqhw->bchw", valid_query_probs, mask_probs)   # (B, C, H, W) e.g., (1, 19, 512, 1024)
    
    # 4. Calculate Approximate Pixel-wise Logits (for MaxLogit)
    # Weighted sum of logits : sum_q ( Logit(c|q) * P(pixel|q) )
    pixel_logits = torch.einsum("bqc,bqhw->bchw", valid_query_logits, mask_probs)
    
    return sem_probs, pixel_logits

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern",
        required=True
    )  
    parser.add_argument('--loadWeights', default="../trained_models/eomt_pretrained.pth")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    # Lists
    msp_list = []
    maxLogit_list = []
    entropy_list = []
    rba_list = [] 
    ood_gts_list = []

    if not os.path.exists('results_eomt.txt'):
        open('results_eomt.txt', 'w').close()
    file = open('results_eomt.txt', 'a')
    
    # Validate weights path early with a helpful message
    if not os.path.isfile(args.loadWeights):
        print(f"Weights file not found: {args.loadWeights}")
        sys.exit(1)

    print(f"Loading EoMT model weights from: {args.loadWeights}")

    # Model Initialization
    encoder = ViT(img_size=(512, 1024), patch_size=16, backbone_name="vit_large_patch14_reg4_dinov2")
    num_queries = 100
    model = EoMT(encoder=encoder, num_q=num_queries, num_classes=NUM_CLASSES)
    
    if not args.cpu:
        model = model.cuda()

    # Load Custom Weights
    checkpoint = torch.load(args.loadWeights, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Clean checkpoint keys (module./model./network.)
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
      
    # Resize positional embeddings and attn_mask_probs if needed, with resizing function
    new_state_dict = resize_pos_embed(new_state_dict, model, target_img_size=(512, 1024))   
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
    model.eval()
    
    #! -- LOOP THROUGH IMAGES --
    # Handle input list or glob pattern
    input_files = []
    for pattern in args.input:  # if multiple patterns are given...
        input_files.extend(glob.glob(os.path.expanduser(pattern)))
        
    for path in input_files:    # for each image path
        filename = os.path.splitext(os.path.basename(path))[0]
        print(filename, end=' - ', flush=True)
        
        # Prepare Input (apply transforms to get a (1, 3, H, W) tensor)
        pil_img = Image.open(path).convert('RGB')
        images = input_transform(pil_img).unsqueeze(0)
        
        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            # EoMT Forward
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                # Take the last level outputs
                # outputs[0]: pred_masks, outputs[1]: pred_logits
                pred_masks = outputs[0][-1]  
                pred_logits = outputs[1][-1]
            elif isinstance(outputs, dict):
                pred_masks = outputs['pred_masks']
                pred_logits = outputs['pred_logits']
            else:
                raise TypeError(f"Formato output inatteso: {type(outputs)}")
            
            # Upsample masks to evaluation resolution (512x1024)
            pred_masks = F.interpolate(pred_masks, size=(512, 1024), mode='bilinear', align_corners=False)
            
            # Obtain pixel-wise maps
            sem_probs, pixel_logits = get_pixel_scores(pred_logits, pred_masks)
            
            # Convert to numpy (remove batch dim) beacause we have batch size 1
            sem_probs_np = sem_probs.squeeze(0).cpu().numpy() # (19, 512, 1024)
            pixel_logits_np = pixel_logits.squeeze(0).cpu().numpy() # (19, 512, 1024)

            # -- COMPUTATION 1 : MSP --
            msp_score = 1 - np.max(sem_probs_np, axis=0)    # 1 - max(Known Class Probabilities)
            
            # -- COMPUTATION 2 : MaxLogit --
            maxLogit_score = - np.max(pixel_logits_np, axis=0)  # - max(Known Class Logits)
            
            # -- COMPUTATION 3 : Entropy --
            entropy_score = - np.sum(sem_probs_np * np.log(sem_probs_np + 1e-8), axis=0)
            
            # -- COMPUTATION 4 : RbA (Rejected by All) --
            # If all class probabilities are low, sum will be low, and RbA will be high.
            rba_score = 1 - np.sum(sem_probs_np, axis=0)    # RbA = 1 - Sum(Probabilities of Known Classes)
        
        #! Management of ground truth (Same as evalAnomaly.py)
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

        # Mapping specific dataset labels to binary OOD (1) / ID (0)
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
            ood_gts_list.append(ood_gts)
            msp_list.append(msp_score)
            maxLogit_list.append(maxLogit_score)
            entropy_list.append(entropy_score)
            rba_list.append(rba_score)
        
        del outputs, sem_probs, msp_score, maxLogit_score, entropy_score, rba_score, ood_gts
        torch.cuda.empty_cache()

    file.write("\n")
    print("\n")
    
    def evaluate_metric(score_list, gt_list, method_name):
        if len(score_list) == 0:
            print(f'No data to evaluate for {method_name}.')
            return
        
        ood_gts = np.array(gt_list)
        anomaly_scores = np.array(score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        res_str = f'[{method_name}] AUPRC: {prc_auc*100.0:.2f} | FPR@95: {fpr*100.0:.2f}'
        print(res_str)
        file.write(res_str + '\n')
    
    # FINAL EVALUATION
    evaluate_metric(msp_list, ood_gts_list, "MSP")
    evaluate_metric(maxLogit_list, ood_gts_list, "MaxLogit")
    evaluate_metric(entropy_list, ood_gts_list, "Entropy")
    evaluate_metric(rba_list, ood_gts_list, "RbA")
    
    file.close()

if __name__ == '__main__':
    main()