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

#--------------------------------------------------------------------------------
def resize_pos_embed(state_dict, model, target_img_size=(512, 1024)):
    """
    Adatta i positional embeddings dal checkpoint alla risoluzione attuale.
    """
    # 1. Gestione Positional Embeddings
    if 'encoder.backbone.pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['encoder.backbone.pos_embed'] # [1, N_chk, C]
        embedding_dim = pos_embed_checkpoint.shape[-1]
        num_patches_checkpoint = pos_embed_checkpoint.shape[1]
        
        # Calcola la griglia del checkpoint (es. 4096 -> 64x64)
        grid_size_chk = int(math.sqrt(num_patches_checkpoint))
        
        # Prepara il tensore per l'interpolazione: [1, C, H, W]
        # Nota: assumiamo patch quadrati. Se grid_size_chk^2 != num_patches, la logica cambia,
        # ma per ViT standard (es. 4096) è 64x64.
        pos_embed_reshaped = pos_embed_checkpoint.transpose(1, 2).reshape(1, embedding_dim, grid_size_chk, grid_size_chk)
        
        # Calcola nuova griglia target (es. 512x1024 con patch 16 -> 32x64)
        patch_size = 16 
        new_h, new_w = target_img_size[0] // patch_size, target_img_size[1] // patch_size
        
        print(f"Resizing pos_embed from {grid_size_chk}x{grid_size_chk} to {new_h}x{new_w}")
        
        new_pos_embed = F.interpolate(
            pos_embed_reshaped, size=(new_h, new_w), mode='bicubic', align_corners=False
        )
        
        # Riporta alla forma [1, N_new, C]
        new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)
        state_dict['encoder.backbone.pos_embed'] = new_pos_embed

    # 2. Gestione attn_mask_probs (3 livelli vs 4 livelli)
    # Se il checkpoint ha 3 pesi e il modello ne aspetta 4, probabilmente
    # dobbiamo adattare il modello o "falsificare" il caricamento. 
    # Qui proviamo ad adattare il tensore del checkpoint facendo padding (o interpolazione),
    # ma la soluzione ideale sarebbe configurare EoMT per usare 3 livelli.
    if 'attn_mask_probs' in state_dict:
        amp = state_dict['attn_mask_probs']
        if amp.shape[0] != model.attn_mask_probs.shape[0]:
            print(f"Adattamento attn_mask_probs da {amp.shape} a {model.attn_mask_probs.shape}")
            # Se ne mancano, facciamo resize (brutale ma permette il caricamento)
            # Soluzione alternativa: Se puoi, modifica EoMT(..., num_feature_levels=3)
            new_amp = F.interpolate(amp.view(1, 1, -1), size=model.attn_mask_probs.shape[0], mode='linear', align_corners=False)
            state_dict['attn_mask_probs'] = new_amp.view(-1)
            
    return state_dict

# in this case we have to exit from eval folder and go to eomt/models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
eomt_package_dir = os.path.join(project_root, 'eomt')

# 4. Aggiungi entrambi i percorsi a sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
if eomt_package_dir not in sys.path:
    sys.path.append(eomt_package_dir)
    
#--------------------------------------------------------------------------------

from models.eomt import EoMT

from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from models.vit import ViT

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19 # Cityscapes classes (ecxluding background/void for metrics ID)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

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

def get_pixel_scores(pred_logits, pred_masks):
    """
    Function to convert mask-based outputs to pixel-wise maps for metrics.
    Returns:
        sem_probs: (19, H, W) Softmax probabilities per class
        pixel_logits: (19, H, W) Approximate logits per pixel (for MaxLogit)
    """
    # pred_logits: (B, Q, K+1) -> K classes + 1 (void)
    # pred_masks: (B, Q, H, W)
    
    # 1. Query probabilities (Softmax over classes)
    query_probs = F.softmax(pred_logits, dim=-1) # (B, Q, K+1)
    
    # 2. Mask spatial probabilities (Sigmoid)
    mask_probs = F.sigmoid(pred_masks) # (B, Q, H, W)
    
    # Remove the last class (void/no-object) if present (assuming K=19+1)
    # If the model has exactly 19 outputs, use all.
    valid_query_probs = query_probs[:, :, :NUM_CLASSES] # (B, Q, 19)
    valid_query_logits = pred_logits[:, :, :NUM_CLASSES] # (B, Q, 19) - Raw logits
    
    # 3. Calculate Pixel-wise Semantic Probabilities
    # Weighted sum: sum_q ( P(c|q) * P(pixel|q) )
    # Output: (B, C, H, W)
    sem_probs = torch.einsum("bqc,bqhw->bchw", valid_query_probs, mask_probs)
    
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
    
    msp_list = []
    maxLogit_list = []
    entropy_list = []
    rba_list = [] # List for Rejected by All
    ood_gts_list = []

    if not os.path.exists('results_eomt.txt'):
        open('results_eomt.txt', 'w').close()
    file = open('results_eomt.txt', 'a')
    
    # Validate weights path early with a helpful message
    if not os.path.isfile(args.loadWeights):
        print(f"Weights file not found: {args.loadWeights}")
        print("Tip: Download the Cityscapes EoMT pretrained checkpoint from the link in eomt/README.md and place it under trained_models (e.g., trained_models/eomt_pretrained.pth), or pass --loadWeights to its path.")
        sys.exit(1)

    print(f"Loading EoMT model weights from: {args.loadWeights}")

    # Model Initialization
    # Note: Ensure parameters (e.g., backbone) match those used in training
    encoder = ViT(img_size=(512, 1024), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4) 
    num_queries = 100
    model = EoMT(encoder=encoder, num_q=num_queries, num_classes=NUM_CLASSES)
    
    if not args.cpu:
        model = model.cuda()

    # Caricamento Pesi Custom
    checkpoint = torch.load(args.loadWeights, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Pulizia chiavi comuni dei checkpoint Lightning/HF (module./model./network.)
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
      
    new_state_dict = resize_pos_embed(new_state_dict, model, target_img_size=(512, 1024))    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
    model.eval()
    
    #! -- LOOP THROUGH IMAGES --
    # Gestione input list o glob pattern
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(os.path.expanduser(pattern)))
        
    for path in input_files:
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
            pred_logits = outputs['pred_logits'] # (B, Q, C+1)
            pred_masks = outputs['pred_masks']   # (B, Q, h, w)
            
            # Upsample masks to evaluation resolution (512x1024)
            pred_masks = F.interpolate(pred_masks, size=(512, 1024), mode='bilinear', align_corners=False)
            
            # Obtain pixel-wise maps
            sem_probs, pixel_logits = get_pixel_scores(pred_logits, pred_masks)
            
            # Convert to numpy (remove batch dim)
            sem_probs_np = sem_probs.squeeze(0).cpu().numpy() # (19, 512, 1024)
            pixel_logits_np = pixel_logits.squeeze(0).cpu().numpy() # (19, 512, 1024)

            # -- COMPUTATION 1 : MSP --
            # Anomaly Score = 1 - max(Known Class Probabilities)
            msp_score = 1 - np.max(sem_probs_np, axis=0)
            
            # -- COMPUTATION 2 : MaxLogit --
            # Anomaly Score = - max(Known Class Logits)
            maxLogit_score = - np.max(pixel_logits_np, axis=0)
            
            # -- COMPUTATION 3 : Entropy --
            entropy_score = - np.sum(sem_probs_np * np.log(sem_probs_np + 1e-8), axis=0)
            
            # -- COMPUTATION 4 : RbA (Rejected by All) --
            # RbA score is high when the pixel is NOT covered by any known class.
            # RbA = 1 - Sum(Probabilità Classi Note)
            # Poiché sem_probs contiene solo le probabilità delle classi note (escluso void/no-object),
            # la loro somma sarà < 1 dove il modello predice "no-object" o è incerto.
            rba_score = 1 - np.sum(sem_probs_np, axis=0)
        
        #! Management of ground truth (Uguale a evalAnomaly.py originale)
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

        # Mapping specifico dataset (come in script originale)
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