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

# --- IMPORTS ---
from models.eomt import EoMT
from models.vit import ViT
# IMPORTANTE: Importiamo la config di LoRA
from models.lora_integration import LoRAConfig 

from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- REPRODUCIBILITY ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CONFIGURAZIONE FONDAMENTALE (Deve matchare il training)
NUM_CLASSES = 19  # Cityscapes standard
PATCH_SIZE = 16   # Come da tuo training
IMG_SIZE = (512, 1024) # Risoluzione inferenza

# GPU Setup
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- TRANSFORMS ---
input_transform = Compose([
    Resize(IMG_SIZE, Image.BILINEAR),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

target_transform = Compose([
    Resize(IMG_SIZE, Image.NEAREST),
])

def resize_pos_embed(state_dict, model, target_img_size=IMG_SIZE):
    """Adatta i positional embeddings se la risoluzione cambia."""
    if "encoder.backbone.pos_embed" in state_dict:
        pos_embed_checkpoint = state_dict["encoder.backbone.pos_embed"]
        embedding_dim = pos_embed_checkpoint.shape[-1]

        num_patches_checkpoint = pos_embed_checkpoint.shape[1]
        grid_size_chk = int(math.sqrt(num_patches_checkpoint))
        
        # Reshape e Interpolazione
        pos_embed_reshaped = pos_embed_checkpoint.transpose(1, 2).reshape(
            1, embedding_dim, grid_size_chk, grid_size_chk
        )
        
        new_h, new_w = target_img_size[0] // PATCH_SIZE, target_img_size[1] // PATCH_SIZE
        new_pos_embed = F.interpolate(
            pos_embed_reshaped, size=(new_h, new_w), mode="bicubic", align_corners=False
        )
        new_pos_embed_spatial = new_pos_embed.flatten(2).transpose(1, 2)

        # Gestione CLS token se necessario
        if hasattr(model.encoder.backbone, "pos_embed"):
             model_pos_embed = model.encoder.backbone.pos_embed
             target_len = model_pos_embed.shape[1]
             if target_len == new_pos_embed_spatial.shape[1] + 1:
                 # print("Adding CLS token to pos_embed")
                 cls_pos_embed = model_pos_embed[:, 0:1, :]
                 new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed_spatial), dim=1)
             else:
                 new_pos_embed = new_pos_embed_spatial
        else:
             new_pos_embed = new_pos_embed_spatial

        state_dict["encoder.backbone.pos_embed"] = new_pos_embed
        
    return state_dict

def get_pixel_scores(pred_logits, pred_masks):
    """Calcola le metriche di anomalia standard (MSP, MaxLogit, RbA)."""
    # Softmax sui logits delle classi note (0-18)
    # Ignoriamo l'ultimo canale (Background/No Object) per le metriche ID
    valid_logits = pred_logits[:, :, :NUM_CLASSES]
    
    # Probabilità maschere
    mask_probs = F.sigmoid(pred_masks)
    
    # Thresholding per pulire il rumore (Opzionale ma aiuta)
    mask_probs[mask_probs < 0.5] = 0.0
    
    # Softmax per avere probabilità
    probs = F.softmax(valid_logits, dim=-1)

    # Proiezione Pixel-wise: Somma pesata (Prob Classe * Prob Maschera)
    sem_probs = torch.einsum("bqc,bqhw->bchw", probs, mask_probs)      # [1, 19, H, W]
    pixel_logits = torch.einsum("bqc,bqhw->bchw", valid_logits, mask_probs) # [1, 19, H, W]

    return sem_probs, pixel_logits

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--loadWeights", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Liste per metriche
    msp_list, maxLogit_list, entropy_list, rba_list, ood_gts_list = [], [], [], [], []

    if not os.path.exists("results_lora.txt"): open("results_lora.txt", "w").close()
    file = open("results_lora.txt", "a")

    print(f"Loading LoRA model from: {args.loadWeights}")

    # --- 1. MODELLO CON LORA ---
    # Configurazione identica al training
    lora_conf = LoRAConfig(
        enabled=True,
        rank=8,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        freeze_base_model=True
    )

    encoder = ViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE, # 16
        backbone_name="vit_base_patch14_reg4_dinov2",
    )
    
    model = EoMT(
        encoder=encoder, 
        num_q=100, 
        num_classes=NUM_CLASSES, # 19
        num_blocks=3,
        lora_config=lora_conf    # <--- FONDAMENTALE
    )

    if not args.cpu: model = model.cuda()

    # --- 2. CARICAMENTO PESI ---
    checkpoint = torch.load(args.loadWeights, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # Pulizia nomi chiavi
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("model.", "").replace("network.", "")
        new_state_dict[name] = v

    # Resize pos embeddings
    new_state_dict = resize_pos_embed(new_state_dict, model)
    
    # Load (strict=False è ok qui, ma dovresti vedere poche missing keys relative a LoRA)
    msg = model.load_state_dict(new_state_dict, strict=False)
    # Se LoRA è configurato bene, msg.missing_keys NON deve contenere "lora_A" o "lora_B"
    print(f"Weights loaded. Missing: {len(msg.missing_keys)}") 
    model.eval()

    # --- 3. LOOP VALUTAZIONE ---
    input_files = []
    for pattern in args.input: input_files.extend(glob.glob(os.path.expanduser(pattern)))

    for path in input_files:
        print(f"Processing: {os.path.basename(path)}", end="\r")
        
        # Inference
        img = input_transform(Image.open(path).convert("RGB")).unsqueeze(0)
        if not args.cpu: img = img.cuda()

        with torch.no_grad():
            outputs = model(img)
            pred_masks = outputs[0][-1]  # Ultimo livello
            pred_logits = outputs[1][-1] # Ultimo livello

            # Upsample e Calcolo Score
            pred_masks = F.interpolate(pred_masks, size=IMG_SIZE, mode="bilinear", align_corners=False)
            sem_probs, pixel_logits = get_pixel_scores(pred_logits, pred_masks)
            
            # Numpy conversion
            sem_probs_np = sem_probs.squeeze(0).cpu().numpy()
            pixel_logits_np = pixel_logits.squeeze(0).cpu().numpy()

            # Metriche Anomaly
            msp_list.append(1 - np.max(sem_probs_np, axis=0))
            maxLogit_list.append(-np.max(pixel_logits_np, axis=0))
            entropy_list.append(-np.sum(sem_probs_np * np.log(sem_probs_np + 1e-8), axis=0))
            rba_list.append(1 - np.sum(sem_probs_np, axis=0))

        # --- GROUND TRUTH LOADING (Standard Logic) ---
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")
        
        mask_np = np.array(target_transform(Image.open(pathGT)))
        
        # Mappatura Label Anomalia
        ood_gts = mask_np.copy() # Default
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((mask_np == 2), 1, ood_gts)
        elif "LostAndFound" in pathGT:
            ood_gts = np.where((mask_np == 0), 255, ood_gts)
            ood_gts = np.where((mask_np == 1), 0, ood_gts)
            ood_gts = np.where((mask_np > 1) & (mask_np < 201), 1, ood_gts)
        elif "Streethazard" in pathGT:
            ood_gts = np.where((mask_np == 14), 255, ood_gts)
            ood_gts = np.where((mask_np < 20), 0, ood_gts)
            ood_gts = np.where((mask_np == 255), 1, ood_gts)
            
        if 1 in np.unique(ood_gts):
            ood_gts_list.append(ood_gts)
        else:
            # Rimuovi gli score appena aggiunti se l'immagine non ha GT valido
            msp_list.pop(); maxLogit_list.pop(); entropy_list.pop(); rba_list.pop()

    print("\n\n--- RESULTS ---")
    
    def eval_metric(scores, gts, name):
        if not scores: return
        # Flattening
        out = np.concatenate([s.flatten() for s in scores])
        lbl = np.concatenate([g.flatten() for g in gts])
        
        # Filter valid pixels (exclude 255)
        mask = lbl != 255
        out, lbl = out[mask], lbl[mask]
        
        fpr = fpr_at_95_tpr(out, lbl)
        auprc = average_precision_score(lbl, out)
        
        res = f"[{name}] FPR95: {fpr*100:.2f} | AUPRC: {auprc*100:.2f}"
        print(res)
        file.write(res + "\n")

    eval_metric(msp_list, ood_gts_list, "MSP")
    eval_metric(maxLogit_list, ood_gts_list, "MaxLogit")
    eval_metric(rba_list, ood_gts_list, "RbA")
    
    file.close()
    print(f"Results saved to results_lora.txt")

if __name__ == "__main__":
    main()