# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Optional
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)

class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        arcface_s: float = 30.0,  # Scala (simile a inverso della temperatura)
        arcface_m: float = 0.50,  # Margine angolare (in radianti)
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        
        # Parametri ArcFace
        self.s = arcface_s
        self.m = arcface_m
        
        # Pre-calcolo valori trigonometrici per efficienza
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # Soglia per evitare instabilità numerica (theta deve stare tra 0 e pi)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """
        Classification loss con ArcFace logic
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float() # Shape: [Batch, Queries, Classes]

        # 1. Normalizzazione (L2) per ottenere i Coseni
        # In ArcFace, l'input alla softmax deve essere il coseno dell'angolo
        cosine = F.normalize(src_logits, p=2, dim=-1)

        # 2. Preparazione Target
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Inizializziamo il target con la classe "No Object" (ultima classe)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_labels, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # 3. Applicazione del Margine ArcFace SOLO sulle classi ground-truth
        # Appiattiamo per facilitare le operazioni vettoriali
        cosine_flat = cosine.view(-1, cosine.shape[-1]) 
        target_flat = target_classes.view(-1)

        # Maschera per identificare dove applicare il margine 
        # (Non applichiamo il margine alla classe "No Object" se è l'ultima, o decidiamo di sì. 
        #  Qui lo applichiamo solo alle classi valide < num_labels)
        mask_valid = target_flat < self.num_labels
        
        if mask_valid.sum() > 0:
            # Estraiamo i coseni delle classi target (ground truth)
            # cosine_of_target.shape: [num_valid_pixels]
            cosine_of_target = cosine_flat[mask_valid, target_flat[mask_valid]]

            # Calcolo cos(theta + m) usando formule di addizione
            # cos(t + m) = cos(t)cos(m) - sin(t)sin(m)
            sine_of_target = torch.sqrt((1.0 - torch.pow(cosine_of_target, 2)).clamp(0, 1))
            phi = cosine_of_target * self.cos_m - sine_of_target * self.sin_m

            # Gestione stabilità numerica (condizione easy_margin spesso usata in ArcFace)
            # Se theta > pi - m, usiamo un'approssimazione per evitare gradienti instabili
            phi = torch.where(cosine_of_target > self.th, phi, cosine_of_target - self.mm)

            # Sostituiamo i valori originali con quelli penalizzati dal margine
            # Creiamo una copia dei logits (coseni) per non modificare in-place in modo errato
            output_flat = cosine_flat.clone()
            
            # Usiamo scatter per aggiornare solo gli indici corretti
            # Dobbiamo calcolare gli indici lineari o usare scatter sulla dimensione classi
            
            # Approccio più pulito con scatter_:
            # Creiamo un one-hot mask solo per i sample validi
            one_hot = torch.zeros_like(cosine_flat)
            one_hot.scatter_(1, target_flat.view(-1, 1), 1.0)
            
            # Dove c'è target valido, mettiamo phi, altrimenti lasciamo cosine originale
            # Nota: questo è un modo vettorizzato. Per efficienza su GPU facciamo:
            # output = (one_hot * phi) + ((1.0 - one_hot) * cosine) -- MA phi è solo per i validi.
            
            # Metodo diretto: iteriamo sui validi (più sicuro in pytorch standard) o scatter avanzato
            # Per semplicità nel contesto Mask2Former, modifichiamo direttamente i valori selezionati
            # Poiché abbiamo già target_flat e mask_valid, possiamo farlo:
            
            # Indici [0...N] per le righe che sono valide
            valid_indices = torch.nonzero(mask_valid).squeeze()
            # Classi target per queste righe
            valid_targets = target_flat[mask_valid]
            
            # Aggiornamento
            output_flat[valid_indices, valid_targets] = phi

        else:
            output_flat = cosine_flat

        # 4. Riscalare per s
        output = output_flat * self.s
        
        # Reshape per tornare [Batch, Queries, Classes] se necessario, ma cross_entropy accetta (N, C)
        # Mask2Former loss di solito aspetta [Batch, NumClasses, Queries] per la transpose
        # Ma qui abbiamo appiattito tutto. Riformattiamo per coerenza con l'originale
        output = output.view(src_logits.shape)

        # Calcolo Loss
        loss_ce = F.cross_entropy(output.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        return loss_total

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx