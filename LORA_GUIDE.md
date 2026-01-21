# 🧠 LoRA Implementation Guide - Membro 2

## Overview

Hai implementato **LoRA (Low-Rank Adaptation)** nell'architettura EoMT per permettere fine-tuning efficiente senza addestrare milioni di parametri.

### Cosa è stato fatto:

1. ✅ **lora_config.py**: Modulo con funzioni PEFT per applicare LoRA
2. ✅ **mask_classification_semantic.py**: Integrazione LoRA nel Lightning Module
3. ✅ **train_with_lora.py**: Script template per training con LoRA

---

## 📋 Come Usare

### Step 1: Installa PEFT (su Lightning AI)

```bash
pip install peft
```

### Step 2: Crea il config YAML con LoRA

Nel tuo config di training (es. `configs/cityscapes_lora.yaml`):

```yaml
model:
  _target_: eomt.training.mask_classification_semantic.MaskClassificationSemantic
  img_size: [640, 640]
  num_classes: 19
  lr: 1e-4
  
  # LoRA Parameters
  use_lora: true
  lora_rank: 16         # Decomposition rank (8-32 consigliato)
  lora_alpha: 32        # Scaling factor (~2x rank)
  lora_dropout: 0.05    # Dropout per LoRA layers
```

### Step 3: Lancia il Training

```bash
# Con LoRA
python train_with_lora.py \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --epochs 50 \
  --batch_size 4

# Oppure senza LoRA (baseline)
python train_with_lora.py \
  --epochs 50 \
  --batch_size 4
```

---

## 🔍 Come Verificare che LoRA Funziona

Quando il training parte, vedrai nei log:

```
================================================================================
TRAINABLE PARAMETERS (LoRA)
================================================================================
  network.encoder.backbone.blocks.0.attn.q_proj.lora_A: 12,288
  network.encoder.backbone.blocks.0.attn.q_proj.lora_B: 12,288
  network.encoder.backbone.blocks.0.attn.v_proj.lora_A: 12,288
  ...
Total trainable: 1,234,567

Total frozen: 1,234,567,890

SUMMARY: 0.10% trainable
================================================================================
```

**Questo è corretto!** Significa che:
- ✅ Solo lo 0.10% dei parametri è trainabile (LoRA adattatori)
- ✅ Il 99.90% è congelato (pre-trained weights)
- ✅ Memoria e tempo computazionale ridotti drasticamente

---

## 📊 Parametri LoRA - Cosa Significano

| Parametro | Default | Range | Consiglio |
|-----------|---------|-------|-----------|
| `lora_rank` (r) | 16 | 8-32 | 16 per modelli grandi |
| `lora_alpha` | 32 | 2x rank | Mantieni 2x rank |
| `lora_dropout` | 0.05 | 0-0.2 | 0.05-0.1 va bene |

**Come regolarli:**
- ↑ **Più rank** = Più capacità, più memoria
- ↓ **Meno rank** = Più veloce, meno parametri

Per EoMT (encoder 300M): **lora_rank=16** è un buon compromesso.

---

## 🔧 Target Modules

LoRA viene applicato a:
```python
target_modules = ["qkv", "proj", "q_proj", "v_proj"]
```

Questi sono i layer di attenzione che contengono le informazioni più importanti.

Se il tuo modello usa nomi diversi (es. "attention.W_q" invece di "q_proj"), modificali in `lora_config.py`:

```python
target_modules = ["W_q", "W_v", "W_proj"]  # Custom names
```

---

## ⚠️ Troubleshooting

### Problema: "AttributeError: No module named 'peft'"

**Soluzione**: Installa PEFT su Lightning AI
```bash
pip install peft
```

### Problema: "Model has no attribute 'peft_config'"

**Soluzione**: Assicurati che `use_lora=True` sia passato al constructor

### Problema: "All parameters are frozen"

**Causa**: LoRA non è stata applicata.
**Check**: 
1. Verifica che il modello sia caricato PRIMA di applicare LoRA
2. Controlla che i target_modules siano corretti

---

## 📈 Cosa Ti Aspetti nei Risultati

**Senza LoRA** (fine-tuning completo):
- Trainable params: ~300M (tutti i parametri del ViT)
- Training time: ~24-48 ore su GPU
- Memory: ~40GB VRAM

**Con LoRA** (rank=16):
- Trainable params: ~1-2M (solo LoRA adattatori)
- Training time: ~2-4 ore su GPU
- Memory: ~8-10GB VRAM
- **Speedup: 10-20x più veloce!**

---

## 🎯 Prossimi Passi

Una volta che il training funziona:

1. **Membro 3** implementa ArcFace + RbA loss
2. **Membro 1** prepara dataset con Cut-Paste
3. **Membro 4** esegue gli esperimenti e raccoglie risultati

---

## 📚 Riferimenti

- PEFT Docs: https://huggingface.co/docs/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685
- EoMT Architecture: Vedi `eomt.py`

---

## 💾 Salvare i Checkpoint LoRA

I checkpoint di LoRA sono **molto più piccoli**:

```python
# Salva solo i pesi LoRA (~2MB)
if hasattr(model, 'save_pretrained'):
    model.save_pretrained("checkpoint_lora")

# Oppure con Lightning:
# Il checkpoint salva automaticamente tutto correttamente
```

---

**Creato da: Membro 2 - The Architect**  
**Data: 2025-01-21**
