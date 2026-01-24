# LoRA Quick Start Guide

Qui puoi trovare una guida rapida per iniziare ad usare LoRA nel progetto EoMT.

## Installazione

Non sono necessarie dipendenze aggiuntive. LoRA usa solo PyTorch che è già nelle dipendenze.

## Uso Base (60 secondi)

### 1. Crea una configurazione LoRA

```python
from models.lora_integration import LoRAConfig

lora_config = LoRAConfig(
    enabled=True,                    # Abilita LoRA
    rank=8,                          # Dimensione delle matrici di basso rango
    lora_alpha=16,                   # Fattore di scaling
    target_modules=["class_head", "mask_head"],  # Applica a questi moduli
    freeze_base_model=True,          # Congela il resto del modello
)
```

### 2. Crea il modello con LoRA

```python
from models.eomt import EoMT
from models.vit import ViT

# Crea l'encoder ViT come al solito
vit_encoder = ViT(
    img_size=(640, 640),
    backbone_name="vit_large_patch14_reg4_dinov2",
)

# Crea EoMT con LoRA
model = EoMT(
    encoder=vit_encoder,
    num_classes=19,
    num_q=100,
    num_blocks=4,
    lora_config=lora_config,  # Passa la configurazione LoRA
)
```

### 3. Addestra normalmente

```python
# Con PyTorch Lightning
from lightning.pytorch import Trainer

trainer = Trainer(max_epochs=10, devices=1)
trainer.fit(lightning_module, dataloader)

# Solo i parametri LoRA verranno aggiornati!
```

## Configurazione Avanzata

### Applicare LoRA a diversi moduli

```python
# Adatta solo i classification head
config1 = LoRAConfig(
    enabled=True,
    rank=4,
    target_modules=["class_head"],
)

# Adatta tutti i moduli
config2 = LoRAConfig(
    enabled=True,
    rank=16,
    target_modules=["class_head", "mask_head", "upscale"],
)
```

### Parametri importanti

| Parametro | Valore | Effetto |
|-----------|--------|--------|
| `rank` | 4, 8, 16, 32 | Più grande = più parametri trainabili |
| `lora_alpha` | `2 * rank` | Fattore di scaling |
| `lora_dropout` | 0.05-0.2 | Regolarizzazione durante training |
| `freeze_base_model` | True | Solo LoRA è trainabile |

## Monitoraggio

### Vedere le statistiche dei parametri

```python
from models.lora_integration import print_lora_summary

model = EoMT(..., lora_config=lora_config)
print_lora_summary(model, lora_config)
```

Output:
```
============================================================
LoRA Configuration Summary
============================================================
Enabled: True
Rank: 8
Alpha: 16
Dropout: 0.1
Target Modules: ['class_head', 'mask_head']
Freeze Base Model: True
------------------------------------------------------------
Total Parameters: 270,548,736
Trainable Parameters: 1,123,456
LoRA Parameters: 1,123,456
Training Efficiency: 0.42%
============================================================
```

## Configurazione con YAML

Aggiungi al tuo config.yaml:

```yaml
model:
  class_path: models.eomt.EoMT
  init_args:
    encoder: ...
    num_classes: 19
    num_q: 100
    lora_config:
      class_path: models.lora_integration.LoRAConfig
      init_args:
        enabled: true
        rank: 8
        lora_alpha: 16
        target_modules: ["class_head", "mask_head"]
        freeze_base_model: true
```

Poi lancia con:
```bash
python main.py fit --config config.yaml
```

## Verificare che funzioni

```python
# Test rapido
import torch
from models.eomt import EoMT
from models.vit import ViT
from models.lora_integration import LoRAConfig

# Crea modello
vit = ViT(img_size=(640, 640))
lora_config = LoRAConfig(enabled=True, rank=8)
model = EoMT(vit, 19, 100, lora_config=lora_config)

# Forward pass
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    masks, classes = model(x)

print(f"Masklogits shape: {[m.shape for m in masks]}")
print(f"Class logits shape: {[c.shape for c in classes]}")
print("✓ LoRA funziona!")
```

## FAQ

**D: Quanto memory serva LoRA?**
A: ~5-10% in meno rispetto al modello completo. Maggiore risparmio se si usa `freeze_base_model=True`.

**D: Quali sono i valori di rank consigliati?**
A: Inizia con 8, aumenta a 16-32 se i risultati non sono buoni.

**D: Posso disabilitare LoRA?**
A: Sì: `LoRAConfig(enabled=False)`. Il modello funzionerà normalmente.

**D: Come salvo solo i pesi LoRA?**
A: 
```python
lora_state = {
    name: param for name, param in model.named_parameters() 
    if "lora_" in name and param.requires_grad
}
torch.save(lora_state, "lora_weights.pt")
```

**D: Posso usare LoRA con fine-tuning completo?**
A: Sì, non usare `freeze_base_model=True` per allenare tutto il modello.

## Risorse

- **Paper**: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **Documentazione**: Vedi [LORA_README.md](LORA_README.md)
- **Esempi**: Vedi [lora_examples.py](lora_examples.py)
- **Test**: `pytest tests/test_lora.py -v`

## Prossimi Step

1. ✅ Implementazione base completata
2. Integrare nei tuoi script di training
3. Sperimentare con diversi valori di rank
4. Monitorare le metriche durante il training
5. Confrontare risultati con/senza LoRA

Buon training! 🚀
