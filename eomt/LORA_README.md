# LoRA (Low-Rank Adaptation) Integration

Questo documento descrive l'implementazione di LoRA nel progetto EoMT (End-to-end Multi-Task) per l'anomaly segmentation.

## Cos'è LoRA?

**LoRA (Low-Rank Adaptation)** è una tecnica di fine-tuning efficiente descritta nel paper:
> *LoRA: Low-Rank Adaptation of Large Language Models*
> https://arxiv.org/abs/2106.09685

Invece di aggiornare tutti i parametri del modello durante il fine-tuning, LoRA aggiunge matrici di basso rango trainabili:

$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$$

dove:
- $\mathbf{W}_0$ sono i pesi originali (congelati)
- $\mathbf{A} \in \mathbb{R}^{d \times r}$ e $\mathbf{B} \in \mathbb{R}^{r \times d}$ sono le matrici LoRA trainabili
- $r$ è il "rank" (di solito molto minore di $d$)
- $\alpha$ è un fattore di scaling

## Vantaggi

1. **Riduzione dei parametri trainabili**: Solo ~1-5% dei parametri originali
2. **Efficienza memoria**: Meno memoria GPU richiesta
3. **Velocità di training**: Fine-tuning più veloce
4. **Portabilità**: I pesi LoRA sono leggeri e facili da condividere
5. **Compatibilità**: Mantiene le capacità del modello pre-trainato

## Struttura dell'Implementazione

### File principali

```
eomt/
├── models/
│   ├── lora.py                    # Core LoRA implementation
│   ├── lora_integration.py        # Integration utilities
│   └── eomt.py                    # EoMT model with LoRA support
├── lora_examples.py               # Usage examples
└── LORA_README.md                 # This file
```

### Moduli LoRA

#### `models/lora.py`

**LoRALinear**: Sostituisce i layer lineari standard con layer LoRA
```python
class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.1)
```

**LoRAAttention**: Adattamento LoRA per i layer di attenzione
```python
class LoRAAttention(nn.Module):
    """LoRA adaptation for attention layers"""
```

**Utility functions**:
- `replace_linear_with_lora()`: Sostituisce i layer lineari in un modulo
- `freeze_lora_params()`: Congela i pesi originali
- `count_lora_parameters()`: Conta i parametri LoRA

#### `models/lora_integration.py`

**LoRAConfig**: Classe di configurazione
```python
class LoRAConfig:
    enabled: bool = True
    rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = ["class_head", "mask_head"]
    freeze_base_model: bool = False
```

**Funzioni di integrazione**:
- `apply_lora_to_vit()`: Applica LoRA al modello ViT/EoMT
- `get_lora_stats()`: Ottiene statistiche sui parametri
- `print_lora_summary()`: Stampa un riepilogo della configurazione

## Come Usare

### 1. Configurazione Base

Nel tuo file di configurazione YAML o nel codice:

```python
from models.lora_integration import LoRAConfig
from models.eomt import EoMT

# Crea la configurazione LoRA
lora_config = LoRAConfig(
    enabled=True,
    rank=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["class_head", "mask_head"],
    freeze_base_model=True,
)

# Crea il modello con LoRA
model = EoMT(
    encoder=encoder,
    num_classes=num_classes,
    num_q=num_q,
    lora_config=lora_config
)
```

### 2. Integrazione con Lightning

Nel tuo `LightningModule`:

```python
from lightning.pytorch import LightningModule
from models.lora_integration import LoRAConfig, print_lora_summary

class YourLightningModule(LightningModule):
    def __init__(self, lora_config: LoRAConfig = None, ...):
        super().__init__()
        
        lora_config = lora_config or LoRAConfig(enabled=False)
        
        self.network = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            lora_config=lora_config
        )
        
        # Print LoRA configuration
        print_lora_summary(self.network, lora_config)
    
    def configure_optimizers(self):
        # Optimizer per i parametri trainabili (solo LoRA se freeze_base_model=True)
        optimizer = torch.optim.AdamW(
            [p for p in self.network.parameters() if p.requires_grad],
            lr=self.hparams.lr
        )
        return optimizer
```

### 3. Configurazione da YAML

Nel tuo file di configurazione Lightning CLI:

```yaml
model:
  class_path: eomt.models.eomt.EoMT
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
        lora_dropout: 0.1
        target_modules: ["class_head", "mask_head"]
        freeze_base_model: true
```

## Parametri LoRA

### rank (default: 8)
Dimensione della matrice di basso rango. Valori tipici: 4, 8, 16, 32.
- Valori **più piccoli** = meno parametri, ma meno espressività
- Valori **più grandi** = più parametri, maggiore espressività

### lora_alpha (default: 16)
Fattore di scaling per l'output LoRA. 
$$\text{scaling} = \frac{\text{lora\_alpha}}{\text{rank}}$$

Suggeriamo `lora_alpha = 2 * rank`

### lora_dropout (default: 0.1)
Dropout applicato alle matrici LoRA durante il training.

### target_modules (default: ["class_head", "mask_head"])
Moduli a cui applicare LoRA. Opzioni per EoMT:
- `"class_head"`: Layer di classificazione
- `"mask_head"`: Layer di predizione delle maschere
- `"upscale"`: Layer di upsampling

### freeze_base_model (default: False)
Se True, congela tutti i parametri non-LoRA, consentendo l'addestramento SOLO dei parametri LoRA.

## Statistiche dei Parametri

Con l'implementazione di default (`rank=8`):

```
Original Model:
  Total Parameters: ~270M (per DINOv2-base)
  
With LoRA on [class_head, mask_head]:
  LoRA Parameters: ~1M (0.4% of total)
  Trainable: ~1M
  
Training Efficiency: Fine-tune con 99.6% riduzione di parametri trainabili
```

## Best Practices

1. **Iniziare con valori conservativi**: `rank=8`, `lora_alpha=16`
2. **Congelare il modello base**: `freeze_base_model=True` per il fine-tuning
3. **Applicare a specifici moduli**: Iniziare con head della classificazione
4. **Sperimentare con il rank**: Aumentare se i risultati sono insufficienti
5. **Monitorare i parametri trainabili**: Usare `print_lora_summary()` durante l'init

## Esempi

Vedi [lora_examples.py](lora_examples.py) per esempi completi:

```python
from lora_examples import (
    create_dummy_lora_config,
    example_basic_usage,
    example_with_config_file,
    example_parameter_comparison,
    example_training_loop,
)
```

## Salvataggio e Caricamento

### Salvare solo i pesi LoRA

```python
# Salvare
lora_state = {
    name: param for name, param in model.named_parameters() 
    if "lora_" in name
}
torch.save(lora_state, "lora_weights.pt")

# Caricare
model = EoMT(..., lora_config=lora_config)
lora_state = torch.load("lora_weights.pt")
model.load_state_dict(lora_state, strict=False)
```

### Fusione dei pesi LoRA

Per produrre un modello standalone senza LoRA:

```python
def merge_lora_weights(model):
    """Merge LoRA weights into base weights"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                # Merge: weight = weight + (alpha/r) * B @ A
                delta = module.lora_b @ module.lora_a
                delta = delta * module.scaling
                module.weight.data += delta.t()
                # Riconverti a Linear standard
                # ... implementation details ...
```

## Troubleshooting

### Memoria insufficiente
- Ridurre il `rank`
- Applicare LoRA solo ai moduli specificati
- Usare `freeze_base_model=True`

### Risultati peggiori
- Aumentare il `rank`
- Aumentare `lora_alpha`
- Aggiungere più moduli a `target_modules`
- Verificare il learning rate

### Parametri non aggiornati
- Verificare che `requires_grad=True` per i parametri LoRA
- Controllare che non siano stati congelati accidentalmente
- Usare `print_lora_summary()` per diagnosticare

## Riferimenti

1. **Paper LoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. **DINOv2**: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
3. **PyTorch Lightning**: https://lightning.ai/

## Domande e Contributi

Per domande o contributi al modulo LoRA, contatta il team di MPS Lab @ TU/e.
