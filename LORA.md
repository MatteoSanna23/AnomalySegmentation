# Architettura LoRA in EoMT

Questo documento descrive l'implementazione e l'integrazione di **LoRA (Low-Rank Adaptation)** all'interno del progetto EoMT per la segmentazione di anomalie.

## 1. Cos'è LoRA? (Teoria in Breve)

LoRA è una tecnica di *Parameter-Efficient Fine-Tuning* (PEFT). Invece di aggiornare tutti i pesi di una matrice densa pre-addestrata durante il fine-tuning, LoRA congela i pesi originali e inietta matrici di rango inferiore (low-rank) addestrabili.

### Matematica di base
Dato un layer lineare pre-addestrato $W_0 \in \mathbb{R}^{d \times k}$, l'aggiornamento dei pesi viene rappresentato come:

$$W = W_0 + \Delta W = W_0 + B A$$

Dove:
* $W_0$: Pesi originali (Congelati / `requires_grad=False`).
* $B \in \mathbb{R}^{d \times r}$: Matrice LoRA B (Inizializzata a zero).
* $A \in \mathbb{R}^{r \times k}$: Matrice LoRA A (Inizializzata random Gaussian).
* $r$: Il **rango** (rank), un iperparametro molto piccolo (es. 8, 16) rispetto alla dimensione originale $d$.

L'output del layer diventa:
$$h = W_0 x + \frac{\alpha}{r} (BAx)$$
Dove $\alpha$ è un fattore di scaling costante.

### Vantaggi nel Progetto
1.  **VRAM ridotta:** Non dobbiamo salvare gli stati dell'ottimizzatore per i milioni di parametri del ViT (DINOv2), ma solo per le piccole matrici A e B.
2.  **Anti-Catastrophic Forgetting:** Il modello originale DINOv2 non viene toccato, mantenendo le feature robuste apprese su dataset massivi.
3.  **Modularità:** È possibile scambiare diversi adattatori LoRA per task diversi senza ricaricare il backbone pesante.

---

## 2. Implementazione nel Progetto

L'integrazione di LoRA in questo repository avviene attraverso tre componenti principali:

### A. Il "Mattone": `models/lora.py`
Definisce la classe `LoRALinear`, che sostituisce il layer `nn.Linear` standard di PyTorch.
* Contiene i pesi originali (`weight`, `bias`) congelati.
* Contiene le matrici `lora_A` e `lora_B` addestrabili.
* Nel metodo `forward`, somma il risultato del linear originale con quello del ramo low-rank.

### B. Il "Costruttore": `models/lora_integration.py`
Questa è la logica di iniezione.
1.  **Ricerca Ricorsiva:** Scansiona l'intero modello ViT.
2.  **Targeting:** Cerca i layer lineari che corrispondono ai nomi specificati nella configurazione (es. "qkv", "fc1").
3.  **Sostituzione:** Sostituisce fisicamente l'oggetto `nn.Linear` con un'istanza di `LoRALinear`, copiando i pesi originali.
4.  **Congelamento Selettivo (`_freeze_base_model`):**
    * Congela tutto il backbone ViT.
    * Sblocca i layer LoRA.
    * **Cruciale:** Sblocca le teste di classificazione (`class_head`, `mask_head`) e i layer di normalizzazione (`norm`), necessari per adattare il modello al nuovo task di segmentazione.

### C. L'Integrazione: `models/eomt.py`
Nel costruttore della classe `EoMT`:
1.  Viene letta la `LoRAConfig`.
2.  Se abilitato, chiama `apply_lora_to_vit(self, config)`.
3.  Questo trasforma il backbone DINOv2 statico in un modello adattabile prima dell'inizio del training.

---

## 3. Configurazione e Target

La configurazione di LoRA è definita nei file YAML (es. `configs/.../eomt_base_640_lora.yaml`).

### Moduli Targettati
Nel Vision Transformer (ViT), applichiamo LoRA ai componenti chiave dei blocchi Transformer:

* **`qkv`**: La proiezione Query-Key-Value nell'Attention Mechanism. Adattare questo permette al modello di "guardare" parti diverse dell'immagine rilevanti per le anomalie.
* **`proj`**: La proiezione di output dell'Attention.
* **`fc1`, `fc2`**: I layer del Feed-Forward Network (MLP). Qui risiede gran parte della "conoscenza" semantica del Transformer.

### Esempio di Configurazione YAML

```yaml
model:
  init_args:
    network:
      init_args:
        # ...
        lora_config:
          class_path: models.lora_integration.LoRAConfig
          init_args:
            enabled: true
            rank: 8             # Dimensione delle matrici low-rank (basso = più leggero)
            lora_alpha: 16      # Fattore di scaling (spesso 2x rank)
            lora_dropout: 0.1   # Regolarizzazione
            
            # Layer dove iniettare LoRA
            target_modules: ["qkv", "proj", "fc1", "fc2"]
            
            # Congela il ViT originale, allena solo LoRA + Heads
            freeze_base_model: true