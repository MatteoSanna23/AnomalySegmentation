# 📚 LoRA Implementation - Complete Documentation Index

Benvenuto nella documentazione di LoRA per il progetto EoMT!

Questo file serve come indice principale per navigare tutta la documentazione e gli esempi.

## 🚀 Quick Start (60 secondi)

Se sei in fretta e vuoi iniziare subito:

**1. Leggi**: [LORA_QUICKSTART.md](LORA_QUICKSTART.md) (2 minuti)  
**2. Copia**: Un esempio da [lora_examples.py](lora_examples.py)  
**3. Integra**: Nel tuo training code  
**4. Addestra**: `python main.py fit --config config_with_lora.yaml`

---

## 📖 Documentazione Principale

### [LORA_QUICKSTART.md](LORA_QUICKSTART.md) ⚡
**Per chi ha fretta: 60 secondi di essenziale**
- Setup base in 3 righe
- Configurazione YAML
- FAQ comuni

### [LORA_README.md](LORA_README.md) 📘
**Documentazione completa e dettagliata**
- Cos'è LoRA e come funziona
- Struttura dell'implementazione
- API di tutte le funzioni
- Best practices
- Troubleshooting
- Salvataggio e caricamento

### [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) 📋
**Overview tecnico dell'implementazione**
- Struttura dei file
- Componenti principali
- Vantaggi della soluzione
- Prossimi step
- FAQ

### [INTEGRATION_GUIDE.py](INTEGRATION_GUIDE.py) 🔧
**Guida all'integrazione step-by-step**
- Checklist completa
- Minimal working example
- Configurazioni comuni
- Consigli sui parametri

---

## 💡 Esempi e Codice

### [lora_examples.py](lora_examples.py)
**5 esempi pratici completi**

1. **Basic Usage** - Hello World di LoRA
2. **From Dictionary** - Caricare config da dict
3. **Parameter Comparison** - Confronto prima/dopo
4. **Training Loop** - Come allenare il modello
5. **Merge Weights** - Come fondere i pesi

**Come usare:**
```bash
python lora_examples.py
```

### [example_lightning_integration.py](example_lightning_integration.py)
**Integrazione completa con PyTorch Lightning**

- `EoMTLightningModule` - Lightning module pronto all'uso
- Configurazioni YAML di esempio
- Setup optimizer e scheduler
- Training/validation/test steps

**Cosa contiene:**
```python
# Lightning module con LoRA support integrato
model = EoMTLightningModule(
    enable_lora=True,
    lora_rank=8,
    ...
)

# YAML config incluso
# Pronto per: python main.py fit --config config_with_lora.yaml
```

---

## 🔧 Implementazione Tecnica

### File Core

#### [models/lora.py](models/lora.py)
Implementazione LoRA basso-livello

**Classi:**
- `LoRALinear` - Layer lineare con LoRA
- `LoRAAttention` - Attention layer con LoRA

**Funzioni:**
- `replace_linear_with_lora()` - Sostituisci layer automaticamente
- `freeze_lora_params()` - Congela pesi
- `count_lora_parameters()` - Conta parametri trainabili

#### [models/lora_integration.py](models/lora_integration.py)
Utilità di integrazione ad alto livello

**Classe:**
- `LoRAConfig` - Configurazione LoRA

**Funzioni:**
- `apply_lora_to_vit()` - Applica LoRA al modello
- `get_lora_stats()` - Statistiche parametri
- `print_lora_summary()` - Stampa riepilogo

#### [models/eomt.py](models/eomt.py)
Modello EoMT (modificato)

**Modifiche:**
- Aggiunto parametro `lora_config` al costruttore
- Integrazione automatica di LoRA se configurato

---

## ✅ Test

### [tests/test_lora.py](tests/test_lora.py)

**Suite di test completa**

Run:
```bash
pytest tests/test_lora.py -v
```

Testa:
- ✅ Creazione layer LoRA
- ✅ Forward pass
- ✅ Gradient flow
- ✅ Parameter counting
- ✅ Configuration
- ✅ Integration
- ✅ End-to-end training

---

## 🎯 Come usare questa documentazione

### Sono completamente nuovo a LoRA:
1. Leggi [LORA_QUICKSTART.md](LORA_QUICKSTART.md)
2. Vedi [lora_examples.py](lora_examples.py) - Example 1
3. Copia il codice nel tuo progetto
4. Vai a "Voglio integrare nel mio codice"

### Voglio capire la teoria:
1. Leggi [LORA_README.md](LORA_README.md) - sezione "Cos'è LoRA?"
2. Controlla i riferimenti al paper
3. Guarda l'implementazione in [models/lora.py](models/lora.py)

### Voglio integrare nel mio codice:
1. Vedi [INTEGRATION_GUIDE.py](INTEGRATION_GUIDE.py) - Checklist
2. Copia il "Minimal Working Example"
3. Segui i step 1-3
4. Testa con `python main.py fit --config config_with_lora.yaml`

### Uso PyTorch Lightning:
1. Leggi [example_lightning_integration.py](example_lightning_integration.py)
2. Usa `EoMTLightningModule` come template
3. Configura via YAML

### Voglio sperimentare:
1. Vedi [LORA_README.md](LORA_README.md) - Parametri LoRA
2. Prova diverse configurazioni in [lora_examples.py](lora_examples.py)
3. Monitora con `print_lora_summary()`

### Ho un problema:
1. Vedi [LORA_README.md](LORA_README.md) - Troubleshooting
2. Vedi [LORA_QUICKSTART.md](LORA_QUICKSTART.md) - FAQ
3. Controlla [tests/test_lora.py](tests/test_lora.py) - Vedi come i test creano i modelli

---

## 📊 Roadmap di lettura consigliato

```
┌─────────────────────────────────────────────────────────┐
│ Livello 1: PRINCIPIANTE (30 min totali)               │
├─────────────────────────────────────────────────────────┤
│ ⏱️  5 min  → LORA_QUICKSTART.md                         │
│ ⏱️ 10 min  → lora_examples.py (Example 1)              │
│ ⏱️ 15 min  → Prova il Minimal Working Example           │
└─────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Livello 2: INTERMEDIO (1.5 ore)                        │
├─────────────────────────────────────────────────────────┤
│ ⏱️ 20 min  → LORA_README.md (sezioni principali)       │
│ ⏱️ 20 min  → lora_examples.py (tutti gli esempi)       │
│ ⏱️ 20 min  → INTEGRATION_GUIDE.py                       │
│ ⏱️ 20 min  → Integra nel tuo codice                    │
│ ⏱️  10 min  → Testa con pytest                          │
└─────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Livello 3: AVANZATO (2-3 ore)                          │
├─────────────────────────────────────────────────────────┤
│ ⏱️ 30 min  → LORA_README.md (tutto in dettaglio)       │
│ ⏱️ 30 min  → models/lora.py (analisi codice)           │
│ ⏱️ 30 min  → models/lora_integration.py                │
│ ⏱️ 30 min  → example_lightning_integration.py          │
│ ⏱️ 30 min  → tests/test_lora.py (capire i test)        │
│ ⏱️  variable→ Sperimentazione con il tuo dataset        │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Lookup Veloce

### Domanda: Come abilito LoRA?
→ [LORA_QUICKSTART.md](LORA_QUICKSTART.md) - Sezione "Uso Base"

### Domanda: Cosa significa rank?
→ [LORA_README.md](LORA_README.md) - Sezione "Parametri LoRA"

### Domanda: Come faccio il fine-tuning?
→ [example_lightning_integration.py](example_lightning_integration.py)

### Domanda: Come salvo i pesi LoRA?
→ [LORA_README.md](LORA_README.md) - Sezione "Salvataggio e Caricamento"

### Domanda: Ho memoria insufficiente
→ [LORA_README.md](LORA_README.md) - Sezione "Troubleshooting"

### Domanda: Come mi connetto a W&B?
→ [INTEGRATION_GUIDE.py](INTEGRATION_GUIDE.py) - Step 5

### Domanda: Dove sono i test?
→ [tests/test_lora.py](tests/test_lora.py)

### Domanda: Mi serve un esempio completo
→ [lora_examples.py](lora_examples.py) - Example 4 (Training Loop)

---

## 📋 Checklist di integrazione

```
LoRA Integration Checklist:

□ Ho letto LORA_QUICKSTART.md
□ Ho eseguito lora_examples.py
□ Ho importato LoRAConfig nel mio codice
□ Ho creato la configurazione LoRA
□ Ho passato lora_config a EoMT
□ Ho eseguito il test forward pass
□ Ho verificato print_lora_summary()
□ Ho aggiornato configure_optimizers()
□ Ho avviato il training
□ Ho confrontato con il baseline

Fatto! ✨
```

---

## 💬 Domande Comuni

**D: Per quanto tempo leggo la documentazione?**
A: Dipende dal livello:
- Principiante: 30 minuti
- Intermedio: 1.5 ore
- Avanzato: 2-3 ore

**D: Devo leggere tutto?**
A: No! Vedi la "Roadmap di lettura consigliato" sopra.

**D: Dove comincio?**
A: [LORA_QUICKSTART.md](LORA_QUICKSTART.md)

**D: Voglio solo il codice**
A: [lora_examples.py](lora_examples.py) - Example 1

**D: Non funziona**
A: [LORA_README.md](LORA_README.md) - Troubleshooting

---

## 📞 Support

Se hai domande:

1. **Per implementazione tecnica**: Vedi [models/lora.py](models/lora.py) e [models/lora_integration.py](models/lora_integration.py)
2. **Per come usare**: Vedi [lora_examples.py](lora_examples.py)
3. **Per integrazione**: Vedi [INTEGRATION_GUIDE.py](INTEGRATION_GUIDE.py)
4. **Per troubleshooting**: Vedi [LORA_README.md](LORA_README.md)
5. **Per test**: Vedi [tests/test_lora.py](tests/test_lora.py)

---

## 📊 Statistiche

- **File di codice**: 4 (lora.py, lora_integration.py, eomt.py modificato, __init__.py modificato)
- **File di documentazione**: 5 (LORA_README.md, LORA_QUICKSTART.md, IMPLEMENTATION_SUMMARY.md, INTEGRATION_GUIDE.py, questo file)
- **File di esempi**: 2 (lora_examples.py, example_lightning_integration.py)
- **File di test**: 1 (test_lora.py)
- **Linee di codice**: ~2000
- **Linee di documentazione**: ~2000
- **Linee di test**: ~400

---

## 🎓 Risorse Esterne

- **Paper LoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **DINOv2**: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- **PyTorch Lightning**: https://lightning.ai/
- **timm library**: https://github.com/huggingface/pytorch-image-models

---

## ✨ Prossimi Step

Sei pronto? Ecco cosa fare:

1. **Subito**: Leggi [LORA_QUICKSTART.md](LORA_QUICKSTART.md)
2. **Tra 5 minuti**: Esegui [lora_examples.py](lora_examples.py)
3. **Tra 10 minuti**: Copia il codice nel tuo progetto
4. **Tra 20 minuti**: Avvia il training con LoRA
5. **Domani**: Confronta i risultati con il baseline

Buon luck! 🚀

---

**Status**: ✅ Documentazione completa  
**Ultimo aggiornamento**: Gennaio 2026  
**Mantainer**: Mobile Perception Systems Lab @ TU/e
