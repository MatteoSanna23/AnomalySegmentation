#!/bin/bash

# Configurazione percorsi
#CKPT="../../epoch_106-step_19902_eomt.ckpt"
CKPT="/teamspace/studios/this_studio/AML/AnomalySegmentation/eomt/eomt/kibcgcng/checkpoints/epoch=1-step=744.ckpt"
BASE_PATH="../../dataset_validation/Validation_Dataset"
OUTPUT_FILE="final_summary_eomt_logit_norm.txt"

echo "=== RISULTATI UNIVERSALI EO_MT ===" > $OUTPUT_FILE
echo "Data esecuzione: $(date)" >> $OUTPUT_FILE
echo "----------------------------------------------------------" >> $OUTPUT_FILE

# Elenco dataset: "Nome_Cartella:Estensione"
DATASETS=(
    "RoadAnomaly:*.jpg"
    "RoadAnomaly21:*.png"
    "RoadObsticle21:*.webp"
    "FS_LostFound_full:*.png"
    "fs_static:*.jpg"
)

for ENTRY in "${DATASETS[@]}"; do
    NAME="${ENTRY%%:*}"
    EXT="${ENTRY#*:}"
    
    echo "Processing $NAME..."
    echo ">>> DATASET: $NAME" >> $OUTPUT_FILE
    
    # Esegue la valutazione e salva i risultati nel log
    python evalAnomaly_eomt.py \
      --input "$BASE_PATH/$NAME/images/$EXT" \
      --loadWeights "$CKPT" >> $OUTPUT_FILE 2>&1
    
    echo "Done with $NAME."
    echo "----------------------------------------------------------" >> $OUTPUT_FILE
done

echo "=== VALUTAZIONE COMPLETATA. Risultati in: $OUTPUT_FILE ==="
