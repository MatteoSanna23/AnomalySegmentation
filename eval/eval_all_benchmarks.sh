#!/bin/bash

# Configurazione percorsi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#CKPT="../../epoch_106-step_19902_eomt.ckpt"
CKPT="/teamspace/studios/this_studio/AML/checkpoints/eomt_djsj53u4_all_checkpoints_epoch=9-step=3720.ckpt"
BASE_PATH="/teamspace/studios/this_studio/AML/dataset_validation/Validation_Dataset"
OUTPUT_FILE="/teamspace/studios/this_studio/AML/AnomalySegmentation/eval/final_summaries/final_summary_eomt_all_combined.txt"

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
    python "$SCRIPT_DIR/evalAnomaly_cut_paste.py" \
      --input "$BASE_PATH/$NAME/images/$EXT" \
      --loadWeights "$CKPT" >> $OUTPUT_FILE 2>&1
    
    echo "Done with $NAME."
    echo "----------------------------------------------------------" >> $OUTPUT_FILE
done

echo "=== VALUTAZIONE COMPLETATA. Risultati in: $OUTPUT_FILE ==="
