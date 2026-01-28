#!/bin/bash

# =============================================================================
# Universal Evaluation Script for EoMT
# =============================================================================
# This script automates the evaluation of the Anomaly Segmentation model across
# multiple datasets (RoadAnomaly, LostAndFound, etc.).
# It iterates through a list of datasets, runs the python evaluation script,
# and aggregates all results into a single summary text file.
# =============================================================================

# --- Configuration & Paths ---

# Get the directory where this script is located to ensure relative paths work
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the model checkpoint to evaluate (LoRA fine-tuned weights)
CKPT="/teamspace/studios/this_studio/AnomalySegmentation/eomt/6w549yeo/checkpoints/epoch=0-step=1487.ckpt"

# Base directory containing the validation datasets
BASE_PATH="/teamspace/studios/this_studio/Validation_Dataset"

# Output file where all evaluation metrics will be appended
OUTPUT_FILE="/teamspace/studios/this_studio/AnomalySegmentation/eval/final_summaries/final_summary_eomt_lora.txt"

# Initialize the summary file with a header and timestamp
echo "=== UNIVERSAL RESULTS EO_MT ===" > $OUTPUT_FILE
echo "Execution Date: $(date)" >> $OUTPUT_FILE
echo "----------------------------------------------------------" >> $OUTPUT_FILE

# --- Dataset Definition ---
# format: "FolderName:FileExtension"
# This allows handling different image formats (jpg, png, webp) for each dataset
DATASETS=(
    "RoadAnomaly:*.jpg"
    "RoadAnomaly21:*.png"
    "RoadObsticle21:*.webp"
    "FS_LostFound_full:*.png"
    "fs_static:*.jpg"
)

# --- Main Evaluation Loop ---

for ENTRY in "${DATASETS[@]}"; do
    # String manipulation to split "Name:Ext"
    NAME="${ENTRY%%:*}"   # Extract substring BEFORE the colon (Dataset Name)
    EXT="${ENTRY#*:}"     # Extract substring AFTER the colon (File Extension)
    
    echo "Processing $NAME..."
    
    # Write dataset header to the output file
    echo ">>> DATASET: $NAME" >> $OUTPUT_FILE
    
    # Run the Python evaluation script
    # $BASE_PATH/$NAME/images/$EXT -> expands to e.g., .../RoadAnomaly/images/*.jpg
    # >> $OUTPUT_FILE 2>&1 -> Redirects both Standard Output (stdout) and 
    #                         Standard Error (stderr) to the summary file.
    python "$SCRIPT_DIR/evalAnomaly_lora.py" \
      --input "$BASE_PATH/$NAME/images/$EXT" \
      --loadWeights "$CKPT" >> $OUTPUT_FILE 2>&1
    
    echo "Done with $NAME."
    echo "----------------------------------------------------------" >> $OUTPUT_FILE
done

echo "=== EVALUATION COMPLETED. Results saved to: $OUTPUT_FILE ==="