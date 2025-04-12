#!/bin/bash
# Complete script to run inference and evaluation

# Default values
DATA_DIR="2025_dataset/train"
MODEL_ID="google/gemma-3-4b-it"  # Default to baseline model
OUTPUT_DIR="evaluation_results"
LIMIT_ARG=""
REPROCESS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_ID="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT_ARG="--limit $2"
      shift 2
      ;;
    --reprocess)
      REPROCESS="--reprocess"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--model_path PATH] [--data_dir DIR] [--output_dir DIR] [--limit N] [--reprocess]"
      exit 1
      ;;
  esac
done

# Set reference JSON path based on data directory
REF_JSON="${DATA_DIR}/train_cvqa.json"

# Create descriptive suffix based on model
MODEL_SUFFIX=$(basename "$MODEL_ID" | tr '/' '_')

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running inference with model: $MODEL_ID"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$LIMIT_ARG" ]; then
  echo "Using limit: ${LIMIT_ARG#--limit }"
fi

# Step 1: Run inference with index conversion
echo "Running inference with index conversion..."
python inference.py \
  --data_dir "$DATA_DIR" \
  --model_id "$MODEL_ID" \
  --processed_dir "processed_data_${MODEL_SUFFIX}" \
  --output_file "${OUTPUT_DIR}/inference_results_${MODEL_SUFFIX}.csv" \
  --json_output "${OUTPUT_DIR}/data_cvqa_sys_${MODEL_SUFFIX}.json" \
  --indexed_json_output "${OUTPUT_DIR}/data_cvqa_sys_indices_${MODEL_SUFFIX}.json" \
  $LIMIT_ARG \
  $REPROCESS

# Step 2: Run evaluation on the indexed JSON
echo "Running evaluation..."
python run_cvqa_eval.py \
  "$REF_JSON" \
  "${OUTPUT_DIR}/data_cvqa_sys_indices_${MODEL_SUFFIX}.json" \
  "$OUTPUT_DIR"

# Rename output files to include model suffix
mv "${OUTPUT_DIR}/scores.json" "${OUTPUT_DIR}/scores_${MODEL_SUFFIX}.json"
mv "${OUTPUT_DIR}/scores_cvqa.json" "${OUTPUT_DIR}/scores_cvqa_${MODEL_SUFFIX}.json"

# Step 3: Show summary of results
echo "Evaluation complete. Summary of results:"
echo "----------------------------------------"
python -c "
import json
with open('${OUTPUT_DIR}/scores_${MODEL_SUFFIX}.json', 'r') as f:
    data = json.load(f)
print(f\"Model: ${MODEL_ID}\")
print(f\"Overall accuracy: {data['accuracy_all']:.4f}\")
print(f\"Number of encounters evaluated: {data['number_evaluated_encounters']}\")
print(\"\\nCategory accuracies:\")
for key in sorted(data.keys()):
    if key.startswith('accuracy_CQID') and key != 'accuracy_all':
        print(f\"  {key.replace('accuracy_', '')}: {data[key]:.4f}\")
"