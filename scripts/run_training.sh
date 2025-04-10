#!/bin/bash
set -xe
export NO_REINSTALL=1

# --- Job info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory:"
free -h

# --- Activate environment ---
source ~/scratch/gemma/venv/bin/activate

# --- GPU check ---
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

# --- Start GPU monitoring ---
NVIDIA_LOG_FILE=Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
python ~/your_project/scripts/nvidia-logs.sh monitor "$NVIDIA_LOG_FILE" --interval 15 &
nvidia_logs_pid=$!
echo "Monitoring GPU usage (PID $nvidia_logs_pid)"

# --- Define paths ---
export PROCESSED_DATA_DIR=~/scratch/gemma/processed_data
export OUTPUT_DIR=~/scratch/gemma/output
export DATASET_NAME=your_dataset_name

# --- Run your training script ---
python train_gemma_multimodal.py \
    --data_dir "$PROCESSED_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --use_flash_attention False \
    --bf16 True

# --- Finalize GPU logs ---
python ~/your_project/scripts/nvidia-logs.sh parse "$NVIDIA_LOG_FILE"