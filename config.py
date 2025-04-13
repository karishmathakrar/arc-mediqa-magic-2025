"""
Configuration settings for the MEDIQA project.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "2025_dataset", "train")
IMAGES_DIR = os.path.join(DATASET_DIR, "images_train")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
VALIDATION_DIR = os.path.join(BASE_DIR, "2025_dataset", "validation")
VALIDATION_IMAGES_DIR = os.path.join(VALIDATION_DIR, "images_validation")

# Create needed directories
for directory in [PROCESSED_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model settings
MODEL_ID = "google/gemma-3-4b-it"

# HuggingFace cache settings
HF_CACHE_DIR = os.path.join(BASE_DIR, ".hf_cache")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["HF_HOME"] = HF_CACHE_DIR

# Training settings
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03

# LoRA config params
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_RANK = 16

# Inference settings
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.5
TOP_P = 0.9

# System prompt for inference/training
SYSTEM_PROMPT = "You are a medical image analysis assistant. Your only task is to examine the provided clinical images and select the exact option text that best describes what you see. Note this is not the full context so if you are unsure or speculate other regions being affected, respond with 'Not mentioned'. You must respond with the full text of one of the provided options, exactly as written. Do not include any additional words or reasoning. Given the medical context, err on the side of caution when uncertain."

# Output directories
BASE_MODEL_DIR = os.path.join(OUTPUTS_DIR, "base_model")
FINETUNED_MODEL_DIR = os.path.join(OUTPUTS_DIR, "finetuned_model")
MERGED_MODEL_DIR = os.path.join(OUTPUTS_DIR, "merged_model")