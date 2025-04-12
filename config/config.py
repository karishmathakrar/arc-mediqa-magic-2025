"""
Configuration settings for the MEDIQA project.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "2025_dataset")
TRAIN_DATA_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DATA_DIR, "images_train")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model settings
MODEL_ID = "google/gemma-3-4b-it"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.path.join(BASE_DIR, ".hf_cache")

# Set cache directory
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["HF_HOME"] = HF_CACHE_DIR

# Training settings
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03

# LoRA config params
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_RANK = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Inference settings
INFERENCE_BATCH_SIZE = 1  # Adjust based on available memory
MAX_NEW_TOKENS = 100

# System prompt for inference/training
SYSTEM_PROMPT = "You are an AI assistant answering medical questions based on clinical images."

# Gemma chat template
GEMMA_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
<start_of_turn>user
{% for content in message['content'] %}
{% if content['type'] == 'text' %}{{ content['text'] }}{% elif content['type'] == 'image' %}<image>{% endif %}
{% endfor %}
<end_of_turn>
{% elif message['role'] == 'assistant' %}
<start_of_turn>model
{% for content in message['content'] %}
{% if content['type'] == 'text' %}{{ content['text'] }}{% endif %}
{% endfor %}
<end_of_turn>
{% elif message['role'] == 'system' %}
<start_of_turn>system
{% for content in message['content'] %}
{% if content['type'] == 'text' %}{{ content['text'] }}{% endif %}
{% endfor %}
<end_of_turn>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<start_of_turn>model
{% endif %}
"""