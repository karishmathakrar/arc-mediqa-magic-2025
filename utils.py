"""
Utility functions for the MEDIQA project.
"""
import os
import sys
import gc
import json
import torch
import logging
from PIL import Image
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_system_info():
    """Print information about the system configuration."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared")

def load_json_file(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}")
        return False

def load_pickle(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        return None

def is_valid_image(image_path):
    """Check if an image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False

def clean_generated_text(text):
    """Clean up the generated text to extract just the answer."""
    lines = text.strip().split('\n')
    model_content = False
    answer_lines = []
    
    for line in lines:
        if '<start_of_turn>model' in line or line.strip() == "model":
            model_content = True
            continue
        if model_content and line.strip() and not line.startswith("<") and not any(tag in line for tag in ["start_of_turn", "end_of_turn"]):
            answer_lines.append(line.strip())
    
    # If we found model content, return it
    if answer_lines:
        return " ".join(answer_lines)
    
    # If all else fails, return the last non-empty line
    for line in reversed(lines):
        if line.strip() and not line.startswith("<") and not any(tag in line for tag in ["start_of_turn", "end_of_turn"]):
            return line.strip()
    
    return ""

def create_dummy_image(size=(224, 224), color='black'):
    """Create a small black image as a placeholder."""
    return Image.new('RGB', size, color=color)