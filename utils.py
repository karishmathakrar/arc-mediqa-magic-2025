"""
Utility functions for the MEDIQA project.
"""
import os
import gc
import sys
import json
import torch
from PIL import Image
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_system_info():
    """Print information about the system configuration."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python version info: {sys.version_info}")
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

def create_dummy_image(size=(224, 224), color='black'):
    """Create a small black image as a placeholder."""
    return Image.new('RGB', size, color=color)

def process_vision_info(messages):
    """
    Extract images from a structured messages list.
    Returns a list of PIL Image objects in RGB format.
    
    Args:
        messages: List of message objects
        
    Returns:
        List of PIL images
    """
    image_inputs = []
    
    for msg in messages:
        content = msg.get("content", [])
        
        if not isinstance(content, list):
            content = [content]
        
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                
                if hasattr(image, 'convert'):
                    image = image.convert("RGB")
                    image_inputs.append(image)
    
    return image_inputs

def load_json_file(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None

def ensure_directory(directory):
    """Ensure that a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

def is_valid_image(image_path):
    """Check if an image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False