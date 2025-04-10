"""
Model utility functions for the MEDIQA project.
"""
import torch
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Import config directly instead of using relative imports
from config.config import *

logger = logging.getLogger(__name__)

def load_processor(model_id=None, token=None):
    """
    Load the model processor.
    
    Args:
        model_id: HuggingFace model ID
        token: HuggingFace token
        
    Returns:
        HuggingFace processor
    """
    model_id = model_id or MODEL_ID
    token = token or HF_TOKEN
    
    logger.info(f"Loading processor for model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    
    # Set chat template
    processor.tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    processor.chat_template = GEMMA_CHAT_TEMPLATE
    
    # Ensure <image> is recognized as a token
    if "<image>" not in processor.tokenizer.get_vocab():
        logger.info("Adding <image> to tokenizer vocabulary")
        processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
    
    # Set boi_token for template rendering
    processor.boi_token = "<image>"
    processor.tokenizer.special_tokens_map['boi_token'] = "<image>"
    
    return processor


def load_model_for_inference(model_id=None, token=None, processor=None):
    """
    Load model for inference with 4-bit quantization.
    
    Args:
        model_id: HuggingFace model ID
        token: HuggingFace token
        processor: Processor to ensure model is compatible
        
    Returns:
        Loaded model
    """
    model_id = model_id or MODEL_ID
    token = token or HF_TOKEN
    
    # Check CUDA capability
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        logger.warning("GPU may not support bfloat16 natively")
    
    # Configure model parameters
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add quantization config for efficient inference
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )
    
    logger.info(f"Loading model for inference: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs, token=token)
    
    # Resize embeddings if processor changed the vocabulary
    if processor is not None and "<image>" in processor.tokenizer.get_vocab():
        logger.info("Resizing token embeddings to match processor")
        model.resize_token_embeddings(len(processor.tokenizer))
    
    return model


def setup_model_and_processor(model_id=None, token=None, for_training=False):
    """
    Setup model and processor together.
    
    Args:
        model_id: HuggingFace model ID
        token: HuggingFace token
        for_training: Whether to configure for training
        
    Returns:
        Tuple of (model, processor)
    """
    processor = load_processor(model_id, token)
    
    if for_training:
        model = load_model_for_training(model_id, token, processor)
    else:
        model = load_model_for_inference(model_id, token, processor)
        
    return model, processor


def load_model_for_training(model_id=None, token=None, processor=None):
    """
    Load model for training.
    
    Args:
        model_id: HuggingFace model ID
        token: HuggingFace token
        processor: Processor to ensure model is compatible
        
    Returns:
        Loaded model
    """
    model_id = model_id or MODEL_ID
    token = token or HF_TOKEN
    
    # Configure model parameters
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add quantization config for efficient training
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )
    
    logger.info(f"Loading model for training: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs, token=token)
    
    # Resize embeddings if processor changed the vocabulary
    if processor is not None and "<image>" in processor.tokenizer.get_vocab():
        logger.info("Resizing token embeddings to match processor")
        model.resize_token_embeddings(len(processor.tokenizer))
    
    return model