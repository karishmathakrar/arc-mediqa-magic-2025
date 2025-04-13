#!/usr/bin/env python
"""
Refactored training script for the MEDIQA project.
Uses modular code structure and reduces redundancy.
"""
import os
import sys
import argparse
import logging
import torch
import gc
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Add parent directory to path to allow absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modules
from config.config import *
from data.loader import MedicalImageDataset, create_collate_fn
from models.model_utils import load_model_for_training, load_processor
from trainer.trainer import train_model, merge_and_save_lora_model
import utils.utils as utils

# For preprocessing integration
from scripts.preprocess import preprocess_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune model on medical images")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed_data/train",
        help="Directory with processed data for training"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="google/gemma-3-4b-it",
        help="Path to model or HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/fine_tuned",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--merged_dir", 
        type=str, 
        default="models/merged",
        help="Directory to save merged model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Per-device batch size for training"
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run preprocessing step before training"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for processing"
    )
    parser.add_argument(
        "--skip_merge", 
        action="store_true",
        help="Skip merging LoRA weights with base model"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Clear GPU memory at start
    utils.clear_gpu_memory()
    
    # Print system info
    utils.print_system_info()
    
    # Step 1: Preprocessing (if needed)
    if args.preprocess:
        logger.info("Starting preprocessing step...")
        csv_file = os.path.join(args.data_dir, "dataset_train.csv")
        
        # Use the preprocess_dataset function from preprocess.py
        _, args.processed_dir = preprocess_dataset(
            csv_file=csv_file,
            data_dir=args.data_dir,
            output_dir=args.processed_dir,
            mode="train",
            limit=args.limit,
            batch_size=5,  # Smaller batch size for preprocessing
            reprocess=True
        )
    else:
        logger.info(f"Using existing processed data from {args.processed_dir}")
    
    # Step 2: Load model and processor
    # Load HuggingFace token from environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    logger.info(f"Loading processor from: {args.model_id}")
    processor = load_processor(args.model_id, hf_token)
    
    logger.info(f"Loading model from: {args.model_id}")
    model = load_model_for_training(args.model_id, hf_token, processor)
    
    # Step 3: Create dataset
    logger.info(f"Creating dataset from: {args.processed_dir}")
    dataset = MedicalImageDataset(args.processed_dir, processor, mode="train")
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    # Create collate function
    collate_fn = create_collate_fn(processor, mode="train")
    
    # Step 4: Train the model
    logger.info("Starting training...")
    train_model(
        model=model,
        train_dataset=dataset,
        processor=processor,
        collate_fn=collate_fn,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Step 5: Merge and save model
    if not args.skip_merge:
        logger.info("Merging LoRA weights with base model...")
        
        # Free up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Merge and save
        merge_and_save_lora_model(
            base_model_id=args.model_id,
            adapter_model_dir=args.output_dir,
            output_dir=args.merged_dir
        )
        
        logger.info(f"Merged model saved to: {args.merged_dir}")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()