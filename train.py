#!/usr/bin/env python
"""
Main training script for the MEDIQA project.
"""
import os
import sys
import argparse
import logging
import torch
from tqdm.auto import tqdm

# Add the current directory to the path so Python can find the local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import *  # Import all config variables
import utils
from data.processor import MedicalDataProcessor
from data.loader import MedicalImageDataset, create_dataloader, create_collate_fn
from models.model_utils import setup_model_and_processor
from models.trainer import train_model, merge_and_save_lora_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune model on medical images.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=TRAIN_DATA_DIR,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default=None,
        help="CSV file with training data (alternative to processing encounters)"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed_data_train",
        help="Directory to save/load processed data"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=MODEL_ID,
        help="Hugging Face model ID or path to local model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="fine_tuned_model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--merged_dir", 
        type=str, 
        default="merged_model",
        help="Directory to save the merged model"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of encounters to process"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE,
        help="Per-device batch size for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=NUM_TRAIN_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess the data even if already processed"
    )
    parser.add_argument(
        "--skip_merge", 
        action="store_true",
        help="Skip merging LoRA weights with base model"
    )
    
    return parser.parse_args()

def process_data(data_dir, processed_dir, csv_file=None, limit=None, reprocess=False):
    """
    Process the data for training.
    
    Args:
        data_dir: Directory containing the dataset
        processed_dir: Directory to save processed data
        csv_file: CSV file with training data (alternative to processing encounters)
        limit: Limit the number of encounters/rows to process
        reprocess: Whether to reprocess the data even if already processed
        
    Returns:
        Path to processed data directory
    """
    if os.path.exists(processed_dir) and not reprocess:
        logger.info(f"Using existing processed data from {processed_dir}")
        return processed_dir
    
    if os.path.exists(processed_dir) and reprocess:
        logger.info(f"Removing existing processed data from {processed_dir}")
        import shutil
        shutil.rmtree(processed_dir)
    
    utils.ensure_directory(processed_dir)
    
    processor = MedicalDataProcessor(base_dir=data_dir)
    
    if csv_file:
        logger.info(f"Processing data from CSV file {csv_file} to {processed_dir}")
        total = processor.process_from_csv(csv_file, batch_size=5, limit=limit, save_dir=processed_dir)
    else:
        logger.info(f"Processing data from {data_dir} to {processed_dir}")
        total = processor.process_dataset(batch_size=5, limit=limit, save_dir=processed_dir)
    
    logger.info(f"Processed {total} examples")
    
    return processed_dir

def main():
    """Main entry point."""
    args = parse_args()
    
    # Update config variables with command line arguments
    global BATCH_SIZE, NUM_TRAIN_EPOCHS, LEARNING_RATE
    BATCH_SIZE = args.batch_size
    NUM_TRAIN_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    
    utils.print_system_info()
    utils.clear_gpu_memory()
    
    # Process data
    processed_dir = process_data(
        args.data_dir, 
        args.processed_dir, 
        csv_file=args.csv_file,
        limit=args.limit, 
        reprocess=args.reprocess
    )
    
    # Load model and processor for training
    model, processor = setup_model_and_processor(args.model_id, for_training=True)
    
    # Load dataset
    dataset = MedicalImageDataset(processed_dir, processor, mode="train")
    logger.info(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    # Create collate function
    collate_fn = create_collate_fn(processor, mode="train")
    
    # Train model
    output_dir = train_model(
        model, 
        dataset, 
        processor, 
        collate_fn=collate_fn, 
        output_dir=args.output_dir
    )
    
    # Free up memory
    del model
    utils.clear_gpu_memory()
    
    # Merge and save the model
    if not args.skip_merge:
        logger.info("Merging LoRA weights with base model...")
        merged_dir = merge_and_save_lora_model(
            args.model_id, 
            args.output_dir, 
            output_dir=args.merged_dir
        )
        logger.info(f"Merged model saved to {merged_dir}")
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()