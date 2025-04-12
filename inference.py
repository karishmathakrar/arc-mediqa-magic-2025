#!/usr/bin/env python
"""
Main inference script for the MEDIQA project.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import torch
import json
from tqdm.auto import tqdm

# Add the current directory to the path so Python can find the local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import *  # Import all config variables
import utils
from data.processor import MedicalDataProcessor
from data.loader import MedicalImageDataset, create_dataloader
from models.model_utils import setup_model_and_processor
from convert_results import convert_csv_to_json

from postprocess_utils import convert_csv_to_indexed_json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on medical images.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=TRAIN_DATA_DIR,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed_data_inference",
        help="Directory to save/load processed data"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=MODEL_ID,
        help="Hugging Face model ID or path to local model"
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
        default=INFERENCE_BATCH_SIZE,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="inference_results.csv",
        help="Path to save inference results CSV"
    )
    parser.add_argument(
        "--json_output", 
        type=str, 
        default="data_cvqa_sys.json",
        help="Path to save inference results in JSON format for evaluation"
    )
    parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess the data even if already processed"
    )
    
    parser.add_argument(
        "--indexed_json_output", 
        type=str, 
        default="data_cvqa_sys_indices.json",
        help="Path to save indexed JSON for evaluation"
    )
    
    return parser.parse_args()

def process_data(data_dir, processed_dir, limit=None, reprocess=False):
    """
    Process the data for inference.
    
    Args:
        data_dir: Directory containing the dataset
        processed_dir: Directory to save processed data
        limit: Limit the number of encounters to process
        reprocess: Whether to reprocess the data even if already processed
        
    Returns:
        Path to processed data directory
    """
    if os.path.exists(processed_dir) and not reprocess:
        logger.info(f"Using existing processed data from {processed_dir}")
        return processed_dir
    
    logger.info(f"Processing data from {data_dir} to {processed_dir}")
    
    if os.path.exists(processed_dir) and reprocess:
        logger.info(f"Removing existing processed data from {processed_dir}")
        import shutil
        shutil.rmtree(processed_dir)
    
    utils.ensure_directory(processed_dir)
    
    processor = MedicalDataProcessor(base_dir=data_dir)
    total = processor.process_dataset(batch_size=5, limit=limit, save_dir=processed_dir)
    
    logger.info(f"Processed {total} examples")
    
    return processed_dir

def run_inference(model, processor, dataset, batch_size=1):
    """
    Run inference on the dataset.
    
    Args:
        model: Model to use for inference
        processor: Processor for tokenization
        dataset: Dataset to run inference on
        batch_size: Batch size for inference
        
    Returns:
        List of result dictionaries
    """
    dataloader = create_dataloader(
        dataset, 
        processor, 
        batch_size=batch_size, 
        shuffle=False, 
        mode="inference"
    )
    
    results = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference with no gradient calculation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move tensor inputs to device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items() if k != "metadata"}
            
            # Generate text
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Deterministic generation
                num_beams=1,      # Simple greedy decoding
                temperature=0,
            )
            
            # Decode generated text
            generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Store results with metadata
            for i, text in enumerate(generated_texts):
                metadata = batch["metadata"][i]
                results.append({
                    "encounter_id": metadata["encounter_id"],
                    "qid": metadata["qid"],
                    "generated_answer": text,
                    "ground_truth": metadata["ground_truth"]
                })
                
    return results

def save_results(results, output_file, json_output_file=None, indexed_json_output=None):
    """
    Save inference results to files.
    
    Args:
        results: List of result dictionaries
        output_file: Path to save CSV results
        json_output_file: Path to save JSON results for evaluation
        indexed_json_output: Path to save indexed JSON for evaluation
        
    Returns:
        DataFrame with results
    """
    results_df = pd.DataFrame(results)
    
    logger.info("\nInference Results:")
    logger.info(f"Total examples processed: {len(results_df)}")
    
    # Save CSV results
    results_df.to_csv(output_file, index=False)
    logger.info(f"CSV results saved to '{output_file}'")
    
    # Convert and save JSON results for evaluation
    if json_output_file:
        # Use the conversion utility to create the expected JSON format
        convert_csv_to_json(output_file, json_output_file)
        logger.info(f"JSON results for evaluation saved to '{json_output_file}'")
    
    # Convert and save indexed JSON results for evaluation
    if indexed_json_output:
        # Convert directly to indexed format
        convert_csv_to_indexed_json(output_file, indexed_json_output)
        logger.info(f"Indexed JSON results for evaluation saved to '{indexed_json_output}'")
    
    return results_df

def main():
    """Main entry point."""
    args = parse_args()
    
    utils.print_system_info()
    utils.clear_gpu_memory()
    
    # Process data
    processed_dir = process_data(
        args.data_dir, 
        args.processed_dir, 
        limit=args.limit, 
        reprocess=args.reprocess
    )
    
    # Load model and processor
    model, processor = setup_model_and_processor(args.model_id)
    
    # Load dataset
    dataset = MedicalImageDataset(processed_dir, processor, mode="inference")
    logger.info(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    # Run inference
    results = run_inference(model, processor, dataset, batch_size=args.batch_size)
    
    # Save results
    results_df = save_results(
        results, 
        args.output_file, 
        json_output_file=args.json_output,
        indexed_json_output=args.indexed_json_output
    )
    
    logger.info("Inference completed successfully.")

if __name__ == "__main__":
    main()