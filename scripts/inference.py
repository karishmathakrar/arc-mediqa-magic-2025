#!/usr/bin/env python
"""
Refactored inference script for the MEDIQA project.
Uses modular code structure and reduces redundancy.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import torch
import json
import gc
from tqdm.auto import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Add parent directory to path to allow absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modules
from config.config import *
from data.loader import MedicalImageDataset, create_collate_fn
from models.model_utils import load_model_for_inference, load_processor
import utils.utils as utils

# For preprocessing integration
from scripts.preprocess import preprocess_dataset
# For evaluation integration
from scripts.evaluation import evaluate_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_generated_answer(text):
    """Clean up the generated answer text."""
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
        csv_file = os.path.join(args.data_dir, "dataset_inference.csv")
        
        # Use the preprocess_dataset function from preprocess.py
        _, args.processed_dir = preprocess_dataset(
            csv_file=csv_file,
            data_dir=args.data_dir,
            output_dir=args.processed_dir,
            mode="inference",
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
    model = load_model_for_inference(args.model_id, hf_token, processor)
    
    # Step 3: Create dataset and run inference
    logger.info(f"Creating dataset from: {args.processed_dir}")
    dataset = MedicalImageDataset(args.processed_dir, processor, mode="inference")
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    logger.info(f"Running inference on {len(dataset)} examples...")
    results = run_inference(
        model=model,
        processor=processor,
        dataset=dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Step 4: Save results
    logger.info("Saving inference results...")
    _, json_path = save_results(results, args.output_dir)
    
    # Step 5: Run evaluation (if requested)
    if args.evaluate:
        logger.info("Running evaluation...")
        eval_results = evaluate_predictions(
            reference_file=args.reference_file,
            prediction_file=json_path,
            output_dir=args.output_dir,
            option_maps_file=args.option_maps,
            text_predictions=True
        )
        
        # Print evaluation results
        logger.info("Evaluation results:")
        for metric, value in eval_results.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            elif metric != "evaluated_encounter_ids":  # Skip printing long lists
                logger.info(f"  {metric}: {value}")
        
        logger.info(f"Overall accuracy: {eval_results.get('accuracy_all', 0.0):.4f}")
    
    # Clean up
    del model
    utils.clear_gpu_memory()
    logger.info("Inference completed successfully")

if __name__ == "__main__":
    main()

def run_inference(model, processor, dataset, batch_size=1, max_new_tokens=64, temperature=0.5):
    """
    Run inference on the dataset.
    
    Args:
        model: Model to use for inference
        processor: Processor for tokenization
        dataset: Dataset containing examples
        batch_size: Batch size for inference
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        
    Returns:
        List of dictionaries containing inference results
    """
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=create_collate_fn(processor, "inference")
    )
    
    results = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference with no gradient calculation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move tensor inputs to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)
            
            # Define stopping tokens
            stop_token_ids = [
                processor.tokenizer.eos_token_id, 
                processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]
            
            # Generate text
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-5),  # Ensure temperature is positive
                num_beams=1,  # Simple greedy or sampling
                eos_token_id=stop_token_ids
            )
            
            # Decode generated text
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], outputs)]
            generated_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Store results with metadata
            for i, text in enumerate(generated_texts):
                metadata = batch["metadata"][i]
                
                results.append({
                    "encounter_id": metadata["encounter_id"],
                    "qid": metadata["qid"],
                    "generated_answer": text,
                    "ground_truth": metadata.get("ground_truth", ""),
                    "cleaned_answer": clean_generated_answer(text)
                })
    
    return results

def save_results(results, output_dir):
    """
    Save inference results to CSV and JSON files.
    
    Args:
        results: List of dictionaries containing inference results
        output_dir: Directory to save results
        
    Returns:
        Tuple of (results_df, json_path, indexed_json_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save CSV results
    csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"CSV results saved to '{csv_path}'")
    
    # Convert to JSON format for evaluation
    json_path = os.path.join(output_dir, "results.json")
    
    # Group by encounter_id
    encounter_groups = results_df.groupby('encounter_id')
    output_data = []
    
    for encounter_id, group in encounter_groups:
        # Create entry for this encounter
        encounter_entry = {"encounter_id": encounter_id}
        
        # Add each question's answer
        for _, row in group.iterrows():
            qid = row['qid']
            answer = row['cleaned_answer']
            encounter_entry[qid] = answer
        
        output_data.append(encounter_entry)
    
    # Save as JSON
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"JSON results saved to '{json_path}'")
    
    return results_df, json_path
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on medical images")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed_data/inference",
        help="Directory with processed data for inference"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="models/merged",
        help="Path to model or HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=64,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.5,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run preprocessing step before inference"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for processing"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run evaluation after inference"
    )
    parser.add_argument(
        "--reference_file", 
        type=str, 
        default="2025_dataset/train/train_cvqa.json",
        help="Path to reference file for evaluation"
    )
    parser.add_argument(
        "--option_maps", 
        type=str, 
        default="2025_dataset/train/option_maps.json",
        help="Path to option maps file for evaluation"
    )
    
    return parser.parse_args()