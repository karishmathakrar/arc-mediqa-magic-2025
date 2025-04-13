#!/usr/bin/env python
"""
One-click pipeline script for the MEDIQA project.
Runs the entire workflow: preprocessing, training, inference, and evaluation.
"""
import os
import sys
import argparse
import logging
import subprocess
import time

import config
from utils import print_system_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete MEDIQA pipeline")
    parser.add_argument(
        "--skip_preprocessing", 
        action="store_true",
        help="Skip data preprocessing"
    )
    parser.add_argument(
        "--skip_training", 
        action="store_true",
        help="Skip model training"
    )
    parser.add_argument(
        "--skip_inference", 
        action="store_true",
        help="Skip model inference"
    )
    parser.add_argument(
        "--skip_evaluation", 
        action="store_true",
        help="Skip result evaluation"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples (for faster testing)"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=config.MODEL_ID,
        help="HuggingFace model ID to use"
    )
    parser.add_argument(
        "--use_validation", 
        action="store_true",
        help="Use validation dataset for inference and evaluation"
    )
    parser.add_argument(
        "--validation_dir", 
        type=str, 
        default="2025_dataset/validation",
        help="Directory containing validation data"
    )
    
    return parser.parse_args()

def run_step(step_name, command, arguments=None):
    """Run a pipeline step as a subprocess."""
    start_time = time.time()
    logger.info(f"Starting {step_name}...")
    
    if arguments is None:
        arguments = []
    
    cmd = [sys.executable, command] + arguments
    logger.info(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.run(cmd)
    
    if process.returncode != 0:
        logger.error(f"{step_name} failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    
    elapsed = time.time() - start_time
    logger.info(f"{step_name} completed successfully in {elapsed:.2f} seconds")

def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    # Print system info
    print_system_info()
    
    # Create necessary directories
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    
    limit_arg = []
    if args.limit:
        limit_arg = [f"--limit={args.limit}"]
    
    # Determine inference and evaluation paths based on validation flag
    if args.use_validation:
        inference_data_dir = args.validation_dir
        processed_dir = "processed_data/validation"
        output_dir = "outputs/validation"
        reference_file = os.path.join(args.validation_dir, "validation_cvqa.json")
    else:
        inference_data_dir = config.DATASET_DIR
        processed_dir = os.path.join(config.PROCESSED_DIR, "inference")
        output_dir = config.OUTPUTS_DIR
        reference_file = os.path.join(config.DATASET_DIR, "train_cvqa.json")
    
    # 1. Preprocess data
    if not args.skip_preprocessing:
        # Always preprocess training data
        run_step(
            "Training data preprocessing",
            "preprocess.py",
            ["--mode=train", "--reprocess"] + limit_arg
        )
        
        # Preprocess inference data (training or validation based on flag)
        if args.use_validation:
            run_step(
                "Validation data preprocessing",
                "preprocess.py",
                ["--mode=inference", "--reprocess", f"--data_dir={inference_data_dir}", f"--output_dir={processed_dir}"] + limit_arg
            )
        else:
            run_step(
                "Inference data preprocessing",
                "preprocess.py",
                ["--mode=inference", "--reprocess"] + limit_arg
            )
    else:
        logger.info("Skipping preprocessing step")
    
    # 2. Train model
    if not args.skip_training:
        run_step(
            "Model training",
            "train.py",
            [f"--model_id={args.model_id}"]
        )
    else:
        logger.info("Skipping training step")
    
    # 3. Run inference
    if not args.skip_inference:
        run_step(
            "Model inference",
            "inference.py",
            [f"--base_model_id={args.model_id}", f"--processed_dir={processed_dir}", f"--output_dir={output_dir}"] + limit_arg
        )
    else:
        logger.info("Skipping inference step")
    
    # 4. Evaluate results
    if not args.skip_evaluation:
        base_prediction_file = os.path.join(output_dir, "base_model", "results.json")
        finetuned_prediction_file = os.path.join(output_dir, "finetuned_model", "results.json")
        
        run_step(
            "Result evaluation",
            "evaluate.py",
            [f"--reference_file={reference_file}", f"--base_prediction_file={base_prediction_file}", f"--finetuned_prediction_file={finetuned_prediction_file}"]
        )
    else:
        logger.info("Skipping evaluation step")
    
    # Print summary of output locations
    logger.info("\nOutput files:")
    logger.info(f"- Processed data: {config.PROCESSED_DIR}")
    logger.info(f"- Fine-tuned model: {config.FINETUNED_MODEL_DIR}")
    logger.info(f"- Merged model: {config.MERGED_MODEL_DIR}")
    logger.info(f"- Inference results: {config.OUTPUTS_DIR}")
    logger.info(f"- Evaluation results: {os.path.join(config.OUTPUTS_DIR, 'comparison')}")

if __name__ == "__main__":
    main()