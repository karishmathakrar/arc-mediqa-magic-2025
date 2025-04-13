#!/usr/bin/env python
"""
Main entry point for the MEDIQA project.
Provides a unified interface to all functionality.
"""
import os
import sys
import argparse
import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_module(module_name, args=None):
    """
    Import and run a module's main function with args.
    
    Args:
        module_name: Name of the module to import
        args: Arguments to pass to the module
        
    Returns:
        Result of the module's main function
    """
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'main'):
            if args:
                # Convert namespace to sys.argv format
                sys.argv = [module_name]
                for key, value in vars(args).items():
                    if key == 'command':
                        continue
                    if isinstance(value, bool):
                        if value:
                            sys.argv.append(f"--{key}")
                    elif value is not None:
                        sys.argv.append(f"--{key}")
                        sys.argv.append(str(value))
                
                # Call the module's main function
                return module.main()
            else:
                return module.main()
        else:
            logger.error(f"Module {module_name} does not have a main function")
            return None
    except ImportError as e:
        logger.error(f"Error importing module {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error running {module_name}: {e}")
        return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MEDIQA Project - Medical Image Question Answering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    preprocess_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="processed_data",
        help="Directory to save processed data"
    )
    preprocess_parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "inference"],
        default="train",
        help="Preprocessing mode (train or inference)"
    )
    preprocess_parser.add_argument(
        "--csv_file", 
        type=str, 
        default=None,
        help="CSV file with data (will be generated if it doesn't exist)"
    )
    preprocess_parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of encounters to process"
    )
    preprocess_parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5,
        help="Batch size for processing"
    )
    preprocess_parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess the data even if already processed"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    train_parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed_data/train",
        help="Directory with processed data for training"
    )
    train_parser.add_argument(
        "--model_id", 
        type=str, 
        default="google/gemma-3-4b-it",
        help="Path to model or HuggingFace model ID"
    )
    train_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/fine_tuned",
        help="Directory to save fine-tuned model"
    )
    train_parser.add_argument(
        "--merged_dir", 
        type=str, 
        default="models/merged",
        help="Directory to save merged model"
    )
    train_parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Per-device batch size for training"
    )
    train_parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    train_parser.add_argument(
        "--epochs", 
        type=int, 
        default=1,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run preprocessing step before training"
    )
    train_parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for processing"
    )
    train_parser.add_argument(
        "--skip_merge", 
        action="store_true",
        help="Skip merging LoRA weights with base model"
    )
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    inference_parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed_data/inference",
        help="Directory with processed data for inference"
    )
    inference_parser.add_argument(
        "--model_id", 
        type=str, 
        default="models/merged",
        help="Path to model or HuggingFace model ID"
    )
    inference_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save inference results"
    )
    inference_parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    inference_parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=64,
        help="Maximum number of new tokens to generate"
    )
    inference_parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.5,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    inference_parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run preprocessing step before inference"
    )
    inference_parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for processing"
    )
    inference_parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run evaluation after inference"
    )
    inference_parser.add_argument(
        "--reference_file", 
        type=str, 
        default="2025_dataset/train/train_cvqa.json",
        help="Path to reference file for evaluation"
    )
    inference_parser.add_argument(
        "--option_maps", 
        type=str, 
        default="2025_dataset/train/option_maps.json",
        help="Path to option maps file for evaluation"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    evaluate_parser.add_argument(
        "--reference_file", 
        type=str, 
        default="2025_dataset/train/train_cvqa.json",
        help="Path to reference data JSON file"
    )
    evaluate_parser.add_argument(
        "--prediction_file", 
        type=str, 
        default="evaluation_results/results.json",
        help="Path to prediction data JSON file"
    )
    evaluate_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    evaluate_parser.add_argument(
        "--option_maps", 
        type=str, 
        default="2025_dataset/train/option_maps.json",
        help="Path to option maps JSON file"
    )
    evaluate_parser.add_argument(
        "--text_predictions", 
        action="store_true",
        help="Set if predictions are text rather than indices"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the MEDIQA project."""
    args = parse_args()
    
    if not args.command:
        logger.error("No command specified. Use one of: preprocess, train, inference, evaluate")
        return
    
    # Map commands to modules
    command_modules = {
        "preprocess": "scripts.preprocess",
        "train": "scripts.train",
        "inference": "scripts.inference",
        "evaluate": "scripts.evaluation"
    }
    
    # Run the specified command
    if args.command in command_modules:
        logger.info(f"Running {args.command} command...")
        run_module(command_modules[args.command], args)
    else:
        logger.error(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()