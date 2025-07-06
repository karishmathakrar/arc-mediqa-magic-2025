#!/usr/bin/env python3
"""
Generate Validation Dataset Script

This script uses the data_preprocessor.py to generate the validation dataset
that can be used for inference with Gemini models.
"""

import os
import sys

# Add latest_code to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'latest_code'))

from data_preprocessor import DataPreprocessor
from finetuning_pipeline.pipeline import Config


def main():
    """Generate the validation dataset."""
    print("Generating validation dataset...")
    
    # Initialize configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config(
        model_name="Qwen2-VL-2B-Instruct",  # Model doesn't matter for data preprocessing
        base_dir=base_dir,
        output_dir=os.path.join(base_dir, "outputs"),
        setup_environment=False,  # Don't need HF setup for data preprocessing
        validate_paths=True
    )
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Generate validation dataset
    print("Preparing validation dataset...")
    val_df = preprocessor.prepare_dataset(mode="val")
    
    print(f"Validation dataset created with {len(val_df)} entries")
    print(f"Dataset saved to: {os.path.join(config.OUTPUT_DIR, 'val_dataset.csv')}")
    
    # Process validation dataset into batch files for inference
    print("Processing validation dataset into batch files...")
    total_processed = preprocessor.preprocess_dataset(
        val_df, 
        batch_size=100, 
        mode="val"
    )
    
    print(f"Processed {total_processed} validation examples into batch files")
    print(f"Batch files saved to: {config.PROCESSED_VAL_DATA_DIR}")
    
    # Inspect a few samples
    print("\nInspecting processed validation data samples:")
    preprocessor.inspect_processed_data(
        processed_dir=config.PROCESSED_VAL_DATA_DIR,
        num_samples=2,
        data_type="val"
    )
    
    print("\nValidation dataset generation complete!")
    return val_df, config


if __name__ == "__main__":
    val_df, config = main()
