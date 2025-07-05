#!/usr/bin/env python3
"""
Example usage of the ReasoningPipeline wrapper class.
This demonstrates how to use the parameterizable interface for medical image analysis.
"""

import os
from reasoning_pipeline import ReasoningConfig, ReasoningPipeline


def example_basic_usage():
    """Example of basic usage with default configuration."""
    print("=== Basic Usage Example ===")
    
    # Create pipeline with default configuration
    pipeline = ReasoningPipeline()
    
    # Process a single encounter
    encounter_id = "ENC00852"  # Replace with actual encounter ID
    try:
        results = pipeline.process_single_encounter(encounter_id)
        print(f"Successfully processed encounter {encounter_id}")
        print(f"Results keys: {list(results.keys())}")
    except Exception as e:
        print(f"Error processing encounter: {e}")


def example_custom_configuration():
    """Example of usage with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = ReasoningConfig(
        use_finetuning=False,
        use_test_dataset=False,
        gemini_model="gemini-2.5-flash-preview-04-17",
        base_dir="/path/to/your/project",  # Customize this
        api_key="your-api-key-here",  # Or set in .env file
        save_intermediate_results=True,
        intermediate_save_frequency=3
    )
    
    # Create pipeline with custom configuration
    pipeline = ReasoningPipeline(config)
    
    # Process all encounters
    try:
        all_results = pipeline.process_all_encounters()
        print(f"Successfully processed {len(all_results)} encounters")
    except Exception as e:
        print(f"Error processing encounters: {e}")


def example_test_dataset_usage():
    """Example of using the pipeline with test dataset."""
    print("\n=== Test Dataset Usage Example ===")
    
    # Configuration for test dataset
    config = ReasoningConfig(
        use_finetuning=False,
        use_test_dataset=True,  # Use test dataset instead of validation
        save_intermediate_results=True,
        intermediate_save_frequency=5
    )
    
    pipeline = ReasoningPipeline(config)
    
    try:
        # Process all test encounters
        test_results = pipeline.process_all_encounters()
        print(f"Successfully processed {len(test_results)} test encounters")
    except Exception as e:
        print(f"Error processing test encounters: {e}")


def example_finetuned_model_usage():
    """Example of using the pipeline with finetuned model predictions."""
    print("\n=== Finetuned Model Usage Example ===")
    
    # Configuration for finetuned model
    config = ReasoningConfig(
        use_finetuning=True,  # Use finetuned model predictions
        use_test_dataset=False,
        gemini_model="gemini-2.5-flash-preview-04-17",
        save_intermediate_results=True
    )
    
    pipeline = ReasoningPipeline(config)
    
    try:
        # Process with finetuned model predictions
        results = pipeline.process_all_encounters()
        print(f"Successfully processed {len(results)} encounters with finetuned model")
    except Exception as e:
        print(f"Error processing with finetuned model: {e}")


def example_step_by_step_usage():
    """Example showing step-by-step pipeline usage."""
    print("\n=== Step-by-Step Usage Example ===")
    
    # Create configuration
    config = ReasoningConfig(
        use_finetuning=False,
        use_test_dataset=False
    )
    
    # Create pipeline
    pipeline = ReasoningPipeline(config)
    
    # Initialize pipeline (this loads data and sets up services)
    try:
        print("Initializing pipeline...")
        pipeline.initialize()
        print("Pipeline initialized successfully")
        
        # Now you can process encounters
        encounter_id = "ENC00852"  # Replace with actual encounter ID
        print(f"Processing encounter {encounter_id}...")
        results = pipeline.process_single_encounter(encounter_id)
        print(f"Encounter processed successfully")
        
    except Exception as e:
        print(f"Error in step-by-step usage: {e}")


if __name__ == "__main__":
    print("ReasoningPipeline Wrapper Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_configuration()
    example_test_dataset_usage()
    example_finetuned_model_usage()
    example_step_by_step_usage()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use in your own code:")
    print("1. Import: from reasoning_pipeline import ReasoningConfig, ReasoningPipeline")
    print("2. Create config: config = ReasoningConfig(...)")
    print("3. Create pipeline: pipeline = ReasoningPipeline(config)")
    print("4. Process data: results = pipeline.process_all_encounters()")
