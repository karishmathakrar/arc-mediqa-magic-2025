#!/usr/bin/env python3
"""
Comprehensive Example Usage of the ReasoningPipeline Wrapper

This script demonstrates how to use the parameterizable ReasoningPipeline wrapper
for medical image analysis with structured reasoning.
"""

import os
import sys


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(base_dir)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Base directory: {base_dir}")
print(f"Python path: {sys.path[-1]}")


def example_basic_usage():
    """Example of basic usage with default configuration."""
    print("=== Example 1: Basic Usage ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Create pipeline with default configuration
        config = ReasoningConfig(
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs"
        )
        
        pipeline = ReasoningPipeline(config)
        
        print(f"✓ Pipeline initialized with base_dir: {base_dir}")
        print(f"✓ Output directory: {config.output_dir}")
        print(f"✓ Using model: {config.gemini_model}")
        print(f"✓ Dataset: {'Test' if config.use_test_dataset else 'Validation'}")
        print(f"✓ Model type: {'Finetuned' if config.use_finetuning else 'Base'}")
        
        print("Pipeline ready for processing!")
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("This is expected if you don't have the required dataset structure or .env file.")


def example_custom_configuration():
    """Example of usage with custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Create custom configuration
        config = ReasoningConfig(
            use_finetuning=True,
            use_test_dataset=False,
            gemini_model="gemini-2.5-flash-preview-04-17",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            # API key will be loaded from .env file
            save_intermediate_results=True,
            intermediate_save_frequency=3
        )
        
        # Create pipeline with custom configuration
        pipeline = ReasoningPipeline(config)
        
        print(f"✓ Custom pipeline configured:")
        print(f"  - Base directory: {config.base_dir}")
        print(f"  - Output directory: {config.output_dir}")
        print(f"  - Intermediate saves every: {config.intermediate_save_frequency} encounters")
        print(f"  - Using validation dataset with base model predictions")
        
    except Exception as e:
        print(f"Error with custom configuration: {e}")


def example_test_dataset_usage():
    """Example of using the pipeline with test dataset."""
    print("\n=== Example 3: Test Dataset Usage ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Configuration for test dataset
        config = ReasoningConfig(
            use_finetuning=True,
            use_test_dataset=True,  # Use test dataset instead of validation
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            save_intermediate_results=True,
            intermediate_save_frequency=5
        )
        
        pipeline = ReasoningPipeline(config)
        
        print(f"✓ Test dataset pipeline configured:")
        print(f"  - Using test dataset: {config.use_test_dataset}")
        print(f"  - Expected dataset path: {config.to_reasoning_args().dataset_path}")
        print(f"  - Expected images directory: {config.to_reasoning_args().images_dir}")
        
    except Exception as e:
        print(f"Error with test dataset configuration: {e}")


def example_finetuned_model_usage():
    """Example of using the pipeline with finetuned model predictions."""
    print("\n=== Example 4: Finetuned Model Usage ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Configuration for finetuned model
        config = ReasoningConfig(
            use_finetuning=True,  # Use finetuned model predictions
            use_test_dataset=False,
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            gemini_model="gemini-2.5-flash-preview-04-17",
            save_intermediate_results=True
        )
        
        pipeline = ReasoningPipeline(config)
        
        print(f"✓ Finetuned model pipeline configured:")
        print(f"  - Using finetuned predictions: {config.use_finetuning}")
        print(f"  - Model predictions directory: {config.to_reasoning_args().model_predictions_dir}")
        print(f"  - Expected prediction prefix: {config.to_reasoning_args().prediction_prefix}")
        
    except Exception as e:
        print(f"Error with finetuned model configuration: {e}")


def example_step_by_step_usage():
    """Example showing step-by-step pipeline usage."""
    print("\n=== Example 5: Step-by-Step Usage ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Create configuration
        config = ReasoningConfig(
            use_finetuning=True,
            use_test_dataset=False,
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs"
        )
        
        # Create pipeline
        pipeline = ReasoningPipeline(config)
        
        print("Step 1: Pipeline created successfully")
        
        # Initialize pipeline (this loads data and sets up services)
        print("Step 2: Initializing pipeline...")
        try:
            pipeline.initialize()
            print("✓ Pipeline initialized successfully")
            print("  - Data loaded and services configured")
            print("  - Ready to process encounters")
        except Exception as e:
            print(f"✗ Pipeline initialization failed: {e}")
            print("  This is expected if model predictions or datasets are not available")
        
    except Exception as e:
        print(f"Error in step-by-step usage: {e}")


def example_single_encounter_processing():
    """Example of processing a single encounter."""
    print("\n=== Example 6: Single Encounter Processing ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Create configuration
        config = ReasoningConfig(
            use_finetuning=True,
            use_test_dataset=False,
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs"
        )
        
        pipeline = ReasoningPipeline(config)
        
        # Example encounter ID (replace with actual encounter ID from your data)
        encounter_id = "ENC00852"
        
        print(f"Attempting to process encounter: {encounter_id}")
        
        try:
            results = pipeline.process_single_encounter(encounter_id)
            print(f"✓ Successfully processed encounter {encounter_id}")
            print(f"  - Results keys: {list(results.keys())}")
            if encounter_id in results:
                questions = results[encounter_id]
                print(f"  - Questions processed: {len(questions)}")
        except Exception as e:
            print(f"✗ Error processing encounter: {e}")
            print("  This is expected if the encounter ID doesn't exist in your data")
        
    except Exception as e:
        print(f"Error in single encounter processing: {e}")


def example_all_encounters_processing():
    """Example of processing all encounters."""
    print("\n=== Example 7: All Encounters Processing ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Create configuration with frequent intermediate saves for demo
        config = ReasoningConfig(
            use_finetuning=True,
            use_test_dataset=False,
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            save_intermediate_results=True,
            intermediate_save_frequency=2  # Save every 2 encounters for demo
        )
        
        pipeline = ReasoningPipeline(config)
        
        print("Attempting to process all encounters...")
        print("Note: This will process all available encounters in your dataset")
        
        try:
            all_results = pipeline.process_all_encounters()
            print(f"✓ Successfully processed {len(all_results)} encounters")
            print("  - Results formatted for evaluation")
            print("  - Intermediate results saved during processing")
        except Exception as e:
            print(f"✗ Error processing all encounters: {e}")
            print("  This is expected if model predictions or datasets are not available")
        
    except Exception as e:
        print(f"Error in all encounters processing: {e}")


def example_configuration_validation():
    """Example of configuration validation."""
    print("\n=== Example 8: Configuration Validation ===")
    
    try:
        from reasoning_pipeline import ReasoningConfig, ReasoningPipeline
        
        # Test valid configuration
        print("Testing valid configuration...")
        try:
            config = ReasoningConfig(
                base_dir=base_dir,
                output_dir=f"{base_dir}/outputs",
                use_finetuning=True,
                use_test_dataset=False
            )
            args = config.to_reasoning_args()
            print(f"✓ Valid config created:")
            print(f"  - Base directory: {args.base_dir}")
            print(f"  - Output directory: {args.output_dir}")
            print(f"  - Dataset path: {args.dataset_path}")
            print(f"  - Images directory: {args.images_dir}")
        except Exception as e:
            print(f"✗ Valid config failed: {e}")
        
        models_to_test = [
            "gemini-2.5-flash-preview-04-17",
        ]
        
        for model in models_to_test:
            try:
                config = ReasoningConfig(
                    base_dir=base_dir,
                    gemini_model=model
                )
                print(f"✓ Model {model} configured successfully")
            except Exception as e:
                print(f"✗ Model {model} configuration failed: {e}")
        
    except Exception as e:
        print(f"Error in configuration validation: {e}")


def main():
    """Run all comprehensive examples."""
    print("ReasoningPipeline Wrapper - Comprehensive Usage Examples")
    print("=" * 60)
    
    # Run all examples
    examples = [
        example_basic_usage,
        example_custom_configuration,
        example_test_dataset_usage,
        example_finetuned_model_usage,
        example_step_by_step_usage,
        example_single_encounter_processing,
        example_all_encounters_processing,
        example_configuration_validation,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nQuick Start Guide:")
    print("1. Ensure you have a .env file with API_KEY set")
    print("2. from reasoning_pipeline import ReasoningConfig, ReasoningPipeline")
    print("3. config = ReasoningConfig(base_dir='/path/to/data')")
    print("4. pipeline = ReasoningPipeline(config)")
    print("5. results = pipeline.process_all_encounters()")
    print("\nFor single encounter:")
    print("6. result = pipeline.process_single_encounter('ENC00852')")
    print("\nFor test dataset:")
    print("7. config = ReasoningConfig(use_test_dataset=True)")
    print("\nFor finetuned model predictions:")
    print("8. config = ReasoningConfig(use_finetuning=True)")


if __name__ == "__main__":
    main()
