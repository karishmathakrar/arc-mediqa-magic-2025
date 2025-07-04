#!/usr/bin/env python3
"""
Example usage of the FineTuningPipeline

This script demonstrates how to use the refactored pipeline components
that can now be easily imported and used in other projects.
"""
# Example 1: Basic usage with default settings
def example_basic_usage():
    """Example of basic pipeline usage with default settings."""
    print("=== Example 1: Basic Usage ===")
    
    # Initialize pipeline with default settings
    # This will use the current directory and validate paths
    try:
        import os
        from finetuning_pipeline.finetuning_pipeline import FineTuningPipeline, Config, ModelManager
        pipeline = FineTuningPipeline()
        
        # Print available models
        print("Available models:", pipeline.get_available_models())
        
        # Print system info
        pipeline.print_system_info()
        
        print("Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("This is expected if you don't have the required dataset structure.")


# Example 2: Custom configuration
def example_custom_config():
    """Example of pipeline usage with custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")
    import os
    from finetuning_pipeline.finetuning_pipeline import FineTuningPipeline, Config, ModelManager

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Initialize with custom settings
    try:
        pipeline = FineTuningPipeline(
            model_name="Qwen2.5-VL-3B-Instruct",  # Different model
            base_dir=base_dir,         # Custom data directory
            output_dir=f"{base_dir}/outputs",    # Custom output directory
            validate_paths=False,                  # Skip path validation for demo
            setup_environment=False                # Skip environment setup
        )
        
        print(f"Pipeline configured with model: {pipeline.get_config().SELECTED_MODEL}")
        print(f"Base directory: {pipeline.get_config().BASE_DIR}")
        print(f"Output directory: {pipeline.get_config().OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error with custom config: {e}")


# Example 3: Training workflow
def example_training_workflow():
    """Example of complete training workflow."""
    print("\n=== Example 3: Training Workflow ===")
    from finetuning_pipeline.finetuning_pipeline import FineTuningPipeline, Config, ModelManager
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,         # Custom data directory
            output_dir=f"{base_dir}/outputs",    # Custom output directory
            validate_paths=True,
        )
        
        # Step 1: Prepare data (would normally process your dataset)
        print("Step 1: Preparing data...")
        train_df, val_df = pipeline.prepare_data(test_mode=True, min_data_size=5)
        
        # Step 2: Train model (would normally train on your data)
        print("Step 2: Training model...")
        trainer = pipeline.train(test_mode=True)
        
        # Step 3: Run inference (would normally run on validation data)
        print("Step 3: Running inference...")
        predictions = pipeline.run_inference(max_samples=10)
        
        print("Training workflow completed! (simulated)")
        
    except Exception as e:
        print(f"Training workflow error: {e}")


def main():
    """Run all examples."""
    print("Medical Vision Pipeline - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_training_workflow()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo use in your own project:")
    print("1. from latest_code import FineTuningPipeline")
    print("2. pipeline = FineTuningPipeline()")
    print("3. trainer = pipeline.train()")
    print("4. predictions = pipeline.run_inference()")


if __name__ == "__main__":
    main()
