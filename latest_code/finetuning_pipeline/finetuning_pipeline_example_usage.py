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
        from finetuning_pipeline import FineTuningPipeline, Config, DataProcessor, ModelManager
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
    
    # Initialize with custom settings
    try:
        from finetuning_pipeline import FineTuningPipeline, Config, DataProcessor, ModelManager
        pipeline = FineTuningPipeline(
            model_name="Qwen2.5-VL-3B-Instruct",  # Different model
            base_dir="../../",         # Custom data directory
            output_dir="../../outputs",    # Custom output directory
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
    
    try:
        from finetuning_pipeline import FineTuningPipeline, Config, DataProcessor, ModelManager
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir="../../",         # Custom data directory
            output_dir="../../outputs",    # Custom output directory
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


# Example 4: Single prediction
def example_single_prediction():
    """Example of making a single prediction."""
    print("\n=== Example 4: Single Prediction ===")
    
    try:
        from finetuning_pipeline import FineTuningPipeline, Config, DataProcessor, ModelManager
        pipeline = FineTuningPipeline(validate_paths=False)
        
        # Example single prediction (would need actual image and model)
        image_path = "/path/to/medical/image.jpg"
        query_text = "What skin condition is visible in this image?"
        
        print(f"Would predict on image: {image_path}")
        print(f"With query: {query_text}")
        
        # prediction = pipeline.predict_single(image_path, query_text)
        # print(f"Prediction: {prediction}")
        
        print("Single prediction example completed! (simulated)")
        
    except Exception as e:
        print(f"Single prediction error: {e}")


# Example 5: Using individual components
def example_individual_components():
    """Example of using individual pipeline components."""
    print("\n=== Example 5: Individual Components ===")
    
    try:
        from finetuning_pipeline import FineTuningPipeline, Config, DataProcessor, ModelManager
        # Create custom configuration
        config = Config(
            model_name="Qwen2-VL-2B-Instruct",
            validate_paths=False
        )
        
        # Use individual components
        data_processor = DataProcessor(config)
        model_manager = ModelManager(config)
        
        print("Individual components initialized successfully!")
        print(f"Model ID: {config.MODEL_ID}")
        print(f"Is Qwen model: {config.IS_QWEN}")
        
    except Exception as e:
        print(f"Individual components error: {e}")


def main():
    """Run all examples."""
    print("Medical Vision Pipeline - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_training_workflow()
    example_single_prediction()
    example_individual_components()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo use in your own project:")
    print("1. from latest_code import FineTuningPipeline")
    print("2. pipeline = FineTuningPipeline()")
    print("3. trainer = pipeline.train()")
    print("4. predictions = pipeline.run_inference()")


if __name__ == "__main__":
    main()
