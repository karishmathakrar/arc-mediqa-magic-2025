#!/usr/bin/env python3
"""
Comprehensive Example Usage of the FineTuningPipeline

This script demonstrates how to use all the refactored pipeline components
that can now be easily imported and used in other projects.
"""

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add latest_code/ to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

# Example 1: Basic usage with default settings
def example_basic_usage():
    """Example of basic pipeline usage with default settings."""
    print("=== Example 1: Basic Usage ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline, Config, ModelManager
        
        # Initialize pipeline with default settings
        pipeline = FineTuningPipeline(
            base_dir=base_dir,
            validate_paths=True,
            setup_environment=True
        )
        
        # Print available models
        print("Available models:", pipeline.get_available_models())
        
        # Print system info
        pipeline.print_system_info()
        
        # Get configuration details
        config = pipeline.get_config()
        print(f"Selected model: {config.SELECTED_MODEL}")
        print(f"Model ID: {config.MODEL_ID}")
        print(f"Base directory: {config.BASE_DIR}")
        print(f"Output directory: {config.OUTPUT_DIR}")
        
        print("Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("This is expected if you don't have the required dataset structure.")


# Example 2: Custom configuration with different models
def example_custom_config():
    """Example of pipeline usage with custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")
    
    from finetuning_pipeline.pipeline import FineTuningPipeline, Config, ModelManager

    # Test different models
    models_to_test = [
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2-VL-2B-Instruct", 
        "llama-3.2-11b-vision"
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            pipeline = FineTuningPipeline(
                model_name=model_name,
                base_dir=base_dir,
                output_dir=f"{base_dir}/outputs",
                validate_paths=False,  # Skip path validation for demo
                setup_environment=False  # Skip environment setup
            )
            
            config = pipeline.get_config()
            print(f"  ✓ Model configured: {config.SELECTED_MODEL}")
            print(f"  ✓ Model ID: {config.MODEL_ID}")
            print(f"  ✓ Is Llama: {config.IS_LLAMA}")
            print(f"  ✓ Is Qwen: {config.IS_QWEN}")
            
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")


# Example 3: Data preprocessing and inspection
def example_data_preprocessing():
    """Example of data preprocessing and inspection functionality."""
    print("\n=== Example 3: Data Preprocessing & Inspection ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=True
        )
        
        # Step 1: Prepare data in test mode
        print("Step 1: Preparing data...")
        train_df, val_df = pipeline.prepare_data(
            use_combined=False, 
            test_mode=True, 
            min_data_size=5
        )
        print(f"  ✓ Training data: {len(train_df) if train_df is not None else 0} samples")
        print(f"  ✓ Validation data: {len(val_df) if val_df is not None else 0} samples")
        
        # Step 2: Inspect processed data
        print("\nStep 2: Inspecting processed data...")
        pipeline.training_pipeline.inspect_processed_data(
            num_samples=2, 
            data_type="train"
        )
        
        # Step 3: Check processed sample
        print("\nStep 3: Checking processed sample...")
        success = pipeline.training_pipeline.data_preprocessor.check_processed_sample(
            use_combined_dataset=False
        )
        print(f"  ✓ Sample check: {'Success' if success else 'Failed'}")
        
        print("Data preprocessing examples completed!")
        
    except Exception as e:
        print(f"Data preprocessing error: {e}")


# Example 4: Test dataset processing
def example_test_dataset_processing():
    """Example of test dataset processing functionality."""
    print("\n=== Example 4: Test Dataset Processing ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=True
        )
        
        # Process test dataset
        print("Processing test dataset...")
        total_test = pipeline.training_pipeline.process_test_dataset(batch_size=50)
        print(f"  ✓ Processed {total_test} test samples")
        
        # Inspect test data
        print("\nInspecting test data...")
        pipeline.training_pipeline.inspect_processed_data(
            num_samples=2, 
            data_type="test"
        )
        
        print("Test dataset processing completed!")
        
    except Exception as e:
        print(f"Test dataset processing error: {e}")


# Example 5: Token analysis
def example_token_analysis():
    """Example of token analysis functionality."""
    print("\n=== Example 5: Token Analysis ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=True
        )
        
        # Prepare small dataset for token analysis
        print("Preparing data for token analysis...")
        train_df, val_df = pipeline.prepare_data(
            test_mode=True, 
            min_data_size=3
        )
        
        # Analyze tokens
        print("Analyzing token usage...")
        token_stats = pipeline.training_pipeline.analyze_dataset_tokens(
            num_samples=5,
            data_type="train"
        )
        
        if token_stats:
            print(f"  ✓ Analyzed {token_stats['summary']['total_samples']} samples")
            print(f"  ✓ Average tokens: {token_stats['summary']['avg_tokens_per_sample']:.1f}")
            print(f"  ✓ Max tokens: {token_stats['summary']['max_tokens']}")
        
        print("Token analysis completed!")
        
    except Exception as e:
        print(f"Token analysis error: {e}")


# Example 6: Training workflow
def example_training_workflow():
    """Example of complete training workflow."""
    print("\n=== Example 6: Training Workflow ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=True
        )
        
        # Step 1: Prepare data
        print("Step 1: Preparing training data...")
        train_df, val_df = pipeline.prepare_data(
            use_combined=False,
            test_mode=True, 
            min_data_size=3
        )
        
        # Step 2: Train model (in test mode)
        print("Step 2: Training model (test mode)...")
        trainer = pipeline.train(
            use_combined=False,
            test_mode=True
        )
        
        if trainer is not None:
            print("  ✓ Training completed successfully!")
        else:
            print("  ✗ Training failed!")
        
        print("Training workflow completed!")
        
    except Exception as e:
        print(f"Training workflow error: {e}")


# Example 7: Inference workflow
def example_inference_workflow():
    """Example of inference workflow."""
    print("\n=== Example 7: Inference Workflow ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=True
        )
        
        # Test base model inference (no fine-tuning)
        print("Testing base model inference...")
        try:
            predictions_df, aggregated_df, formatted_predictions = pipeline.run_inference(
                use_finetuning=False,  # Use base model
                test_mode=False,       # Use validation data
                max_samples=3          # Limit samples for testing
            )
            print(f"  ✓ Base model inference: {len(predictions_df)} predictions")
        except Exception as e:
            print(f"  ✗ Base model inference failed: {e}")
        
        # Test fine-tuned model inference (if available)
        print("\nTesting fine-tuned model inference...")
        try:
            predictions_df, aggregated_df, formatted_predictions = pipeline.run_inference(
                use_finetuning=True,   # Use fine-tuned model
                test_mode=False,       # Use validation data
                max_samples=3          # Limit samples for testing
            )
            print(f"  ✓ Fine-tuned model inference: {len(predictions_df)} predictions")
        except Exception as e:
            print(f"  ✗ Fine-tuned model inference failed: {e}")
            print("    (This is expected if no fine-tuned model exists)")
        
        # Test inference on test data
        print("\nTesting inference on test data...")
        try:
            predictions_df, aggregated_df, formatted_predictions = pipeline.run_inference(
                use_finetuning=False,  # Use base model
                test_mode=True,        # Use test data
                max_samples=3          # Limit samples for testing
            )
            print(f"  ✓ Test data inference: {len(predictions_df)} predictions")
        except Exception as e:
            print(f"  ✗ Test data inference failed: {e}")
        
        print("Inference workflow completed!")
        
    except Exception as e:
        print(f"Inference workflow error: {e}")


# Example 8: Single prediction
def example_single_prediction():
    """Example of single image prediction."""
    print("\n=== Example 8: Single Prediction ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline
        import glob
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=True
        )
        
        # Find a sample image
        image_patterns = [
            f"{base_dir}/2025_dataset/train/images_train/*.jpg",
            f"{base_dir}/2025_dataset/valid/images_valid/*.jpg"
        ]
        
        sample_image = None
        for pattern in image_patterns:
            images = glob.glob(pattern)
            if images:
                sample_image = images[0]
                break
        
        if sample_image:
            print(f"Testing single prediction with image: {os.path.basename(sample_image)}")
            
            # Create a sample query
            query_text = """MAIN QUESTION TO ANSWER: What do you see in this medical image?
Question Metadata: Type: Open, Category: General
Available Options (choose from these): Normal, Abnormal, Not mentioned"""
            
            try:
                prediction = pipeline.predict_single(
                    image_path=sample_image,
                    query_text=query_text,
                    max_new_tokens=50
                )
                print(f"  ✓ Prediction: {prediction}")
            except Exception as e:
                print(f"  ✗ Single prediction failed: {e}")
                print("    (This is expected if no fine-tuned model exists)")
        else:
            print("  ✗ No sample images found for testing")
        
        print("Single prediction example completed!")
        
    except Exception as e:
        print(f"Single prediction error: {e}")


# Example 9: Model management
def example_model_management():
    """Example of model management functionality."""
    print("\n=== Example 9: Model Management ===")
    
    try:
        from finetuning_pipeline.pipeline import FineTuningPipeline, ModelManager
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir=base_dir,
            output_dir=f"{base_dir}/outputs",
            validate_paths=False  # Skip validation for demo
        )
        
        # Test model manager
        print("Testing model manager...")
        model_manager = ModelManager(pipeline.get_config())
        
        # Get model configuration
        model_config = model_manager.get_model_config()
        print(f"  ✓ Model config created: {len(model_config)} parameters")
        
        # Test model loading (without actually loading to save memory)
        print("  ✓ Model manager initialized successfully")
        
        print("Model management example completed!")
        
    except Exception as e:
        print(f"Model management error: {e}")


# Example 10: Configuration validation
def example_configuration_validation():
    """Example of configuration validation."""
    print("\n=== Example 10: Configuration Validation ===")
    
    try:
        from latest_code.finetuning_pipeline.pipeline import Config
        
        # Test valid configuration
        print("Testing valid configuration...")
        try:
            config = Config(
                model_name="Qwen2-VL-2B-Instruct",
                base_dir=base_dir,
                validate_paths=True
            )
            print(f"  ✓ Valid config created for model: {config.SELECTED_MODEL}")
        except Exception as e:
            print(f"  ✗ Valid config failed: {e}")
        
        # Test invalid model name
        print("\nTesting invalid model name...")
        try:
            config = Config(
                model_name="invalid-model-name",
                base_dir=base_dir,
                validate_paths=False
            )
            print("  ✗ Invalid model should have failed!")
        except ValueError as e:
            print(f"  ✓ Invalid model correctly rejected: {e}")
        
        # Test missing paths
        print("\nTesting missing paths...")
        try:
            config = Config(
                model_name="Qwen2-VL-2B-Instruct",
                base_dir="/nonexistent/path",
                validate_paths=True
            )
            print("  ✗ Missing paths should have failed!")
        except FileNotFoundError as e:
            print(f"  ✓ Missing paths correctly detected")
        
        print("Configuration validation completed!")
        
    except Exception as e:
        print(f"Configuration validation error: {e}")


def main():
    """Run all comprehensive examples."""
    print("Medical Vision Pipeline - Comprehensive Usage Examples")
    print("=" * 60)
    
    # Run all examples
    examples = [
        example_basic_usage,
        example_custom_config,
        example_data_preprocessing,
        example_test_dataset_processing,
        example_token_analysis,
        example_training_workflow,
        example_inference_workflow,
        example_single_prediction,
        example_model_management,
        example_configuration_validation
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
    print("1. from latest_code.finetuning_pipeline import FineTuningPipeline")
    print("2. pipeline = FineTuningPipeline(base_dir='/path/to/data')")
    print("3. train_df, val_df = pipeline.prepare_data()")
    print("4. trainer = pipeline.train()")
    print("5. predictions = pipeline.run_inference()")
    print("\nFor single predictions:")
    print("6. prediction = pipeline.predict_single(image_path, query_text)")
    print("\nFor test data processing:")
    print("7. pipeline.training_pipeline.process_test_dataset()")
    print("\nFor token analysis:")
    print("8. pipeline.training_pipeline.analyze_dataset_tokens()")


if __name__ == "__main__":
    main()
