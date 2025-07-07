#!/usr/bin/env python3
"""
RAG Pipeline - Usage Examples

This script demonstrates how to use the parameterizable RAG pipeline wrapper
for diagnosis-based medical analysis with knowledge retrieval.
"""

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from rag_pipeline import RAGConfig, RAGPipeline


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=== Example 1: Basic RAG Pipeline Usage ===")
    
    # Create a basic configuration
    config = RAGConfig(
        base_dir=base_dir,
        output_dir=f"{base_dir}/outputs"
    )
    
    # Initialize the pipeline
    pipeline = RAGPipeline(config)
    
    print(f"Pipeline initialized with configuration:")
    print(f"- Using {'test' if config.use_test_dataset else 'validation'} dataset")
    print(f"- Using {'fine-tuned' if config.use_finetuning else 'base'} model predictions")
    print(f"- Gemini model: {config.gemini_model}")
    print(f"- Max reflection cycles: {config.max_reflection_cycles}")
    print(f"- Confidence threshold: {config.confidence_threshold}")
    print()


def example_custom_configuration():
    """Example 2: Custom configuration for specific use case."""
    print("=== Example 2: Custom RAG Configuration ===")
    
    # Create a custom configuration
    config = RAGConfig(
        use_finetuning=True,
        use_test_dataset=False,  # Use validation dataset
        gemini_model="gemini-2.0-flash-exp-2025-01-29",
        max_reflection_cycles=3,
        confidence_threshold=0.8,
        save_intermediate_results=True,
        intermediate_save_frequency=3,
        # Knowledge base configuration
        top_k_semantic=10,
        top_k_keyword=10,
        top_k_hybrid=15,
        top_k_rerank=7,
        # Use consistent base directory
        base_dir=base_dir,
        output_dir=f"{base_dir}/outputs"
    )
    
    print(f"Custom configuration created:")
    print(f"- Dataset: {'test' if config.use_test_dataset else 'validation'}")
    print(f"- Model type: {'fine-tuned' if config.use_finetuning else 'base'}")
    print(f"- Reflection cycles: {config.max_reflection_cycles}")
    print(f"- Confidence threshold: {config.confidence_threshold}")
    print(f"- Top-K semantic: {config.top_k_semantic}")
    print(f"- Top-K hybrid: {config.top_k_hybrid}")
    print(f"- Base directory: {config.base_dir}")
    print(f"- Output directory: {config.output_dir}")
    print()


def example_process_single_encounter():
    """Example 3: Process a single encounter."""
    print("=== Example 3: Process Single Encounter ===")
    
    # Create configuration
    config = RAGConfig(
        use_test_dataset=False,  # Use validation for testing
        save_intermediate_results=True,
        base_dir=base_dir,
        output_dir=f"{base_dir}/outputs"
    )
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Example encounter ID (you would replace this with a real encounter ID)
    encounter_id = "ENC00001"
    
    print(f"Processing single encounter: {encounter_id}")
    print("Note: This would process the encounter through the complete RAG pipeline:")
    print("1. Load and analyze images")
    print("2. Extract clinical context")
    print("3. Retrieve relevant medical knowledge")
    print("4. Integrate all evidence sources")
    print("5. Apply reasoning with self-reflection")
    print("6. Generate final answers")
    print()
    
    try:
        results = pipeline.process_single_encounter(encounter_id)
        print(f"Successfully processed encounter {encounter_id}")
        print(f"Results contain {len(results[encounter_id])} questions")
    except Exception as e:
        print(f"Error processing encounter: {e}")


def example_process_sample_encounters():
    """Example 4: Process a sample of encounters for testing."""
    print("=== Example 4: Process Sample Encounters ===")
    
    # Create configuration optimized for sampling
    config = RAGConfig(
        use_test_dataset=False,  # Use validation for testing
        max_reflection_cycles=2,
        confidence_threshold=0.7,
        save_intermediate_results=True,
        intermediate_save_frequency=2,
        base_dir=base_dir,
        output_dir=f"{base_dir}/outputs"
    )
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Process 3 sample encounters
    num_samples = 3
    
    print(f"Processing {num_samples} sample encounters for testing")
    print("This is useful for:")
    print("- Testing the pipeline configuration")
    print("- Validating the setup before full processing")
    print("- Quick experimentation with parameters")
    print()
    
    try:
        results = pipeline.process_sample_encounters(num_samples)
        print(f"Successfully processed {len(results)} sample encounters")
    except Exception as e:
        print(f"Error processing samples: {e}")


def example_process_all_encounters():
    """Example 5: Process all encounters (production use)."""
    print("=== Example 5: Process All Encounters (Production) ===")
    
    # Create production configuration
    config = RAGConfig(
        use_finetuning=True,
        use_test_dataset=True,  # Use test dataset for final predictions
        gemini_model="gemini-2.5-flash-preview-04-17",
        max_reflection_cycles=2,
        confidence_threshold=0.75,
        save_intermediate_results=True,
        intermediate_save_frequency=5,
        # Optimized knowledge retrieval
        top_k_semantic=7,
        top_k_keyword=7,
        top_k_hybrid=10,
        top_k_rerank=5,
        base_dir=base_dir,
        output_dir=f"{base_dir}/outputs"
    )
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    print("Production configuration for processing all encounters:")
    print(f"- Dataset: {'test' if config.use_test_dataset else 'validation'}")
    print(f"- Model predictions: {'fine-tuned' if config.use_finetuning else 'base'}")
    print(f"- Intermediate saves every {config.intermediate_save_frequency} encounters")
    print(f"- Knowledge retrieval: top-{config.top_k_rerank} final results")
    print()
    print("This would process ALL encounters in the dataset.")
    print("Expected output:")
    print("- Complete results JSON file")
    print("- Formatted predictions for evaluation")
    print("- Intermediate checkpoint files")
    print()

    try:
        complete_results, formatted_predictions = pipeline.process_all_encounters()
        print(f"Successfully processed all encounters")
        print(f"Complete results: {len(complete_results)} encounters")
        print(f"Formatted predictions: {len(formatted_predictions)} encounters")
    except Exception as e:
        print(f"Error processing all encounters: {e}")


def example_custom_knowledge_configuration():
    """Example 6: Custom knowledge base and retrieval configuration."""
    print("=== Example 6: Custom Knowledge Configuration ===")
    
    # Custom question type retrieval configuration
    custom_retrieval_config = {
        "Site Location": {"use_rag": False, "weight": 0.1},
        "Lesion Color": {"use_rag": False, "weight": 0.1},
        "Size": {"use_rag": False, "weight": 0.1},
        "Skin Description": {"use_rag": True, "weight": 0.4},
        "Onset": {"use_rag": True, "weight": 0.5},
        "Itch": {"use_rag": True, "weight": 0.5},
        "Extent": {"use_rag": True, "weight": 0.3},  # Enable RAG for extent
        "Treatment": {"use_rag": True, "weight": 0.8},
        "Lesion Evolution": {"use_rag": True, "weight": 0.6},
        "Texture": {"use_rag": True, "weight": 0.4},
        "Lesion Count": {"use_rag": False, "weight": 0.1},
        "Differential": {"use_rag": True, "weight": 0.9},
        "Specific Diagnosis": {"use_rag": True, "weight": 0.9},
    }
    
    # Create configuration with custom knowledge settings
    config = RAGConfig(
        # Knowledge base configuration
        embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_semantic=10,
        top_k_keyword=8,
        top_k_hybrid=15,
        top_k_rerank=7,
        dataset_name_huggingface="brucewayne0459/Skin_diseases_and_care",
        # Custom retrieval configuration
        question_type_retrieval_config=custom_retrieval_config,
        default_rag_config={"use_rag": True, "weight": 0.5},
        base_dir=base_dir,
        output_dir=f"{base_dir}/outputs"
    )
    
    print("Custom knowledge configuration:")
    print(f"- Embedding model: {config.embedding_model}")
    print(f"- Cross-encoder: {config.cross_encoder_model}")
    print(f"- Semantic search: top-{config.top_k_semantic}")
    print(f"- Hybrid search: top-{config.top_k_hybrid}")
    print(f"- Final reranking: top-{config.top_k_rerank}")
    print(f"- HuggingFace dataset: {config.dataset_name_huggingface}")
    print()
    print("Question type RAG usage:")
    for q_type, settings in custom_retrieval_config.items():
        status = "enabled" if settings["use_rag"] else "disabled"
        print(f"- {q_type}: RAG {status} (weight: {settings['weight']})")
    print()


def example_directory_configuration():
    """Example 7: Custom directory and path configuration."""
    print("=== Example 7: Custom Directory Configuration ===")
    
    # Get current working directory as base
    current_dir = os.getcwd()
    
    # Create configuration with custom paths
    config = RAGConfig(
        # Directory configuration
        base_dir=current_dir,
        output_dir=os.path.join(current_dir, "custom_outputs"),
        model_predictions_dir=os.path.join(current_dir, "outputs", "model_predictions"),
        knowledge_db_path=os.path.join(current_dir, "custom_knowledge_db"),
        # Dataset paths (auto-determined based on base_dir if not specified)
        dataset_path=None,  # Will be auto-determined
        images_dir=None,    # Will be auto-determined
        # Processing configuration
        save_intermediate_results=True,
        intermediate_save_frequency=3
    )
    
    print("Custom directory configuration:")
    print(f"- Base directory: {config.base_dir}")
    print(f"- Output directory: {config.output_dir}")
    print(f"- Model predictions: {config.model_predictions_dir}")
    print(f"- Knowledge database: {config.knowledge_db_path}")
    print(f"- Dataset path: {config.dataset_path or 'Auto-determined'}")
    print(f"- Images directory: {config.images_dir or 'Auto-determined'}")
    print()
    print("The pipeline will:")
    print("- Create output directories if they don't exist")
    print("- Save intermediate results every 3 encounters")
    print("- Use custom knowledge database location")
    print()


def main():
    """Run all examples."""
    print("RAG Pipeline - Usage Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_process_single_encounter()
        example_process_sample_encounters()
        example_process_all_encounters()
        example_custom_knowledge_configuration()
        example_directory_configuration()
        
        print("=" * 50)
        print("All examples completed successfully!")
        print()
        print("To actually run the pipeline:")
        print("1. Ensure you have the required data files and API keys")
        print("2. Install required dependencies (see requirements.txt)")
        print("3. Run the specific example you want to test")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
