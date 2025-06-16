#!/usr/bin/env python3
"""
Medical CVQA Evaluation Script
Evaluates prediction results against ground truth using official evaluation metrics.
"""

import os
import sys
import json
import re
import datetime


def load_official_evaluation():
    """
    Try to load the official evaluation script.
    """
    try:
        # Add the path to the official evaluation scripts
        eval_path = os.path.join(os.getcwd(), "evaluation")
        if eval_path not in sys.path:
            sys.path.append(eval_path)
        
        import score_cvqa
        return score_cvqa
    except ImportError as e:
        print(f"Error: Could not import official evaluation script: {e}")
        print("Please ensure the 'evaluation' directory with score_cvqa.py is available.")
        return None


def extract_model_info(filename):
    """
    Extract model name and timestamp from prediction filename.
    
    Args:
        filename: Name of the prediction file
        
    Returns:
        String with model identifier
    """
    # Try to extract model and timestamp using regex
    match = re.search(r'data_cvqa_sys_([^_]+)_(\d+)', filename)
    if match:
        model_name = match.group(1)
        timestamp = match.group(2)
        return f"{model_name}_{timestamp}"
    else:
        # If no match, use the filename without extension as identifier
        return os.path.splitext(filename)[0].replace("data_cvqa_sys_", "")


def create_subset_reference(reference_file, prediction_file, output_dir, model_info):
    """
    Create a subset of the reference file that matches the encounters in predictions.
    
    Args:
        reference_file: Path to the full reference file
        prediction_file: Path to the prediction file
        output_dir: Directory to save the subset reference
        model_info: Model identifier string
        
    Returns:
        Path to the created subset reference file
    """
    # Load predictions to get encounter IDs
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    
    # Load full reference
    with open(reference_file, 'r') as f:
        reference_data = json.load(f)
    
    # Get encountered IDs we have predictions for
    prediction_encounters = {item['encounter_id'] for item in predictions}
    
    # Filter reference to only include those encounters
    subset_reference = [item for item in reference_data 
                       if item['encounter_id'] in prediction_encounters]
    
    # Save filtered reference temporarily with model info
    subset_reference_file = os.path.join(output_dir, f"subset_reference_{model_info}.json")
    with open(subset_reference_file, 'w') as f:
        json.dump(subset_reference, f, indent=2)
    
    return subset_reference_file


def evaluate_predictions(reference_file, prediction_file, output_dir=None):
    """
    Evaluate predictions against reference using official evaluation.
    
    Args:
        reference_file: Path to the reference/ground truth file
        prediction_file: Path to the prediction file
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    # Load official evaluation script
    score_cvqa = load_official_evaluation()
    if score_cvqa is None:
        return None
    
    # Set default output directory
    if output_dir is None:
        output_dir = "outputs/evaluation"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model info from filename
    filename = os.path.basename(prediction_file)
    model_info = extract_model_info(filename)
    
    # Create subset reference file
    subset_reference_file = create_subset_reference(
        reference_file, prediction_file, output_dir, model_info
    )
    
    # Run evaluation
    try:
        results = score_cvqa.main(subset_reference_file, prediction_file)
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None
    
    # Add metadata to results
    results["model_info"] = model_info
    results["prediction_file"] = prediction_file
    results["evaluation_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to file with model info
    results_file = os.path.join(output_dir, f"scores_cvqa_{model_info}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results, results_file


def print_evaluation_summary(results):
    """
    Print a summary of evaluation results.
    
    Args:
        results: Dictionary with evaluation results
    """
    if not results:
        print("No results to display.")
        return
    
    print(f"\nEvaluation Results for: {results.get('model_info', 'Unknown Model')}")
    print("=" * 60)
    
    # Print overall accuracy
    overall_acc = results.get('accuracy_all', 0.0)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    # Print per-question accuracies
    print("\nPer-Question Accuracies:")
    print("-" * 30)
    for key, value in results.items():
        if key.startswith('accuracy_CQID'):
            qid = key.replace('accuracy_', '')
            print(f"{qid}: {value:.4f}")
    
    # Print additional metrics if available
    if 'total_questions' in results:
        print(f"\nTotal Questions Evaluated: {results['total_questions']}")
    
    if 'correct_predictions' in results:
        print(f"Correct Predictions: {results['correct_predictions']}")


def main():
    """Main function to run evaluation."""
    print("Medical CVQA Evaluation Script")
    print("=" * 40)
    
    # Default configuration
    reference_file = "2025_dataset/valid/valid_cvqa.json"
    prediction_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/val_cvqa_sys_test.json"
    output_dir = "outputs/evaluation"
    
    # Check if files exist
    if not os.path.exists(reference_file):
        print(f"Error: Reference file {reference_file} does not exist!")
        return 1
    
    if not os.path.exists(prediction_file):
        print(f"Error: Prediction file {prediction_file} does not exist!")
        return 1
    
    print(f"Reference file: {reference_file}")
    print(f"Prediction file: {prediction_file}")
    print(f"Output directory: {output_dir}")
    
    # Run evaluation
    try:
        evaluation_result = evaluate_predictions(reference_file, prediction_file, output_dir)
        
        if evaluation_result is None:
            print("Evaluation failed!")
            return 1
        
        results, results_file = evaluation_result
        
        # Print summary
        print_evaluation_summary(results)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
