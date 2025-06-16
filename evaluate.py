#!/usr/bin/env python
import os
import sys
import json
import re

# Add the path to the official evaluation scripts
sys.path.append(os.path.join(os.getcwd(), "evaluation"))
import score_cvqa

def main():
    reference_file = "2025_dataset/valid/valid_cvqa.json"
    prediction_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/val_cvqa_sys_test.json"
    output_dir = "outputs/evaluation"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model name and timestamp from prediction file
    model_info = ""
    filename = os.path.basename(prediction_file)
    
    # Try to extract model and timestamp using regex
    match = re.search(r'data_cvqa_sys_([^_]+)_(\d+)', filename)
    if match:
        model_name = match.group(1)
        timestamp = match.group(2)
        model_info = f"{model_name}_{timestamp}"
    else:
        # If no match, use the filename without extension as identifier
        model_info = os.path.splitext(filename)[0].replace("data_cvqa_sys_", "")
    
    # Load predictions
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
    
    # Run evaluation
    results = score_cvqa.main(subset_reference_file, prediction_file)
    
    # Add metadata to results
    results["model_info"] = model_info
    results["prediction_file"] = prediction_file
    results["evaluation_timestamp"] = import_time = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to file with model info
    results_file = os.path.join(output_dir, f"scores_cvqa_{model_info}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nEvaluation Results for:", model_info)
    print(f"Overall Accuracy: {results['accuracy_all']:.4f}")
    for key, value in results.items():
        if key.startswith('accuracy_CQID'):
            print(f"{key}: {value:.4f}")
    
    print(f"\nResults saved to {results_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())