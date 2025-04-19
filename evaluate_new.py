#!/usr/bin/env python
import os
import sys
import json

# Add the path to the official evaluation scripts
sys.path.append(os.path.join(os.getcwd(), "evaluation"))
import score_cvqa

def main():
    reference_file = "2025_dataset/valid/valid_cvqa.json"
    prediction_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/val_cvqa_sys_test.json"
    output_dir = "outputs/evaluation"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Save filtered reference temporarily
    subset_reference_file = os.path.join(output_dir, "subset_reference.json")
    with open(subset_reference_file, 'w') as f:
        json.dump(subset_reference, f, indent=2)
    
    # Run evaluation
    results = score_cvqa.main(subset_reference_file, prediction_file)
    
    # Save results to file
    results_file = os.path.join(output_dir, "scores_cvqa.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {results['accuracy_all']:.4f}")
    for key, value in results.items():
        if key.startswith('accuracy_CQID'):
            print(f"{key}: {value:.4f}")
    
    print(f"\nResults saved to {results_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())