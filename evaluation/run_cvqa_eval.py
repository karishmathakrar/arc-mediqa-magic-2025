#!/usr/bin/env python
"""
Evaluation script for CVQA that handles partial results (subset of encounters).
This script only evaluates encounters that exist in both reference and prediction datasets.
"""
import sys
import json
import os
from pprint import pprint

# Import functions from score_cvqa.py but don't import the module directly
# to avoid namespace conflicts
from score_cvqa import organize_values, get_accuracy_score

def calculate_accuracy_partial(qid2val_byencounterid_gold, qid2val_byencounterid_sys, qidparents):
    """
    Calculate accuracy only for encounters that exist in both datasets.
    
    Args:
        qid2val_byencounterid_gold: Reference data organized by encounter and question IDs
        qid2val_byencounterid_sys: Prediction data organized by encounter and question IDs
        qidparents: List of question category IDs
        
    Returns:
        Dictionary of accuracy results
    """
    results = {}
    x_all = []
    y_all = []
    
    # Only process encounters that exist in both datasets
    common_encounter_ids = set(qid2val_byencounterid_gold.keys()).intersection(set(qid2val_byencounterid_sys.keys()))
    common_encounter_ids = sorted(list(common_encounter_ids))
    
    # Early return if no common encounters
    if not common_encounter_ids:
        print("ERROR: No encounters in common between reference and prediction data")
        return {
            "error": "No common encounters to evaluate",
            "accuracy_all": 0.0,
            "number_evaluated_encounters": 0
        }
    
    print(f"Evaluating {len(common_encounter_ids)} encounters (out of {len(qid2val_byencounterid_gold)} total)")
    
    for qid in qidparents:
        goldlist = []
        syslist = []
        
        for encounter_id in common_encounter_ids:
            # Check if this question category exists for this encounter in both datasets
            if qid in qid2val_byencounterid_gold.get(encounter_id, {}) and qid in qid2val_byencounterid_sys.get(encounter_id, {}):
                goldlist.append(qid2val_byencounterid_gold[encounter_id][qid])
                syslist.append(qid2val_byencounterid_sys[encounter_id][qid])
        
        if goldlist and syslist:
            x_all.extend(goldlist)
            y_all.extend(syslist)
            results[f'accuracy_{qid}'] = get_accuracy_score(goldlist, syslist)
            results[f'count_{qid}'] = len(goldlist)
    
    if x_all and y_all:
        results['accuracy_all'] = get_accuracy_score(x_all, y_all)
    else:
        results['accuracy_all'] = 0.0
        
    results['number_evaluated_encounters'] = len(common_encounter_ids)
    results['evaluated_encounter_ids'] = common_encounter_ids
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        reference_fn = sys.argv[1]
    else:
        reference_fn = '2025_dataset/train/train_cvqa.json'
        
    if len(sys.argv) > 2:
        prediction_fn = sys.argv[2]
    else:
        prediction_fn = 'data_cvqa_sys.json'
        
    if len(sys.argv) > 3:
        score_dir = sys.argv[3]
    else:
        score_dir = 'evaluation_results'
    
    # Create output directory if it doesn't exist
    os.makedirs(score_dir, exist_ok=True)
    
    print(f"Running CVQA evaluation...")
    print(f"Reference file: {reference_fn}")
    print(f"Prediction file: {prediction_fn}")
    print(f"Output directory: {score_dir}")
    
    # Load data
    with open(reference_fn) as f:
        data_ref = json.load(f)
    with open(prediction_fn) as f:
        data_sys = json.load(f)
    
    print(f'Detected {len(data_ref)} instances for reference.')
    print(f'Detected {len(data_sys)} instances for predictions.')

    # Define question categories
    qid_parents = [
        "CQID010", "CQID011", "CQID012", "CQID015", 
        "CQID020", "CQID025", "CQID034", "CQID035", "CQID036"
    ]
    
    # Organize values by encounter and question IDs
    print('Organizing Values by Question IDs')
    try:
        qid2val_byencounterid_gold = organize_values(data_ref)
        qid2val_byencounterid_sys = organize_values(data_sys)
    except Exception as e:
        print(f"Error organizing values: {e}")
        sys.exit(1)
    
    # Calculate accuracy
    print('Calculating Accuracy')
    results = calculate_accuracy_partial(qid2val_byencounterid_gold, qid2val_byencounterid_sys, qid_parents)
    
    # Save results
    output_file = os.path.join(score_dir, 'scores_cvqa.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Also save as general scores.json for compatibility
    general_output = os.path.join(score_dir, 'scores.json')
    with open(general_output, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results
    print("\nEvaluation Results:")
    pprint(results)
    
    print(f"\nResults saved to {output_file} and {general_output}")