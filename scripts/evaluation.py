#!/usr/bin/env python
"""
Unified evaluation script for the MEDIQA project.
Handles evaluation of model predictions against ground truth.
"""
import os
import sys
import argparse
import logging
import json
import pandas as pd
from pprint import pprint

# Add parent directory to path to allow absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions from evaluation modules
from scripts.score_cvqa import organize_values, get_accuracy_score

import utils.utils as utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument(
        "--reference_file", 
        type=str, 
        default="2025_dataset/train/train_cvqa.json",
        help="Path to reference data JSON file"
    )
    parser.add_argument(
        "--prediction_file", 
        type=str, 
        default="evaluation_results/results.json",
        help="Path to prediction data JSON file"
    )
    parser.add_argument(
        "--indexed_prediction_file", 
        type=str, 
        default=None,
        help="Path to indexed prediction data (optional)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--option_maps", 
        type=str, 
        default="2025_dataset/train/option_maps.json",
        help="Path to option maps JSON file"
    )
    parser.add_argument(
        "--text_predictions", 
        action="store_true",
        help="Set if predictions are text rather than indices"
    )
    
    return parser.parse_args()

def convert_text_to_indices(predictions, option_maps):
    """
    Convert text predictions to indices using option maps.
    
    Args:
        predictions: List of prediction dictionaries with text answers
        option_maps: Dictionary mapping question IDs to option indices
        
    Returns:
        List of prediction dictionaries with index answers
    """
    indexed_predictions = []
    
    for entry in predictions:
        indexed_entry = {"encounter_id": entry["encounter_id"]}
        
        for key, value in entry.items():
            if key == "encounter_id":
                continue
                
            # Get base question ID
            base_qid = key.split("-")[0]
            
            if base_qid in option_maps:
                # Clean text answer
                text_answer = value.strip()
                
                # Remove option number prefix if present
                text_answer = text_answer.split('. ', 1)[-1] if text_answer.split('. ', 1)[0].isdigit() else text_answer
                
                # Find matching option
                if text_answer in option_maps[base_qid]:
                    indexed_entry[key] = option_maps[base_qid][text_answer]
                else:
                    # Try case-insensitive match
                    match_found = False
                    for opt_text, opt_idx in option_maps[base_qid].items():
                        if opt_text.lower() == text_answer.lower():
                            indexed_entry[key] = opt_idx
                            match_found = True
                            break
                    
                    if not match_found:
                        logger.warning(f"No matching option found for {key}: '{text_answer}'")
                        indexed_entry[key] = -1
            else:
                logger.warning(f"No option mapping found for question: {base_qid}")
                indexed_entry[key] = -1
        
        indexed_predictions.append(indexed_entry)
    
    return indexed_predictions

def calculate_accuracy_partial(reference_data, prediction_data, question_categories):
    """
    Calculate accuracy only for encounters that exist in both datasets.
    
    Args:
        reference_data: Dictionary mapping encounter IDs to question answers (ground truth)
        prediction_data: Dictionary mapping encounter IDs to question answers (predictions)
        question_categories: List of question category IDs
        
    Returns:
        Dictionary of accuracy results
    """
    results = {}
    all_gold_values = []
    all_pred_values = []
    
    # Get encounter IDs that exist in both datasets
    common_encounter_ids = set(reference_data.keys()).intersection(set(prediction_data.keys()))
    
    if not common_encounter_ids:
        logger.error("No encounters in common between reference and prediction data")
        return {
            "error": "No common encounters to evaluate",
            "accuracy_all": 0.0,
            "number_evaluated_encounters": 0
        }
    
    # Convert to list and sort for reproducibility
    common_encounter_ids = sorted(list(common_encounter_ids))
    logger.info(f"Evaluating {len(common_encounter_ids)} encounters (out of {len(reference_data)} total)")
    
    # Calculate accuracy for each question category
    for qid in question_categories:
        gold_values = []
        pred_values = []
        
        for encounter_id in common_encounter_ids:
            # Check if this question category exists for this encounter in both datasets
            if qid in reference_data.get(encounter_id, {}) and qid in prediction_data.get(encounter_id, {}):
                gold_values.append(reference_data[encounter_id][qid])
                pred_values.append(prediction_data[encounter_id][qid])
        
        if gold_values and pred_values:
            all_gold_values.extend(gold_values)
            all_pred_values.extend(pred_values)
            results[f'accuracy_{qid}'] = get_accuracy_score(gold_values, pred_values)
            results[f'count_{qid}'] = len(gold_values)
    
    # Calculate overall accuracy
    if all_gold_values and all_pred_values:
        results['accuracy_all'] = get_accuracy_score(all_gold_values, all_pred_values)
    else:
        results['accuracy_all'] = 0.0
    
    # Add metadata
    results['number_evaluated_encounters'] = len(common_encounter_ids)
    results['evaluated_encounter_ids'] = common_encounter_ids
    
    return results

def evaluate_predictions(reference_file, prediction_file, output_dir, option_maps_file=None, text_predictions=False):
    """
    Evaluate model predictions against reference data.
    
    Args:
        reference_file: Path to reference data JSON file
        prediction_file: Path to prediction data JSON file
        output_dir: Directory to save evaluation results
        option_maps_file: Path to option maps JSON file (required for text predictions)
        text_predictions: Whether predictions are text rather than indices
        
    Returns:
        Dictionary of evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Evaluating predictions from {prediction_file}")
    logger.info(f"Reference data: {reference_file}")
    
    # Load reference data
    with open(reference_file, 'r') as f:
        reference_data = json.load(f)
    
    # Load prediction data
    with open(prediction_file, 'r') as f:
        prediction_data = json.load(f)
    
    logger.info(f"Loaded {len(reference_data)} reference items")
    logger.info(f"Loaded {len(prediction_data)} prediction items")
    
    # If predictions are text answers, convert to indices
    if text_predictions:
        if not option_maps_file or not os.path.exists(option_maps_file):
            logger.error("Option maps file is required for text predictions")
            return {"error": "Option maps file not found"}
        
        # Load option maps
        with open(option_maps_file, 'r') as f:
            option_maps = json.load(f)
        
        logger.info("Converting text predictions to indices")
        indexed_predictions = convert_text_to_indices(prediction_data, option_maps)
        
        # Save indexed predictions
        indexed_file = os.path.join(output_dir, "results_indices.json")
        with open(indexed_file, 'w') as f:
            json.dump(indexed_predictions, f, indent=2)
        
        logger.info(f"Saved indexed predictions to {indexed_file}")
        
        # Use indexed predictions for evaluation
        prediction_data = indexed_predictions
    
    # Define question categories to evaluate
    question_categories = [
        "CQID010", "CQID011", "CQID012", "CQID015", 
        "CQID020", "CQID025", "CQID034", "CQID035", "CQID036"
    ]
    
    # Organize data by encounter ID and question ID
    logger.info("Organizing reference and prediction data")
    reference_organized = organize_values(reference_data)
    prediction_organized = organize_values(prediction_data)
    
    # Calculate accuracy
    logger.info("Calculating accuracy metrics")
    results = calculate_accuracy_partial(reference_organized, prediction_organized, question_categories)
    
    # Save results
    results_file = os.path.join(output_dir, "scores.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Also save as specific scores file for compatibility
    cvqa_scores_file = os.path.join(output_dir, "scores_cvqa.json")
    with open(cvqa_scores_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {results_file} and {cvqa_scores_file}")
    
    return results

def main():
    """Main entry point for evaluation."""
    args = parse_args()
    
    # Print system info
    utils.print_system_info()
    
    # Evaluate predictions
    results = evaluate_predictions(
        args.reference_file,
        args.prediction_file,
        args.output_dir,
        args.option_maps,
        args.text_predictions
    )
    
    # Print results
    logger.info("\nEvaluation Results:")
    pprint(results)

if __name__ == "__main__":
    main()