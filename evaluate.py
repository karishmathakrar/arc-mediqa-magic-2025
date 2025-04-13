"""
Evaluation script for the MEDIQA project.
Runs evaluation on inference results using the conference evaluation scripts.
"""
import os
import sys
import argparse
import logging
import json
import subprocess
from pprint import pprint

import config
from utils import print_system_info, save_json_file, load_json_file

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
        default=os.path.join(config.DATASET_DIR, "train_cvqa.json"),
        help="Path to reference data JSON file"
    )
    parser.add_argument(
        "--base_prediction_file", 
        type=str, 
        default=os.path.join(config.OUTPUTS_DIR, "base_model", "results.json"),
        help="Path to base model prediction data"
    )
    parser.add_argument(
        "--finetuned_prediction_file", 
        type=str, 
        default=os.path.join(config.OUTPUTS_DIR, "finetuned_model", "results.json"),
        help="Path to fine-tuned model prediction data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=config.OUTPUTS_DIR,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--option_maps", 
        type=str, 
        default=os.path.join(config.DATASET_DIR, "option_maps.json"),
        help="Path to option maps JSON file"
    )
    parser.add_argument(
        "--skip_base", 
        action="store_true",
        help="Skip evaluation of base model"
    )
    parser.add_argument(
        "--skip_finetuned", 
        action="store_true",
        help="Skip evaluation of fine-tuned model"
    )
    
    return parser.parse_args()

def convert_text_to_indices(results_file, output_file, option_maps_file):
    """Convert text predictions to indices for the evaluation."""
    logger.info(f"Converting text predictions to indices: {results_file}")
    
    # Load data
    with open(results_file, "r") as f:
        results = json.load(f)
    
    with open(option_maps_file, "r") as f:
        option_maps = json.load(f)
    
    # Convert answers to indices
    indexed_results = []
    
    for encounter in results:
        indexed_encounter = {"encounter_id": encounter["encounter_id"]}
        
        for key, value in encounter.items():
            if key == "encounter_id":
                continue
            
            # Get the base question ID
            base_qid = key.split("-")[0]
            
            if base_qid in option_maps:
                # Clean the text
                text = value.strip()
                
                # Remove option number if present (e.g., "1. single spot" -> "single spot")
                if text and text[0].isdigit() and ". " in text:
                    text = text.split(". ", 1)[1]
                
                # Find the index
                if text in option_maps[base_qid]:
                    indexed_encounter[key] = option_maps[base_qid][text]
                else:
                    # Try case-insensitive matching
                    found = False
                    for opt_text, idx in option_maps[base_qid].items():
                        if opt_text.lower() == text.lower():
                            indexed_encounter[key] = idx
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"No match found for '{text}' in question {key}")
                        # Default to "Not mentioned" if available
                        if "Not mentioned" in option_maps[base_qid]:
                            indexed_encounter[key] = option_maps[base_qid]["Not mentioned"]
                        else:
                            indexed_encounter[key] = -1
            else:
                logger.warning(f"No option map found for question {base_qid}")
                indexed_encounter[key] = -1
        
        indexed_results.append(indexed_encounter)
    
    # Save indexed results
    with open(output_file, "w") as f:
        json.dump(indexed_results, f, indent=2)
    
    logger.info(f"Saved indexed results to {output_file}")
    return indexed_results

def run_conference_evaluation(reference_file, prediction_file, output_dir):
    """Run the conference evaluation script."""
    logger.info(f"Running evaluation: {reference_file} vs {prediction_file}")
    
    # Ensure the evaluation output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to evaluation script
    eval_script = os.path.join("evaluation", "run_cvqa_eval.py")
    
    # Run the evaluation script as a subprocess
    command = [
        sys.executable,
        eval_script,
        reference_file,
        prediction_file,
        output_dir
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info(f"Evaluation completed with exit code: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            return None
        
        # Load the evaluation results
        scores_file = os.path.join(output_dir, "scores_cvqa.json")
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                scores = json.load(f)
            return scores
        else:
            logger.error(f"Evaluation results file not found: {scores_file}")
            return None
    
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return None

def main():
    """Main entry point for evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Print system info
    print_system_info()
    
    # Create output directories
    base_output_dir = os.path.join(args.output_dir, "base_model", "evaluation")
    finetuned_output_dir = os.path.join(args.output_dir, "finetuned_model", "evaluation")
    comparison_dir = os.path.join(args.output_dir, "comparison")
    
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(finetuned_output_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Process base model results if not skipped
    if not args.skip_base and os.path.exists(args.base_prediction_file):
        # Convert text predictions to indices
        base_indexed_file = os.path.join(os.path.dirname(args.base_prediction_file), "results_indices.json")
        convert_text_to_indices(args.base_prediction_file, base_indexed_file, args.option_maps)
        
        # Run evaluation
        base_scores = run_conference_evaluation(args.reference_file, base_indexed_file, base_output_dir)
        
        if base_scores:
            logger.info("\nBase Model Evaluation Results:")
            print_accuracy_metrics(base_scores)
    else:
        logger.info("Skipping base model evaluation")
        base_scores = None
    
    # Process fine-tuned model results if not skipped
    if not args.skip_finetuned and os.path.exists(args.finetuned_prediction_file):
        # Convert text predictions to indices
        finetuned_indexed_file = os.path.join(os.path.dirname(args.finetuned_prediction_file), "results_indices.json")
        convert_text_to_indices(args.finetuned_prediction_file, finetuned_indexed_file, args.option_maps)
        
        # Run evaluation
        finetuned_scores = run_conference_evaluation(args.reference_file, finetuned_indexed_file, finetuned_output_dir)
        
        if finetuned_scores:
            logger.info("\nFine-tuned Model Evaluation Results:")
            print_accuracy_metrics(finetuned_scores)
    else:
        logger.info("Skipping fine-tuned model evaluation")
        finetuned_scores = None
    
    # Generate comparison if both models were evaluated
    if base_scores and finetuned_scores:
        generate_comparison(base_scores, finetuned_scores, comparison_dir)

def print_accuracy_metrics(scores):
    """Print the accuracy metrics from scores."""
    # Print overall accuracy
    logger.info(f"Overall accuracy: {scores.get('accuracy_all', 0.0):.4f}")
    
    # Print per-category accuracies
    for key, value in scores.items():
        if key.startswith("accuracy_CQID") and isinstance(value, (int, float)):
            category = key.replace("accuracy_", "")
            logger.info(f"{category} accuracy: {value:.4f}")
    
    # Print number of evaluated encounters
    logger.info(f"Number of evaluated encounters: {scores.get('number_evaluated_encounters', 0)}")

def generate_comparison(base_scores, finetuned_scores, output_dir):
    """Generate a comparison between base and fine-tuned model scores."""
    logger.info("\nGenerating model comparison...")
    
    comparison = {
        "base_model": {},
        "finetuned_model": {},
        "difference": {},
        "percent_improvement": {}
    }
    
    # Extract metrics from both models
    for key, base_value in base_scores.items():
        if key.startswith("accuracy_") and isinstance(base_value, (int, float)):
            finetuned_value = finetuned_scores.get(key, 0.0)
            
            comparison["base_model"][key] = base_value
            comparison["finetuned_model"][key] = finetuned_value
            comparison["difference"][key] = finetuned_value - base_value
            
            # Calculate percent improvement
            if base_value > 0:
                percent_improvement = ((finetuned_value - base_value) / base_value) * 100
                comparison["percent_improvement"][key] = percent_improvement
            else:
                comparison["percent_improvement"][key] = float('inf') if finetuned_value > 0 else 0.0
    
    # Save comparison
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    save_json_file(comparison, comparison_file)
    
    # Print comparison summary
    logger.info("\nComparison Summary:")
    logger.info(f"Overall accuracy - Base: {comparison['base_model'].get('accuracy_all', 0.0):.4f}, Fine-tuned: {comparison['finetuned_model'].get('accuracy_all', 0.0):.4f}")
    logger.info(f"Absolute improvement: {comparison['difference'].get('accuracy_all', 0.0):.4f}")
    logger.info(f"Relative improvement: {comparison['percent_improvement'].get('accuracy_all', 0.0):.2f}%")
    
    # Generate a simple markdown report
    generate_markdown_report(comparison, output_dir)

def generate_markdown_report(comparison, output_dir):
    """Generate a markdown report with the comparison results."""
    report_path = os.path.join(output_dir, "model_comparison.md")
    
    with open(report_path, "w") as f:
        f.write("# Model Comparison: Base vs. Fine-tuned\n\n")
        
        # Overall results
        f.write("## Overall Results\n\n")
        f.write("| Metric | Base Model | Fine-tuned Model | Absolute Diff | Relative Improvement |\n")
        f.write("|--------|------------|-----------------|---------------|---------------------|\n")
        
        overall_key = "accuracy_all"
        base_overall = comparison["base_model"].get(overall_key, 0.0)
        ft_overall = comparison["finetuned_model"].get(overall_key, 0.0)
        abs_diff = comparison["difference"].get(overall_key, 0.0)
        rel_imp = comparison["percent_improvement"].get(overall_key, 0.0)
        
        f.write(f"| Overall Accuracy | {base_overall:.4f} | {ft_overall:.4f} | {abs_diff:.4f} | {rel_imp:.2f}% |\n\n")
        
        # Per-category results
        f.write("## Results by Category\n\n")
        f.write("| Category | Base Model | Fine-tuned Model | Absolute Diff | Relative Improvement |\n")
        f.write("|----------|------------|-----------------|---------------|---------------------|\n")
        
        for key in sorted(comparison["base_model"].keys()):
            if key != "accuracy_all" and key.startswith("accuracy_"):
                category = key.replace("accuracy_", "")
                base_value = comparison["base_model"].get(key, 0.0)
                ft_value = comparison["finetuned_model"].get(key, 0.0)
                abs_diff = comparison["difference"].get(key, 0.0)
                rel_imp = comparison["percent_improvement"].get(key, 0.0)
                
                f.write(f"| {category} | {base_value:.4f} | {ft_value:.4f} | {abs_diff:.4f} | {rel_imp:.2f}% |\n")
    
    logger.info(f"Markdown report saved to: {report_path}")

if __name__ == "__main__":
    main()