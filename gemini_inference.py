#!/usr/bin/env python3
"""
Gemini 2.5 Flash Inference Script

This script runs inference on the validation dataset using Gemini 2.5 Flash
with the EXACT same prompting as used for BASE models in the finetuning pipeline.
"""

import os
import sys
import json
import pickle
import datetime
import traceback
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Import Gemini
from google import genai

# Add latest_code to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'latest_code'))

from data_preprocessor import DataPreprocessor
from finetuning_pipeline.pipeline import Config


class GeminiInference:
    """Handles inference using Gemini 2.5 Flash with exact BASE model prompting."""
    
    def __init__(self, api_key=None, model_name="gemini-2.5-flash-preview-04-17"):
        """
        Initialize Gemini inference.
        
        Args:
            api_key: Gemini API key (defaults to environment variable)
            model_name: Gemini model to use
        """
        if api_key is None:
            api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("API key not found. Set API_KEY or GEMINI_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Gemini inference with model: {model_name}")
    
    def predict(self, query_text, image_path, max_new_tokens=100):
        """
        Generate prediction for a single query and image using EXACT BASE model prompting.
        
        This uses the identical system message and format as the BASE models in 
        finetuning_pipeline/pipeline.py MedicalImageInference.predict() method.
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # EXACT system message from BASE model prompting in finetuning pipeline
            system_message = """You are a medical assistant. Your task is to examine the provided information, and select the option(s) that best answer my question.

            IMPORTANT: 
            - Respond ONLY with the exact text of the option(s) that apply
            - Do not provide any explanations
            - Do not include the numbers that appear before options (like '1.' or '2')
            - Do not write "Options:" or similar prefixes
            - Do not write "Answer:" or similar prefixes
            - Multiple answers should be separated by commas
            - If unsure, respond with "Not mentioned"
            - The 'Background Clinical Information' section contains context to help you answer the main question
            - ONLY answer the question listed under "MAIN QUESTION TO ANSWER:" at the beginning
            """
            
            # Create the prompt exactly as done in the BASE model
            prompt = f"{system_message}\n\n{query_text}"
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            
            prediction = response.text.strip()
            
            # Apply the EXACT same post-processing as BASE models
            # Clean up prediction text (from finetuning_pipeline/pipeline.py)
            if prediction.startswith("assistant\n\n"):
                prediction = prediction[len("assistant\n\n"):]
            if prediction.startswith("assistant\n"):
                prediction = prediction[len("assistant\n"):]
            if prediction.startswith("system\n"):
                prediction = prediction[len("system\n"):]
            if prediction.startswith("model\n"):
                prediction = prediction[len("model\n"):]

            import re
            prediction = re.sub(r'^\*+\s*', '', prediction)
            prediction = re.sub(r'\n\*+\s*', ' ', prediction)
            prediction = re.sub(r'\*\s*', '', prediction)

            if prediction.startswith("Note:") or prediction.startswith("Disclaimer:") or prediction.startswith("*Note:"):
                if "\n" in prediction:
                    prediction = prediction.split("\n", 1)[1].strip()

            if "Answer:" in prediction:
                parts = prediction.split("Answer:")
                if len(parts) > 1:
                    prediction = parts[1].strip()

            if prediction.endswith("."):
                prediction = prediction[:-1]

            if prediction.startswith("<start_of_turn>model") or prediction.startswith("<start_of_turn>assistant"):
                prediction = prediction.split("\n", 1)[1] if "\n" in prediction else ""
            if prediction.endswith("<end_of_turn>"):
                prediction = prediction[:-len("<end_of_turn>")]
            
            return prediction.strip()
            
        except Exception as e:
            print(f"Error during prediction for {image_path}: {e}")
            traceback.print_exc()
            return "Not mentioned"
    
    def batch_predict(self, processed_data_dir, output_file=None, max_samples=None):
        """
        Run inference on a batch of preprocessed validation data.
        
        Args:
            processed_data_dir: Directory containing processed validation batch files
            output_file: Output CSV file for predictions
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            DataFrame with predictions
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"gemini_predictions_{timestamp}.csv"
        
        results = []
        sample_count = 0
        
        # Find validation batch files
        batch_files = sorted([f for f in os.listdir(processed_data_dir) 
                             if f.startswith("val_batch_") and f.endswith(".pkl")])
        
        if not batch_files:
            print(f"Warning: No validation batch files found in {processed_data_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(batch_files)} validation batch files")
        
        for batch_file in tqdm(batch_files, desc="Processing batches"):
            with open(os.path.join(processed_data_dir, batch_file), 'rb') as f:
                batch_data = pickle.load(f)
            
            for sample in tqdm(batch_data, desc=f"Predicting {batch_file}", leave=False):
                prediction = self.predict(sample["query_text"], sample["image_path"])
                time.sleep(1)
                
                results.append({
                    "encounter_id": sample.get("encounter_id", sample.get("id", "")),
                    "base_qid": sample.get("base_qid", sample.get("qid", "")),
                    "image_id": os.path.basename(sample["image_path"]),
                    "prediction": prediction
                })
                
                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
        
        print(f"Processed {sample_count} samples for prediction")
        
        if not results:
            print("Warning: No prediction results generated")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
        
        return results_df
    
    def aggregate_predictions(self, predictions_df, validation_df=None):
        """
        Aggregate predictions for each encounter and question ID.
        Uses the EXACT same aggregation logic as the BASE models.
        """
        # Import aggregation logic from finetuning pipeline
        from collections import Counter
        import ast
        import random
        
        max_answers = {
            'CQID010': 1, 'CQID011': 6, 'CQID012': 6, 'CQID015': 1, 'CQID020': 9,
            'CQID025': 1, 'CQID034': 1, 'CQID035': 1, 'CQID036': 1
        }

        default_max_answers = 1
        grouped = predictions_df.groupby(['encounter_id', 'base_qid'])
        aggregated_results = []

        for (encounter_id, base_qid), group in tqdm(grouped, desc="Aggregating predictions"):
            predictions = group['prediction'].tolist()
            image_ids = group['image_id'].tolist()

            cleaned_predictions = []
            for pred in predictions:
                if isinstance(pred, str):
                    pred = pred.replace(" (please specify)", "")

                    if pred.startswith('[') and pred.endswith(']'):
                        try:
                            pred_list = ast.literal_eval(pred)
                            if isinstance(pred_list, list):
                                pred_list = [p.replace(" (please specify)", "") if isinstance(p, str) else p for p in pred_list]
                                cleaned_predictions.extend(pred_list)
                                continue
                        except:
                            pass

                    if ',' in pred:
                        items = [p.strip().replace(" (please specify)", "") for p in pred.split(',')]
                        cleaned_predictions.extend(items)
                    else:
                        cleaned_predictions.append(pred.strip())
                else:
                    cleaned_predictions.append(str(pred).strip())

            all_cleaned_predictions = cleaned_predictions.copy()
            cleaned_predictions = [p.lower() if isinstance(p, str) else str(p).lower() for p in cleaned_predictions]
            prediction_counts = Counter(cleaned_predictions)

            question_type = base_qid.split('-')[0] if '-' in base_qid else base_qid
            allowed_max = max_answers.get(question_type, default_max_answers)

            sorted_predictions = sorted(prediction_counts.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)

            top_predictions = [p[0] for p in sorted_predictions[:allowed_max]]

            if len(sorted_predictions) > allowed_max:
                cutoff_count = sorted_predictions[allowed_max-1][1]
                tied_predictions = [p[0] for p in sorted_predictions if p[1] == cutoff_count]

                if len(tied_predictions) > 1 and len(top_predictions) > allowed_max - len(tied_predictions):
                    top_predictions = [p for p in top_predictions if p not in tied_predictions]

                    random.seed(42)
                    slots_remaining = allowed_max - len(top_predictions)
                    selected_tied = random.sample(tied_predictions, slots_remaining)

                    top_predictions.extend(selected_tied)

            if len(top_predictions) > 1 and "not mentioned" in top_predictions:
                top_predictions.remove("not mentioned")

            combined_prediction = ", ".join(top_predictions)

            options_en = None
            if validation_df is not None:
                matching_rows = validation_df[(validation_df['encounter_id'] == encounter_id) & 
                                             (validation_df['base_qid'] == base_qid)]
                if not matching_rows.empty:
                    options_en = matching_rows.iloc[0].get('options_en')

            result_dict = {
                "encounter_id": encounter_id,
                "base_qid": base_qid,
                "image_ids": image_ids,
                "unique_predictions": top_predictions,
                "combined_prediction": combined_prediction,
                "all_raw_predictions": all_cleaned_predictions,
                "all_sorted_predictions": sorted_predictions
            }

            if options_en is not None:
                result_dict["options_en"] = options_en

            aggregated_results.append(result_dict)

        aggregated_df = pd.DataFrame(aggregated_results)
        return aggregated_df
    
    def format_predictions_for_evaluation(self, aggregated_df, output_file):
        """Format predictions for official evaluation using EXACT same logic as BASE models."""
        QIDS = [
            "CQID010-001",
            "CQID011-001", "CQID011-002", "CQID011-003", "CQID011-004", "CQID011-005", "CQID011-006",
            "CQID012-001", "CQID012-002", "CQID012-003", "CQID012-004", "CQID012-005", "CQID012-006",
            "CQID015-001",
            "CQID020-001", "CQID020-002", "CQID020-003", "CQID020-004", "CQID020-005", 
            "CQID020-006", "CQID020-007", "CQID020-008", "CQID020-009",
            "CQID025-001",
            "CQID034-001",
            "CQID035-001",
            "CQID036-001",
        ]
        
        qid_variants = {}
        for qid in QIDS:
            base_qid, variant = qid.split('-')
            if base_qid not in qid_variants:
                qid_variants[base_qid] = []
            qid_variants[base_qid].append(qid)
        
        required_base_qids = set(qid.split('-')[0] for qid in QIDS)
        
        formatted_predictions = []
        for encounter_id, group in aggregated_df.groupby('encounter_id'):
            encounter_base_qids = set(group['base_qid'].unique())
            
            if not required_base_qids.issubset(encounter_base_qids):
                print(f"Skipping encounter {encounter_id} - missing required questions")
                continue
            
            pred_entry = {'encounter_id': encounter_id}
            
            for _, row in group.iterrows():
                base_qid = row['base_qid']
                
                if base_qid not in qid_variants:
                    continue
                
                # Use safe_convert_options from data_preprocessor
                from data_preprocessor import DataPreprocessor
                preprocessor = DataPreprocessor(None)
                options = preprocessor.safe_convert_options(row.get('options_en', []))
                
                not_mentioned_index = None
                for i, opt in enumerate(options):
                    if opt == "Not mentioned":
                        not_mentioned_index = i
                        break
                
                if not_mentioned_index is None:
                    not_mentioned_index = len(options) - 1
                
                if 'unique_predictions' in row and row['unique_predictions']:
                    answers = row['unique_predictions']
                    if isinstance(answers, str):
                        try:
                            import ast
                            answers = ast.literal_eval(answers)
                        except:
                            answers = [answers]
                    
                    prediction_indices = []
                    for pred in answers:
                        pred_text = str(pred).strip()
                        
                        found = False
                        for i, option in enumerate(options):
                            clean_option = option.replace(" (please specify)", "").lower()
                            
                            if pred_text.lower() == clean_option:
                                prediction_indices.append(i)
                                found = True
                                break
                        
                        if not found:
                            prediction_indices.append(not_mentioned_index)
                    
                    unique_indices = []
                    for idx in prediction_indices:
                        if idx not in unique_indices:
                            unique_indices.append(idx)
                    
                    if len(unique_indices) > 1 and not_mentioned_index in unique_indices:
                        unique_indices.remove(not_mentioned_index)
                    
                    available_variants = qid_variants[base_qid]
                    
                    if len(available_variants) == 1:
                        if unique_indices:
                            pred_entry[available_variants[0]] = unique_indices[0]
                        else:
                            pred_entry[available_variants[0]] = not_mentioned_index
                    else:
                        for i, idx in enumerate(unique_indices):
                            if i < len(available_variants):
                                pred_entry[available_variants[i]] = idx
                        
                        for i in range(len(unique_indices), len(available_variants)):
                            pred_entry[available_variants[i]] = not_mentioned_index
                else:
                    # Default to not mentioned for all variants
                    for variant in qid_variants[base_qid]:
                        pred_entry[variant] = not_mentioned_index
            
            formatted_predictions.append(pred_entry)
        
        with open(output_file, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)
        
        print(f"Formatted predictions saved to {output_file} ({len(formatted_predictions)} complete encounters)")
        return formatted_predictions


def main():
    """Main function to run Gemini inference on validation dataset."""
    print("Gemini 2.5 Flash Inference on Validation Dataset")
    print("=" * 60)
    
    # Initialize configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config(
        model_name="Qwen2-VL-2B-Instruct",  # Model doesn't matter for data paths
        base_dir=base_dir,
        output_dir=os.path.join(base_dir, "outputs"),
        setup_environment=False,
        validate_paths=True
    )
    
    # Check if validation dataset exists
    val_dataset_path = os.path.join(config.OUTPUT_DIR, "val_dataset.csv")
    if not os.path.exists(val_dataset_path):
        print("Validation dataset not found. Please run generate_val_dataset.py first.")
        return 1
    
    # Check if processed validation data exists
    if not os.path.exists(config.PROCESSED_VAL_DATA_DIR):
        print("Processed validation data not found. Please run generate_val_dataset.py first.")
        return 1
    
    # Load validation dataset for aggregation
    print("Loading validation dataset...")
    validation_df = pd.read_csv(val_dataset_path)
    print(f"Loaded validation dataset with {len(validation_df)} entries")
    
    # Initialize Gemini inference
    print("Initializing Gemini inference...")
    gemini = GeminiInference()
    
    # Run inference
    print("Running Gemini inference on validation data...")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_file = os.path.join(config.OUTPUT_DIR, f"gemini_val_predictions_{timestamp}.csv")
    
    predictions_df = gemini.batch_predict(
        processed_data_dir=config.PROCESSED_VAL_DATA_DIR,
        output_file=predictions_file,
        max_samples=None  # Process all samples
    )
    
    if predictions_df.empty:
        print("No predictions generated. Exiting.")
        return 1
    
    # Aggregate predictions
    print("Aggregating predictions...")
    aggregated_df = gemini.aggregate_predictions(predictions_df, validation_df)
    
    aggregated_file = os.path.join(config.OUTPUT_DIR, f"gemini_val_aggregated_{timestamp}.csv")
    aggregated_df.to_csv(aggregated_file, index=False)
    print(f"Aggregated predictions saved to: {aggregated_file}")
    
    # Format for evaluation - EXACT same naming as finetuning pipeline
    print("Formatting predictions for evaluation...")
    formatted_file = os.path.join(
        config.OUTPUT_DIR,
        f"val_data_cvqa_sys_gemini-2.5-flash_{timestamp}.json"
    )
    formatted_predictions = gemini.format_predictions_for_evaluation(aggregated_df, formatted_file)
    
    print("\nGemini inference completed successfully!")
    print(f"Raw predictions: {predictions_file}")
    print(f"Aggregated predictions: {aggregated_file}")
    print(f"Formatted for evaluation: {formatted_file}")
    print(f"Total encounters processed: {len(formatted_predictions)}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
