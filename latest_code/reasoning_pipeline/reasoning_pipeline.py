#!/usr/bin/env python3
"""
Reasoning-based Medical Analysis Pipeline
Uses Gemini to analyze medical images and clinical context with structured reasoning.
"""

import os
import glob
import pandas as pd
import ast
import re
from collections import defaultdict
import json
import datetime
import traceback
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from PIL import Image
from dotenv import load_dotenv
from google import genai


class Args:
    """Configuration arguments for the reasoning pipeline."""
    
    def __init__(self, use_finetuning=False, use_test_dataset=False, base_dir=None, output_dir=None, 
                 model_predictions_dir=None, images_dir=None, dataset_path=None, gemini_model=None):
        """
        Initialize arguments with options for dataset and model type.
        
        Parameters:
        - use_finetuning: Whether to use the fine-tuned model predictions (True) or base model predictions (False)
        - use_test_dataset: Whether to use the test dataset (True) or validation dataset (False)
        - base_dir: Base directory for the project (defaults to current working directory)
        - output_dir: Output directory for results (defaults to base_dir/outputs)
        - model_predictions_dir: Directory containing model predictions (defaults to output_dir/val-base-predictions)
        - images_dir: Directory containing images (auto-determined based on dataset if not provided)
        - dataset_path: Path to dataset CSV file (auto-determined based on dataset if not provided)
        - gemini_model: Gemini model to use (defaults to gemini-2.5-flash-preview-04-17)
        """
        self.use_finetuning = use_finetuning
        self.use_test_dataset = use_test_dataset
        
        # Set base directory
        self.base_dir = base_dir or os.getcwd()
        
        # Set output directory
        self.output_dir = output_dir or os.path.join(self.base_dir, "outputs")
        
        # Set model predictions directory
        self.model_predictions_dir = model_predictions_dir or os.path.join(self.output_dir, "val-base-predictions")
        
        # Set dataset-specific configurations
        if self.use_test_dataset:
            self.dataset_name = "test"
            self.dataset_path = dataset_path or os.path.join(self.output_dir, "test_dataset.csv")
            self.images_dir = images_dir or os.path.join(self.base_dir, "2025_dataset", "test", "images_test")
            self.prediction_prefix = "aggregated_test_predictions_"
        else:
            self.dataset_name = "validation"
            self.dataset_path = dataset_path or os.path.join(self.output_dir, "val_dataset.csv")
            self.images_dir = images_dir or os.path.join(self.base_dir, "2025_dataset", "valid", "images_valid")
            self.prediction_prefix = "aggregated_predictions_"
        
        self.model_type = "finetuned" if self.use_finetuning else "base"
        
        self.gemini_model = gemini_model or "gemini-2.5-flash-preview-04-17"
        
        print(f"\nConfiguration initialized:")
        print(f"- Base directory: {self.base_dir}")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Using {'test' if self.use_test_dataset else 'validation'} dataset")
        print(f"- Looking for {self.model_type} model predictions")
        print(f"- Dataset path: {self.dataset_path}")
        print(f"- Images directory: {self.images_dir}")
        print(f"- Model predictions directory: {self.model_predictions_dir}")
        print(f"- Prediction file prefix: {self.prediction_prefix}")
        print(f"- Gemini model: {self.gemini_model}")


class DataLoader:
    """Handles loading and processing of model predictions and validation data."""
    
    @staticmethod
    def get_latest_aggregated_files(args):
        """Get the latest aggregated prediction files for each model."""
        pattern = os.path.join(args.model_predictions_dir, f"{args.prediction_prefix}*_{args.model_type}_*.csv")
        
        agg_files = glob.glob(pattern)
        
        if len(agg_files) == 0:
            return []
        
        latest_files = {}
        
        for file_path in agg_files:
            file_name = os.path.basename(file_path)
            
            parts = file_name.split(f"_{args.model_type}_")
            if len(parts) != 2:
                print(f"Warning: Unexpected filename format: {file_name}")
                continue
            
            model_part = parts[0].replace(args.prediction_prefix, "")
            model_name = model_part
            
            timestamps = re.findall(r'(\d+)', parts[1])
            if len(timestamps) < 2:
                print(f"Warning: Could not find timestamps in {file_name}")
                continue
            
            timestamp = int(timestamps[1])
            
            if model_name not in latest_files or timestamp > latest_files[model_name]['timestamp']:
                latest_files[model_name] = {
                    'file_path': file_path,
                    'timestamp': timestamp
                }
        
        return [info['file_path'] for _, info in latest_files.items()]
    
    @staticmethod
    def load_all_model_predictions(args):
        """Load all model predictions from aggregated files."""
        latest_files = DataLoader.get_latest_aggregated_files(args)
        
        if not latest_files:
            print("No aggregated prediction files found. Cannot proceed.")
            return {}
        
        model_predictions = {}
        
        for file_path in latest_files:
            file_name = os.path.basename(file_path)
            
            parts = file_name.split(f"_{args.model_type}_")
            if len(parts) != 2:
                print(f"Warning: Unexpected filename format: {file_name}")
                continue
                
            model_name = parts[0].replace(args.prediction_prefix, "")
            
            try:
                df = pd.read_csv(file_path)
                df['model_name'] = model_name
                model_predictions[model_name] = df
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return model_predictions

    @staticmethod
    def load_validation_dataset(args):
        """Load the validation dataset."""
        val_df = pd.read_csv(args.dataset_path)
        
        val_df = DataLoader.process_validation_dataset(val_df)
        
        encounter_question_data = defaultdict(lambda: {
            'images': [],
            'data': None
        })
        
        for _, row in val_df.iterrows():
            encounter_id = row['encounter_id']
            base_qid = row['base_qid']
            key = (encounter_id, base_qid)
            
            if 'image_path' in row and row['image_path']:
                image_filename = os.path.basename(row['image_path'])
                image_path = os.path.join(args.images_dir, image_filename)
                encounter_question_data[key]['images'].append(image_path)
            elif 'image_id' in row and row['image_id']:
                image_path = os.path.join(args.images_dir, row['image_id'])
                encounter_question_data[key]['images'].append(image_path)
            
            if encounter_question_data[key]['data'] is None:
                encounter_question_data[key]['data'] = row.to_dict()
        
        grouped_data = []
        for (encounter_id, base_qid), data in encounter_question_data.items():
            entry = data['data'].copy()
            entry['all_images'] = data['images']
            entry['encounter_id'] = encounter_id
            entry['base_qid'] = base_qid
            grouped_data.append(entry)
        
        return pd.DataFrame(grouped_data)
    
    @staticmethod
    def safe_convert_options(options_str):
        """Safely convert a string representation of a list to an actual list."""
        if not isinstance(options_str, str):
            return options_str
            
        try:
            return ast.literal_eval(options_str)
        except (SyntaxError, ValueError):
            if options_str.startswith('[') and options_str.endswith(']'):
                return [opt.strip().strip("'\"") for opt in options_str[1:-1].split(',')]
            elif ',' in options_str:
                return [opt.strip() for opt in options_str.split(',')]
            else:
                return [options_str]
    
    @staticmethod
    def process_validation_dataset(val_df):
        """Process and clean the validation dataset."""
        if 'options_en' in val_df.columns:
            val_df['options_en'] = val_df['options_en'].apply(DataLoader.safe_convert_options)
            
            def clean_options(options):
                if not isinstance(options, list):
                    return options
                    
                cleaned_options = []
                for opt in options:
                    if isinstance(opt, str):
                        cleaned_opt = opt.strip("'\" ").replace(" (please specify)", "")
                        cleaned_options.append(cleaned_opt)
                    else:
                        cleaned_options.append(str(opt).strip("'\" "))
                return cleaned_options
                
            val_df['options_en_cleaned'] = val_df['options_en'].apply(clean_options)
        
        if 'question_text' in val_df.columns:
            val_df['question_text_cleaned'] = val_df['question_text'].apply(
                lambda q: q.replace(" Please specify which affected area for each selection.", "") 
                          if isinstance(q, str) and "Please specify which affected area for each selection" in q 
                          else q
            )
            
            val_df['question_text_cleaned'] = val_df['question_text_cleaned'].apply(
                lambda q: re.sub(r'^\d+\s+', '', q) if isinstance(q, str) else q
            )
        
        if 'base_qid' not in val_df.columns and 'qid' in val_df.columns:
            val_df['base_qid'] = val_df['qid'].apply(
                lambda q: q.split('-')[0] if isinstance(q, str) and '-' in q else q
            )
        
        return val_df


class DataProcessor:
    """Handles data processing for query creation."""
    
    @staticmethod
    def create_query_context(row, args=None):
        """Create query context from validation data similar to the inference process."""
        question = row.get('question_text_cleaned', row.get('question_text', 'What do you see in this image?'))
        
        metadata = ""
        if 'question_type_en' in row:
            metadata += f"Type: {row['question_type_en']}"
            
        if 'question_category_en' in row:
            metadata += f", Category: {row['question_category_en']}"
        
        query_title = row.get('query_title_en', '')
        query_content = row.get('query_content_en', '')
        
        clinical_context = ""
        if query_title or query_content:
            clinical_context += "Background Clinical Information (to help with your analysis):\n"
            if query_title:
                clinical_context += f"{query_title}\n"
            if query_content:
                clinical_context += f"{query_content}\n"
        
        options = row.get('options_en_cleaned', row.get('options_en', ['Yes', 'No', 'Not mentioned']))
        if isinstance(options, list):
            options_text = ", ".join(options)
        else:
            options_text = str(options)
        
        query_text = (f"MAIN QUESTION TO ANSWER: {question}\n"
                     f"Question Metadata: {metadata}\n"
                     f"{clinical_context}"
                     f"Available Options (choose from these): {options_text}")
        
        return query_text


class AgenticRAGData:
    """Manages combined data for agentic reasoning."""
    
    def __init__(self, all_models_df, validation_df):
        self.all_models_df = all_models_df
        self.validation_df = validation_df
        
        self.model_predictions = {}
        for (encounter_id, base_qid), group in all_models_df.groupby(['encounter_id', 'base_qid']):
            self.model_predictions[(encounter_id, base_qid)] = group
        
        self.validation_data = {}
        for _, row in validation_df.iterrows():
            self.validation_data[(row['encounter_id'], row['base_qid'])] = row
    
    def get_combined_data(self, encounter_id, base_qid):
        """Retrieve combined data for a specific encounter and question."""
        model_preds = self.model_predictions.get((encounter_id, base_qid), None)
        
        val_data = self.validation_data.get((encounter_id, base_qid), None)
        
        if model_preds is None:
            print(f"No model predictions found for encounter {encounter_id}, question {base_qid}")
            return None
            
        if val_data is None:
            print(f"No validation data found for encounter {encounter_id}, question {base_qid}")
            return None
        
        if 'query_context' not in val_data:
            val_data['query_context'] = DataProcessor.create_query_context(val_data)
        
        model_predictions_dict = {}
        for _, row in model_preds.iterrows():
            model_name = row['model_name']
            
            model_predictions_dict[model_name] = self._process_model_predictions(row)
        
        return {
            'encounter_id': encounter_id,
            'base_qid': base_qid,
            'query_context': val_data['query_context'],
            'images': val_data.get('all_images', []),
            'options': val_data.get('options_en_cleaned', val_data.get('options_en', [])),
            'question_type': val_data.get('question_type_en', ''),
            'question_category': val_data.get('question_category_en', ''),
            'model_predictions': model_predictions_dict
        }
    
    def _process_model_predictions(self, row):
        """Process model predictions from row data."""
        
        return {
            'model_prediction': row.get('combined_prediction', '')
        }
    
    def get_all_encounter_question_pairs(self):
        """Return a list of all unique encounter_id, base_qid pairs."""
        return list(self.validation_data.keys())
    
    def get_sample_data(self, n=5):
        """Get a sample of combined data for n random encounter-question pairs."""
        import random
        
        all_pairs = self.get_all_encounter_question_pairs()
        sample_pairs = random.sample(all_pairs, min(n, len(all_pairs)))
        
        return [self.get_combined_data(encounter_id, base_qid) for encounter_id, base_qid in sample_pairs]


class AnalysisService:
    """Service for analyzing medical images and clinical context using Gemini."""
    
    def __init__(self, api_key=None, args=None):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("API_KEY")
        
        self.client = genai.Client(api_key=api_key)
        self.args = args
    
    def extract_dermatological_analysis(self, sample_data):
        """
        Extract structured analysis of images for an encounter.
        
        Args:
            sample_data: Dictionary containing encounter data with images
            
        Returns:
            Dictionary with structured dermatological analysis
        """
        encounter_id = sample_data['encounter_id']
        image_paths = sample_data['images']
        
        image_analyses = []
        
        structured_prompt = self._create_dermatology_prompt()
        
        for idx, img_path in enumerate(image_paths):
            analysis = self._analyze_single_image(
                img_path, 
                structured_prompt, 
                encounter_id, 
                idx, 
                len(image_paths)
            )
            image_analyses.append(analysis)
        
        aggregated_analysis = self._aggregate_analyses(image_analyses, encounter_id)
        
        return {
            "encounter_id": encounter_id,
            "image_count": len(image_paths),
            "individual_analyses": image_analyses,
            "aggregated_analysis": aggregated_analysis
        }
    
    def _create_dermatology_prompt(self):
        """Create the structured dermatology analysis prompt."""
        return """As dermatology specialist analyzing skin images, extract and structure all clinically relevant information from this dermatological image.

Organize your response in a JSON dictionary:

1. SIZE: Approximate dimensions of lesions/affected areas, size comparison (thumbnail, palm, larger), Relative size comparisons for multiple lesions
2. SITE_LOCATION: Visible body parts in the image, body areas showing lesions/abnormalities, Specific anatomical locations affected
3. SKIN_DESCRIPTION: Lesion morphology (flat, raised, depressed), Texture of affected areas, Surface characteristics (scales, crust, fluid), Appearance of lesion boundaries
4. LESION_COLOR: Predominant color(s) of affected areas, Color variations within lesions, Color comparison to normal skin, Color distribution patterns
5. LESION_COUNT: Number of distinct lesions/affected areas, Single vs multiple presentation, Distribution pattern if multiple, Any counting limitations
6. EXTENT: How widespread the condition appears, Localized vs widespread assessment, Approximate percentage of visible skin affected, Limitations in determining full extent
7. TEXTURE: Expected tactile qualities, Smooth vs rough assessment, Notable textural features, Texture consistency across affected areas
8. ONSET_INDICATORS: Visual clues about condition duration, Acute vs chronic presentation features, Healing/progression/chronicity signs, Note: precise timing cannot be determined from images
9. ITCH_INDICATORS: Scratch marks/excoriations/trauma signs, Features associated with itchy conditions, Pruritic vs non-pruritic visual indicators, Note: sensation cannot be directly observed
10. OVERALL_IMPRESSION: Brief description (1-2 sentences), Key diagnostic features, Potential diagnoses (2-3)

Be concise and use medical terminology where appropriate. If information for a section is cannot be determined, state "Cannot determine from image".
"""
    
    def _analyze_single_image(self, img_path, prompt, encounter_id, idx, total_images):
        """Analyze a single dermatological image."""
        try:
            image = Image.open(img_path)
            
            
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=[prompt, image]
            )
            
            analysis_text = response.text
            
            structured_analysis = self._parse_json_response(analysis_text)
            
            return {
                "image_index": idx + 1,
                "image_path": os.path.basename(img_path),
                "structured_analysis": structured_analysis
            }
            
        except Exception as e:
            print(f"Error analyzing image {img_path}: {str(e)}")
            return {
                "image_index": idx + 1,
                "image_path": os.path.basename(img_path),
                "error": str(e)
            }
    
    def _parse_json_response(self, text):
        """Parse JSON from LLM response."""
        cleaned_text = text
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1]
        if "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[0]
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse as JSON")
            return {"parse_error": "Could not parse as JSON", "raw_text": text}
    
    def _aggregate_analyses(self, image_analyses, encounter_id):
        """Aggregate structured analyses from multiple images."""
        valid_analyses = [a for a in image_analyses if "error" not in a and "structured_analysis" in a]
        
        if not valid_analyses:
            return {
                "error": "No valid analyses to aggregate",
                "message": "Unable to generate aggregated analysis due to errors in individual analyses."
            }
        
        if len(valid_analyses) == 1:
            return valid_analyses[0]["structured_analysis"]
        
        analysis_jsons = []
        for analysis in valid_analyses:
            analysis_json = json.dumps(analysis["structured_analysis"])
            analysis_jsons.append(f"Image {analysis['image_index']} ({analysis['image_path']}): {analysis_json}")
        
        aggregation_prompt = self._create_aggregation_prompt(analysis_jsons)
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=[aggregation_prompt]
            )
            
            aggregation_text = response.text
            
            aggregated_analysis = self._parse_json_response(aggregation_text)            
            
            return aggregated_analysis
            
        except Exception as e:
            print(f"Error creating aggregated analysis for encounter {encounter_id}: {str(e)}")
            return {
                "error": str(e),
                "aggregation_error": "Failed to generate aggregated analysis"
            }
    
    def _create_aggregation_prompt(self, analysis_jsons):
        """Create a prompt for aggregating multiple image analyses."""
        return f"""As dermatology specialist reviewing multiple skin image analyses for the same patient, combine these analyses and organize your response in a JSON dictionary:

1. SIZE: Approximate dimensions of lesions/affected areas, size comparison (thumbnail, palm, larger), Relative size comparisons for multiple lesions
2. SITE_LOCATION: Visible body parts in the image, body areas showing lesions/abnormalities, Specific anatomical locations affected
3. SKIN_DESCRIPTION: Lesion morphology (flat, raised, depressed), Texture of affected areas, Surface characteristics (scales, crust, fluid), Appearance of lesion boundaries
4. LESION_COLOR: Predominant color(s) of affected areas, Color variations within lesions, Color comparison to normal skin, Color distribution patterns
5. LESION_COUNT: Number of distinct lesions/affected areas, Single vs multiple presentation, Distribution pattern if multiple, Any counting limitations
6. EXTENT: How widespread the condition appears, Localized vs widespread assessment, Approximate percentage of visible skin affected, Limitations in determining full extent
7. TEXTURE: Expected tactile qualities, Smooth vs rough assessment, Notable textural features, Texture consistency across affected areas
8. ONSET_INDICATORS: Visual clues about condition duration, Acute vs chronic presentation features, Healing/progression/chronicity signs, Note: precise timing cannot be determined from images
9. ITCH_INDICATORS: Scratch marks/excoriations/trauma signs, Features associated with itchy conditions, Pruritic vs non-pruritic visual indicators, Note: sensation cannot be directly observed
10. OVERALL_IMPRESSION: Brief description (1-2 sentences), Key diagnostic features, Potential diagnoses (2-3)
    
{' '.join(analysis_jsons)}
"""
    
    def extract_clinical_context(self, sample_data):
        """
        Extract structured clinical information from an encounter's query context.
        
        Args:
            sample_data: Dictionary containing encounter data with query_context
            
        Returns:
            Dictionary with structured clinical information
        """
        encounter_id = sample_data['encounter_id']
        
        query_context = sample_data['query_context']
        
        clinical_text = self._extract_clinical_text(query_context)
        
        if not clinical_text:
            return {
                "encounter_id": encounter_id,
                "clinical_summary": "No clinical information available"
            }
        
        prompt = self._create_clinical_context_prompt(clinical_text)
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=[prompt]
            )
            
            return {
                "encounter_id": encounter_id,
                "raw_clinical_text": clinical_text,
                "structured_clinical_context": response.text
            }
                
        except Exception as e:
            print(f"Error extracting clinical context for encounter {encounter_id}: {str(e)}")
            return {
                "encounter_id": encounter_id,
                "raw_clinical_text": clinical_text,
                "error": str(e)
            }
    
    def _extract_clinical_text(self, query_context):
        """Extract clinical text from query context."""
        clinical_lines = []
        capturing = False
        for line in query_context.split('\n'):
            if "Background Clinical Information" in line:
                capturing = True
                continue
            elif "Available Options" in line:
                capturing = False
            elif capturing:
                clinical_lines.append(line)
        
        return "\n".join(clinical_lines).strip()
    
    def _create_clinical_context_prompt(self, clinical_text):
        """Create prompt for extracting structured clinical information."""
        return f"""You are a dermatology specialist analyzing patient information. 
Extract and structure all clinically relevant information from this patient description:

{clinical_text}

Organize your response in the following JSON structure:

1. DEMOGRAPHICS: Age, sex, and any other demographic data
2. SITE_LOCATION: Body parts affected by the condition as described in the text
3. SKIN_DESCRIPTION: Any mention of lesion morphology (flat, raised, depressed), texture, surface characteristics (scales, crust, fluid), appearance of lesion boundaries
4. LESION_COLOR: Any description of color(s) of affected areas, color variations, comparison to normal skin
5. LESION_COUNT: Any information about number of lesions, single vs multiple presentation, distribution pattern
6. EXTENT: How widespread the condition appears based on the description, localized vs widespread
7. TEXTURE: Any description of tactile qualities, smooth vs rough, notable textural features
8. ONSET_INDICATORS: Information about onset, duration, progression, or evolution of symptoms
9. ITCH_INDICATORS: Mentions of scratching, itchiness, or other sensory symptoms
10. OTHER_SYMPTOMS: Any additional symptoms mentioned (pain, burning, etc.)
11. TRIGGERS: Identified factors that worsen/improve the condition
12. HISTORY: Relevant past medical history or previous treatments
13. DIAGNOSTIC_CONSIDERATIONS: Any mentioned or suggested diagnoses in the text

Be concise and use medical terminology where appropriate. If information for a section is 
not available, indicate "Not mentioned".
"""
    
    def apply_reasoning_layer(self, encounter_id, base_qid, image_analysis, clinical_context, sample_data):
        """
        Apply a reasoning layer to determine the best answer(s) for a specific encounter-question pair.
        
        Args:
            encounter_id: The encounter ID
            base_qid: The question ID
            image_analysis: Structured image analysis for this encounter
            clinical_context: Structured clinical context for this encounter
            sample_data: Combined data for this encounter-question pair
        
        Returns:
            Dictionary with reasoning and final answer(s)
        """
        question_text = sample_data['query_context'].split("MAIN QUESTION TO ANSWER:")[1].split("\n")[0].strip()
        options = sample_data['options']
        question_type = sample_data['question_type']
        model_predictions = sample_data['model_predictions']
        
        model_prediction_text = self._format_model_predictions(model_predictions)
        
        prompt = self._create_reasoning_prompt(
            question_text, 
            question_type, 
            options, 
            image_analysis, 
            clinical_context, 
            model_prediction_text
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=[prompt]
            )
            
            reasoning_text = response.text
            
            reasoning_result = self._parse_json_response(reasoning_text)
            
            validated_answer = self._validate_answer(reasoning_result.get('answer', ''), options)
            reasoning_result['validated_answer'] = validated_answer
            
            return reasoning_result
            
        except Exception as e:
            print(f"Error applying reasoning layer for {encounter_id}, {base_qid}: {str(e)}")
            return {
                "reasoning": f"Error: {str(e)}",
                "answer": "Not mentioned",
                "validated_answer": "Not mentioned",
                "error": str(e)
            }
    
    def _format_model_predictions(self, model_predictions):
        """Format model predictions for the prompt."""
        model_prediction_text = ""
        for model_name, predictions in model_predictions.items():
            combined_pred = predictions.get('model_prediction', '')
            if isinstance(combined_pred, float) and pd.isna(combined_pred):
                combined_pred = "No prediction"
            model_prediction_text += f"- {model_name}: {combined_pred}\n"
        return model_prediction_text

    def _create_reasoning_prompt(self, question_text, question_type, options, image_analysis, clinical_context, model_prediction_text):
        """Create a prompt for the reasoning layer."""

        specialized_guidance = ""
        include_clinical_context = True

        multiple_answers_allowed = question_type in ["Site Location", "Size", "Skin Description"]

        if multiple_answers_allowed:
            task_description = """Based on all the evidence above, determine the most accurate answer(s) to the question. Your task is to:
    1. Analyze the evidence from the image analysis{0}
    2. Consider the model predictions, noting any consensus or disagreement, but maintain your critical judgment
    3. Provide a brief reasoning for your conclusion
    4. Select the final answer(s) from the available options

    If selecting multiple answers is appropriate, provide them in a comma-separated list. If no answer can be determined, select "Not mentioned".""".format(' and clinical context' if include_clinical_context else '')
        else:
            task_description = """Based on all the evidence above, determine the SINGLE most accurate answer to the question. Your task is to:
    1. Analyze the evidence from the image analysis{0}
    2. Consider the model predictions, noting any consensus or disagreement, but maintain your critical judgment
    3. Provide a brief reasoning for your conclusion
    4. Select ONLY ONE answer option that is most accurate

    For this question type, you must select ONLY ONE option as your answer. If no answer can be determined, select "Not mentioned".""".format(' and clinical context' if include_clinical_context else '')

        if question_type == "Size" and all(option in ", ".join(options) for option in ["size of thumb nail", "size of palm", "larger area"]):
            specialized_guidance = """
    SPECIALIZED GUIDANCE FOR SIZE ASSESSMENT:
    When answering this size-related question, interpret the options as follows:
    - "size of thumb nail": Individual lesions or affected areas approximately 1-2 cm in diameter
    - "size of palm": Affected areas larger than the size of a thumb nail and roughly the size of a palm (approximately 1% of body surface area), which may include multiple smaller lesions across a region
    - "larger area": Widespread involvement significantly larger than a palm, affecting a substantial portion(s) of the body

    IMPORTANT: For cases with multiple small lesions that are visible in the images, but without extensive widespread involvement across large body regions, "size of palm" is likely the most appropriate answer.

    Base your assessment PRIMARILY on the current state shown in the IMAGES and their analysis, not on descriptions of progression or potential future spread mentioned in the clinical context. Prioritize what you can directly observe in the image analysis over clinical descriptions.
    """
            include_clinical_context = False

        elif question_type == "Lesion Color" and "combination" in ", ".join(options):
            specialized_guidance = """
    SPECIALIZED GUIDANCE FOR LESION COLOR:
    When answering color-related questions, pay careful attention to whether there are multiple distinct colors present across the affected areas. "Combination" would be appropriate when different lesions display different colors (e.g., some lesions appear red while others appear white), or when individual lesions show mixed or varied coloration patterns.
    """

        base_prompt = f"""You are a medical expert analyzing dermatological images. Use the provided evidence to determine the most accurate answer(s) for the following question:

    QUESTION: {question_text}
    QUESTION TYPE: {question_type}
    OPTIONS: {", ".join(options)}

    IMAGE ANALYSIS:
    {json.dumps(image_analysis['aggregated_analysis'], indent=2)}
    """

        if include_clinical_context:
            base_prompt += f"""
    CLINICAL CONTEXT:
    {clinical_context['structured_clinical_context']}
    """
        else:
            base_prompt += """
    NOTE: For this question type, the analysis is based primarily on image evidence rather than clinical descriptions.
    """

        return base_prompt + f"""
    MODEL PREDICTIONS:
    {model_prediction_text}

    {specialized_guidance}

    IMPORTANT: While multiple model predictions are provided, be aware that these predictions can be inaccurate or inconsistent. Do not assume majority agreement equals correctness. Evaluate the evidence critically and independently from these predictions. Your job is to determine the correct answer based primarily on the image analysis, treating model predictions as secondary suggestions that may contain errors.

    {task_description}

    Format your response as a JSON object with these fields:
    1. "reasoning": Your step-by-step reasoning process
    2. "answer": Your final answer(s) as a single string or comma-separated list of options

    When providing your answer, strictly adhere to the available options and only select from them.
    """
    
    def _validate_answer(self, answer, options):
        """Validate the answer against available options."""
        answer = answer.lower()
        valid_answers = []
        
        if ',' in answer:
            answer_parts = [part.strip() for part in answer.split(',')]
            for part in answer_parts:
                for option in options:
                    if part == option.lower():
                        valid_answers.append(option)
        else:
            for option in options:
                if answer == option.lower():
                    valid_answers.append(option)
        
        if not valid_answers:
            if "not mentioned" in answer:
                valid_answers = ["Not mentioned"]
            else:
                valid_answers = ["Not mentioned"]
        
        return ", ".join(valid_answers)


class DermatologyPipeline:
    """Main pipeline for dermatology analysis with reasoning."""
    
    def __init__(self, analysis_service):
        self.analysis_service = analysis_service
    
    def process_single_encounter(self, agentic_data, encounter_id):
        """
        Process a single encounter with all its questions using the reasoning layer.
        
        Args:
            agentic_data: AgenticRAGData instance containing all encounter data
            encounter_id: The specific encounter ID to process
            
        Returns:
            Dictionary with all questions processed with reasoning for this encounter
        """
        
        all_pairs = agentic_data.get_all_encounter_question_pairs()
        encounter_pairs = [pair for pair in all_pairs if pair[0] == encounter_id]
        
        if not encounter_pairs:
            print(f"No data found for encounter {encounter_id}")
            return None
        
        
        encounter_results = {encounter_id: {}}
        
        sample_data = agentic_data.get_combined_data(encounter_pairs[0][0], encounter_pairs[0][1])
        image_analysis = self.analysis_service.extract_dermatological_analysis(sample_data)
        
        clinical_context = self.analysis_service.extract_clinical_context(sample_data)
        
        for i, (encounter_id, base_qid) in enumerate(encounter_pairs):
            
            sample_data = agentic_data.get_combined_data(encounter_id, base_qid)
            if not sample_data:
                print(f"Warning: No data found for {encounter_id}, {base_qid}")
                continue
            
            reasoning_result = self.analysis_service.apply_reasoning_layer(
                encounter_id,
                base_qid,
                image_analysis,
                clinical_context,
                sample_data
            )
            
            encounter_results[encounter_id][base_qid] = {
                "query_context": sample_data['query_context'],
                "options": sample_data['options'],
                "model_predictions": sample_data['model_predictions'],
                "reasoning_result": reasoning_result,
                "final_answer": reasoning_result.get('validated_answer', 'Not mentioned')
            }
            
        output_file = os.path.join(self.analysis_service.args.output_dir, f"reasoning_results_{encounter_id}.json")
    
        with open(output_file, 'w') as f:
            json.dump(encounter_results, f, indent=2)
        
        return encounter_results
    
    def format_results_for_evaluation(self, encounter_results, output_file):
        """Format results for official evaluation."""
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
        for encounter_id, questions in encounter_results.items():
            encounter_base_qids = set(questions.keys())
            if not required_base_qids.issubset(encounter_base_qids):
                print(f"Skipping encounter {encounter_id} - missing required questions")
                continue
            
            pred_entry = {'encounter_id': encounter_id}
            
            for base_qid, question_data in questions.items():
                if base_qid not in qid_variants:
                    continue
                
                final_answer = question_data['final_answer']
                options = question_data['options']
                
                not_mentioned_index = self._find_not_mentioned_index(options)
                
                self._process_answers(
                    pred_entry, 
                    base_qid, 
                    final_answer, 
                    options, 
                    qid_variants, 
                    not_mentioned_index
                )
            
            formatted_predictions.append(pred_entry)
        
        with open(output_file, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)
        
        return formatted_predictions
    
    def _find_not_mentioned_index(self, options):
        """Find the index of 'Not mentioned' in options."""
        for i, opt in enumerate(options):
            if opt.lower() == "not mentioned":
                return i
        return len(options) - 1
    
    def _process_answers(self, pred_entry, base_qid, final_answer, options, qid_variants, not_mentioned_index):
        """Process answers and add to prediction entry."""
        if ',' in final_answer:
            answer_parts = [part.strip() for part in final_answer.split(',')]
            answer_indices = []
            
            for part in answer_parts:
                found = False
                for i, opt in enumerate(options):
                    if part.lower() == opt.lower():
                        answer_indices.append(i)
                        found = True
                        break
                
                if not found:
                    answer_indices.append(not_mentioned_index)
            
            available_variants = qid_variants[base_qid]
            
            for i, idx in enumerate(answer_indices):
                if i < len(available_variants):
                    pred_entry[available_variants[i]] = idx
            
            for i in range(len(answer_indices), len(available_variants)):
                pred_entry[available_variants[i]] = not_mentioned_index
            
        else:
            answer_index = not_mentioned_index
            
            for i, opt in enumerate(options):
                if final_answer.lower() == opt.lower():
                    answer_index = i
                    break
            
            pred_entry[qid_variants[base_qid][0]] = answer_index
            
            if len(qid_variants[base_qid]) > 1:
                for i in range(1, len(qid_variants[base_qid])):
                    pred_entry[qid_variants[base_qid][i]] = not_mentioned_index


def run_all_encounters_pipeline(args=None):
    """Run the pipeline for all available encounters and combine the results."""
    if args is None:
        args = Args(use_finetuning=False, use_test_dataset=False)
    
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    dataset_df = DataLoader.load_validation_dataset(args)
    agentic_data = AgenticRAGData(all_models_df, dataset_df)
    
    all_pairs = agentic_data.get_all_encounter_question_pairs()
    unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
    print(f"Found {len(unique_encounter_ids)} unique encounters to process")
    
    analysis_service = AnalysisService(args=args)
    
    pipeline = DermatologyPipeline(analysis_service)
    
    all_encounter_results = {}
    for i, encounter_id in enumerate(unique_encounter_ids):
        print(f"Processing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
        encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
        if encounter_results:
            all_encounter_results.update(encounter_results)
        
        if (i+1) % 5 == 0 or (i+1) == len(unique_encounter_ids):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            intermediate_output_file = os.path.join(
                args.output_dir, 
                f"intermediate_results_{i+1}_of_{len(unique_encounter_ids)}_{timestamp}.json"
            )
            with open(intermediate_output_file, 'w') as f:
                json.dump(all_encounter_results, f, indent=2)
            print(f"Saved intermediate results after processing {i+1} encounters")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset_name}_data_cvqa_sys_reasoned_all_{timestamp}.json"
    )
    
    formatted_predictions = pipeline.format_results_for_evaluation(all_encounter_results, output_file)
    
    print(f"Processed {len(formatted_predictions)} encounters successfully")
    return formatted_predictions


def run_single_encounter_pipeline(encounter_id, args=None):
    """Run the pipeline for a single encounter."""
    if args is None:
        args = Args(use_finetuning=False, use_test_dataset=False)
    
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    
    dataset_df = DataLoader.load_validation_dataset(args)
    agentic_data = AgenticRAGData(all_models_df, dataset_df)
    
    analysis_service = AnalysisService(args=args)
    
    pipeline = DermatologyPipeline(analysis_service)
    encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
    
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset_name}_data_cvqa_sys_reasoned_{encounter_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    formatted_predictions = pipeline.format_results_for_evaluation(encounter_results, output_file)
    
    return formatted_predictions


def main():
    """Main function to run the reasoning pipeline."""
    print("Reasoning-based Medical Analysis Pipeline")
    print("=" * 50)
    
    # Initialize arguments
    args = Args(use_finetuning=False, use_test_dataset=False)
    
    # For testing a single encounter
    # encounter_id = "ENC00852"
    # formatted_predictions = run_single_encounter_pipeline(encounter_id, args)
    # print(f"Processed encounter {encounter_id} with {len(formatted_predictions)} prediction entries")
    
    # To run all encounters
    formatted_predictions = run_all_encounters_pipeline(args)
    print(f"Total complete encounters processed: {len(formatted_predictions)}")
    
    return 0


@dataclass
class ReasoningConfig:
    """Configuration for the Reasoning Pipeline."""
    
    # Model and dataset configuration
    use_finetuning: bool = False
    use_test_dataset: bool = False
    gemini_model: str = "gemini-2.5-flash-preview-04-17"
    
    # Directory paths
    base_dir: Optional[str] = None
    output_dir: Optional[str] = None
    model_predictions_dir: Optional[str] = None
    images_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    
    # API configuration
    api_key: Optional[str] = None
    
    # Processing options
    save_intermediate_results: bool = True
    intermediate_save_frequency: int = 5
    
    def to_reasoning_args(self) -> Args:
        """Convert to Args format."""
        # Pass all configuration parameters directly to Args constructor
        args = Args(
            use_finetuning=self.use_finetuning,
            use_test_dataset=self.use_test_dataset,
            base_dir=self.base_dir,
            output_dir=self.output_dir,
            model_predictions_dir=self.model_predictions_dir,
            images_dir=self.images_dir,
            dataset_path=self.dataset_path,
            gemini_model=self.gemini_model
        )
        
        return args


class ReasoningPipeline:
    """
    Main wrapper class for the reasoning-based medical analysis pipeline.
    
    This class provides a clean, parameterizable interface for running medical image
    analysis with structured reasoning using Gemini models.
    """
    
    def __init__(self, config: Optional[ReasoningConfig] = None):
        """
        Initialize the reasoning pipeline.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or ReasoningConfig()
        self.args = self.config.to_reasoning_args()
        self.analysis_service = None
        self.agentic_data = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the pipeline components."""
        if self._initialized:
            return
            
        # Initialize analysis service
        self.analysis_service = AnalysisService(
            api_key=self.config.api_key,
            args=self.args
        )
        
        # Load data
        model_predictions_dict = DataLoader.load_all_model_predictions(self.args)
        
        if not model_predictions_dict:
            raise ValueError("No model predictions found. Please check your configuration.")
            
        all_models_df = self._concat_model_predictions(model_predictions_dict)
        dataset_df = DataLoader.load_validation_dataset(self.args)
        
        self.agentic_data = AgenticRAGData(all_models_df, dataset_df)
        self._initialized = True
        
    def _concat_model_predictions(self, model_predictions_dict: Dict) -> Any:
        """Safely concatenate model predictions."""
        if not model_predictions_dict:
            raise ValueError("No model predictions to concatenate")
            
        return pd.concat(model_predictions_dict.values(), ignore_index=True)
        
    def process_single_encounter(self, encounter_id: str) -> Dict[str, Any]:
        """
        Process a single encounter with reasoning analysis.
        
        Args:
            encounter_id: The encounter ID to process
            
        Returns:
            Dictionary containing the processed results
        """
        if not self._initialized:
            self.initialize()
            
        pipeline = DermatologyPipeline(self.analysis_service)
        encounter_results = pipeline.process_single_encounter(self.agentic_data, encounter_id)
        
        if not encounter_results:
            raise ValueError(f"No results generated for encounter {encounter_id}")
            
        return encounter_results
        
    def process_all_encounters(self) -> List[Dict[str, Any]]:
        """
        Process all available encounters with reasoning analysis.
        
        Returns:
            List of formatted predictions ready for evaluation
        """
        if not self._initialized:
            self.initialize()
            
        all_pairs = self.agentic_data.get_all_encounter_question_pairs()
        unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
        
        print(f"Found {len(unique_encounter_ids)} unique encounters to process")
        
        pipeline = DermatologyPipeline(self.analysis_service)
        all_encounter_results = {}
        
        for i, encounter_id in enumerate(unique_encounter_ids):
            print(f"Processing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
            
            encounter_results = pipeline.process_single_encounter(self.agentic_data, encounter_id)
            if encounter_results:
                all_encounter_results.update(encounter_results)
            
            # Save intermediate results if configured
            if (self.config.save_intermediate_results and 
                ((i+1) % self.config.intermediate_save_frequency == 0 or (i+1) == len(unique_encounter_ids))):
                self._save_intermediate_results(all_encounter_results, i+1, len(unique_encounter_ids))
        
        # Format and save final results
        return self._format_and_save_final_results(all_encounter_results, unique_encounter_ids)
    
    def _save_intermediate_results(self, results: Dict, current: int, total: int) -> None:
        """Save intermediate results during processing."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        intermediate_output_file = os.path.join(
            self.args.output_dir, 
            f"intermediate_results_{current}_of_{total}_{timestamp}.json"
        )
        with open(intermediate_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved intermediate results after processing {current} encounters")
    
    def _format_and_save_final_results(self, all_encounter_results: Dict, unique_encounter_ids: List) -> List[Dict[str, Any]]:
        """Format and save final results."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(
            self.args.output_dir, 
            f"{self.args.dataset_name}_data_cvqa_sys_reasoned_all_{timestamp}.json"
        )
        
        pipeline = DermatologyPipeline(self.analysis_service)
        formatted_predictions = pipeline.format_results_for_evaluation(all_encounter_results, output_file)
        
        print(f"Processed {len(formatted_predictions)} encounters successfully")
        return formatted_predictions


if __name__ == "__main__":
    import sys
    sys.exit(main())
