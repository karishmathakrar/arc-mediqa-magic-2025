"""
Medical Image Fine-tuning and Inference Pipeline
Fine-tunes vision-language models on medical imaging data and runs inference.
"""

import os
import gc
import re
import ast
import json
import shutil
import random
import pickle
import datetime
import traceback
import glob
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText, 
    BitsAndBytesConfig, 
    MllamaForConditionalGeneration, 
    AutoModelForVision2Seq, 
    Qwen2VLForConditionalGeneration, 
    Qwen2_5_VLForConditionalGeneration, 
    AutoTokenizer
)

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from tensorboard.backend.event_processing import event_accumulator

from dotenv import load_dotenv
import zipfile
import requests

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not available. Some Qwen functionality may be limited.")
    process_vision_info = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessor import DataPreprocessor


class Config:
    """Configuration class for the pipeline."""
    
    def __init__(self, 
                 model_name="Qwen2-VL-2B-Instruct",
                 base_dir=None,
                 output_dir=None,
                 hf_token=None,
                 setup_environment=True,
                 validate_paths=True):
        """
        Initialize configuration.
        
        Args:
            model_name: Name of the model to use
            base_dir: Base directory containing dataset (defaults to script directory)
            output_dir: Output directory for models and results (defaults to base_dir/outputs)
            hf_token: HuggingFace token (defaults to environment variable)
            setup_environment: Whether to setup environment variables
            validate_paths: Whether to validate that required paths exist
        """
        # Environment setup
        if setup_environment:
            self._setup_environment()
        
        # Available models
        self.AVAILABLE_MODELS = {
            "llama-3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "gemma-3-4b-it": "google/gemma-3-4b-it",
            "gemma-3-12b-it": "google/gemma-3-12b-it",
            "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct"
        }
        
        # Validate model selection
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")
        
        # Model selection
        self.SELECTED_MODEL = model_name
        
        # Directory setup
        if base_dir is None:
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        else:
            self.BASE_DIR = os.path.abspath(base_dir)
            
        self.DATASET_DIR = os.path.join(self.BASE_DIR, "2025_dataset")
        self.TRAIN_DIR = os.path.join(self.DATASET_DIR, "train")
        self.VAL_DIR = os.path.join(self.DATASET_DIR, "valid")
        self.TEST_DIR = os.path.join(self.DATASET_DIR, "test")
        
        self.TRAIN_IMAGES_DIR = os.path.join(self.TRAIN_DIR, "images_train")
        self.VAL_IMAGES_DIR = os.path.join(self.VAL_DIR, "images_valid")
        self.TEST_IMAGES_DIR = os.path.join(self.TEST_DIR, "images_test")
        
        if output_dir is None:
            self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "outputs")
        else:
            self.OUTPUT_DIR = os.path.abspath(output_dir)

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Data paths
        self.QUESTIONS_PATH = os.path.join(self.TRAIN_DIR, "closedquestions_definitions_imageclef2025.json")
        self.TRAIN_JSON_PATH = os.path.join(self.TRAIN_DIR, "train.json")
        self.VAL_JSON_PATH = os.path.join(self.VAL_DIR, "valid.json")
        self.TEST_JSON_PATH = os.path.join(self.TEST_DIR, "test.json")
        self.TRAIN_CVQA_PATH = os.path.join(self.TRAIN_DIR, "train_cvqa.json")
        self.VAL_CVQA_PATH = os.path.join(self.VAL_DIR, "valid_cvqa.json")
        
        # Model configuration
        self.MODEL_ID = self.AVAILABLE_MODELS[self.SELECTED_MODEL]
        self.MODEL_NAME = self.MODEL_ID.split('/')[-1]
        self.IS_LLAMA = "llama" in self.MODEL_ID.lower()
        self.IS_QWEN = "qwen" in self.MODEL_ID.lower()
        
        self.HF_TOKEN = hf_token or os.getenv("HF_TOKEN")
        
        # Output paths
        self.TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.MODEL_SAVE_DIRECTORY = os.path.join(
            self.OUTPUT_DIR, "finetuned-model", f"{self.MODEL_NAME}_{self.TIMESTAMP}"
        )
        os.makedirs(self.MODEL_SAVE_DIRECTORY, exist_ok=True)
        
        # Processed data directories
        self.PROCESSED_TRAIN_DATA_DIR = os.path.join(
            self.OUTPUT_DIR, f"processed_train_data-{self.SELECTED_MODEL}-V3"
        )
        self.PROCESSED_VAL_DATA_DIR = os.path.join(
            self.OUTPUT_DIR, f"processed_val_data-{self.SELECTED_MODEL}-V3"
        )
        self.PROCESSED_COMBINED_DATA_DIR = os.path.join(
            self.OUTPUT_DIR, f"processed_combined_data-{self.SELECTED_MODEL}-V3"
        )
        self.PROCESSED_TEST_DATA_DIR = os.path.join(
            self.OUTPUT_DIR, f"processed_test_data-{self.SELECTED_MODEL}-V3"
        )
        
        # Validate paths if requested
        if validate_paths:
            self.validate_paths()
    
    def _setup_environment(self):
        """Setup environment variables for the pipeline."""
        load_dotenv()
        if "TRANSFORMERS_CACHE" in os.environ:
            print(f"Removing existing TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
            del os.environ["TRANSFORMERS_CACHE"]
        
        os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
        print(f"HF_HOME: {os.getenv('HF_HOME')}")
    
    def validate_paths(self):
        """Validate that required data paths exist."""
        required_paths = [
            self.DATASET_DIR,
            self.TRAIN_DIR,
            self.VAL_DIR,
            self.QUESTIONS_PATH,
            self.TRAIN_JSON_PATH,
            self.VAL_JSON_PATH,
            self.TRAIN_CVQA_PATH,
            self.VAL_CVQA_PATH
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            raise FileNotFoundError(f"Required paths not found: {missing_paths}")
        
        return True


class ModelManager:
    """Handles model loading and configuration."""
    
    def __init__(self, config):
        self.config = config
    
    def get_model_config(self, torch_dtype=None):
        """Create standardized model configuration dictionary."""
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
        model_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"]
        )
        
        return model_kwargs
    
    def load_model_and_processor(self):
        """Load model and processor based on configuration."""
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            print("WARNING: GPU may not fully support bfloat16. Consider using float16 instead.")

        model_kwargs = self.get_model_config(torch_dtype=torch.bfloat16)

        if self.config.IS_LLAMA:
            model = MllamaForConditionalGeneration.from_pretrained(
                self.config.MODEL_ID, **model_kwargs
            )
            if hasattr(model, "tie_weights"):
                model.tie_weights()
        elif self.config.IS_QWEN:
            if "2.5" in self.config.MODEL_ID:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.MODEL_ID, **model_kwargs
                )
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.MODEL_ID, **model_kwargs
                )
        else:
            non_llama_kwargs = model_kwargs.copy()
            non_llama_kwargs["attn_implementation"] = "eager"
            model = AutoModelForImageTextToText.from_pretrained(
                self.config.MODEL_ID, **non_llama_kwargs
            )

        processor = AutoProcessor.from_pretrained(self.config.MODEL_ID, token=self.config.HF_TOKEN)

        print(f"Default chat template: {processor.tokenizer.chat_template}")
        print(f"Special tokens map: {processor.tokenizer.special_tokens_map}")
        
        return model, processor


class MedicalImageDataset(torch.utils.data.Dataset):
    """Custom dataset for medical image training."""
    
    def __init__(self, data_dir, processor):
        self.processor = processor
        self.examples = []
        
        for batch_file in sorted(os.listdir(data_dir)):
            if batch_file.startswith("batch_") and batch_file.endswith(".pkl"):
                with open(os.path.join(data_dir, batch_file), 'rb') as f:
                    batch_data = pickle.load(f)
                    self.examples.extend(batch_data)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        image = Image.open(example['image_path']).convert("RGB")
        
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
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example['query_text']},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example['answer_text']}],
            },
        ]
        
        return {"messages": messages}


class InferenceManager:
    """Handles model inference and prediction generation."""
    
    def __init__(self, config):
        self.config = config
        self.model_manager = ModelManager(config)
    
    def merge_lora_model(self, checkpoint_path=None, token=None, output_dir=None):
        """Merge LoRA weights into base model and save to output directory."""
        if checkpoint_path is None:
            checkpoint_pattern = os.path.join(
                self.config.OUTPUT_DIR, "finetuned-model", f"{self.config.MODEL_NAME}_*", "checkpoint-*"
            )
            checkpoint_dirs = glob.glob(checkpoint_pattern)
            
            if not checkpoint_dirs:
                raise FileNotFoundError(f"No checkpoints found matching pattern {checkpoint_pattern}")
                
            checkpoint_dirs = sorted(checkpoint_dirs,
                                    key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)),
                                    reverse=True)
            checkpoint_path = checkpoint_dirs[0]
            print(f"Using latest checkpoint: {checkpoint_path}")
            
        if output_dir is None:
            checkpoint_name = os.path.basename(checkpoint_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = os.path.join(
                self.config.OUTPUT_DIR, "merged", f"{self.config.MODEL_NAME}_{checkpoint_name}_{timestamp}"
            )
            
        model_kwargs = self.model_manager.get_model_config(torch_dtype=torch.bfloat16)
        
        if self.config.IS_LLAMA:
            model = MllamaForConditionalGeneration.from_pretrained(
                self.config.MODEL_ID,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token
            )
        elif self.config.IS_QWEN:
            if "2.5" in self.config.MODEL_ID:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.MODEL_ID,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                    token=token
                )
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.MODEL_ID,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                    token=token
                )
        else:
            non_llama_kwargs = model_kwargs.copy()
            non_llama_kwargs["attn_implementation"] = "eager"
            model = AutoModelForImageTextToText.from_pretrained(
                self.config.MODEL_ID,
                low_cpu_mem_usage=True,
                **non_llama_kwargs,
                token=token
            )
            
        print(f"Applying adapter from {checkpoint_path}...")
        peft_model = PeftModel.from_pretrained(model, checkpoint_path)
        print("Merging weights...")
        merged_model = peft_model.merge_and_unload()
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving merged model to {output_dir}...")
        merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
        
        processor = AutoProcessor.from_pretrained(self.config.MODEL_ID, token=token)
        processor.save_pretrained(output_dir)
        
        del model
        del peft_model
        del merged_model
        torch.cuda.empty_cache()
        print(f"Merged model saved to {output_dir}")
        return output_dir


class MedicalImageInference:
    """Handles inference for medical image analysis."""
    
    def __init__(self, model_path, token=None, adapter_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the inference class for medical image analysis.
        
        Parameters:
        - model_path: Path to the model (base model or merged model)
        - token: HF token for downloading models
        - adapter_path: Path to adapter weights (only used if not using merged model)
        - device: Computing device (cuda or cpu)
        """
        self.device = device
        print(f"Loading processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, token=token)

        base_kwargs = dict(
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            token=token
        )

        print(f"Loading model from {model_path}...")
        IS_LLAMA = "llama" in model_path.lower()
        IS_QWEN = "qwen" in model_path.lower()
        
        if IS_LLAMA and adapter_path:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token
            )
            print(f"Loading adapter from {adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        elif IS_QWEN and adapter_path:
            base_kwargs["attn_implementation"] = "flash_attention_2" if torch.cuda.is_available() else "eager"
            if "Qwen2.5-VL" in model_path:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **base_kwargs
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **base_kwargs
                )
            print(f"Loading adapter from {adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            if IS_LLAMA:
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    token=token
                )
            elif IS_QWEN:
                base_kwargs["attn_implementation"] = "flash_attention_2" if torch.cuda.is_available() else "eager"
                if "Qwen2.5-VL" in model_path:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        **base_kwargs
                    )
                else:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        **base_kwargs
                    )
            else:
                non_llama_kwargs = base_kwargs.copy()
                non_llama_kwargs["attn_implementation"] = "eager"
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    **non_llama_kwargs
                )
        self.model.eval()
        self.IS_QWEN = IS_QWEN
        print("Model loaded successfully")

    def predict(self, query_text, image_path, max_new_tokens=100):
        """Generate prediction for a single query and image."""
        try:
            image = Image.open(image_path).convert("RGB")

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

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_text},
                        {"type": "image", "image": image},
                    ],
                }
            ]

            if self.IS_QWEN and process_vision_info:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=self.processor.apply_chat_template(messages, tokenize=False),
                    images=image,
                    return_tensors="pt"
                )
            inputs = inputs.to(self.device)
                
            with torch.no_grad():
                generation_params = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 64
                }

                if not (self.IS_QWEN or "llama" in str(type(self.model)).lower()):
                    generation_params["disable_compile"] = True

                generated_ids = self.model.generate(
                    **inputs,
                    **generation_params
                )

            input_length = inputs.input_ids.shape[1]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            prediction = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            prediction = prediction.strip()

            # Clean up prediction text
            if prediction.startswith("assistant\n\n"):
                prediction = prediction[len("assistant\n\n"):]
            if prediction.startswith("assistant\n"):
                prediction = prediction[len("assistant\n"):]
            if prediction.startswith("system\n"):
                prediction = prediction[len("system\n"):]
            if prediction.startswith("model\n"):
                prediction = prediction[len("model\n"):]

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
    
    def batch_predict(self, processed_data_dir=None, output_file=None, max_samples=None):
        """Run inference on a batch of preprocessed data."""
        if processed_data_dir is None:
            processed_data_dir = os.path.join(os.getcwd(), "outputs", "processed_val_data")

        if output_file is None:
            output_file = os.path.join(os.getcwd(), "outputs", "predictions.csv")

        results = []
        sample_count = 0

        # Determine batch file prefix
        if "test" in processed_data_dir.lower():
            batch_prefix = "test_batch_"
        else:
            batch_prefix = "val_batch_"

        batch_files = sorted([f for f in os.listdir(processed_data_dir) 
                              if f.startswith(batch_prefix) and f.endswith(".pkl")])

        if not batch_files:
            print(f"Warning: No batch files found in {processed_data_dir} with prefix {batch_prefix}")
            return pd.DataFrame()

        for batch_file in tqdm(batch_files, desc="Processing batches"):
            with open(os.path.join(processed_data_dir, batch_file), 'rb') as f:
                batch_data = pickle.load(f)

            for sample in tqdm(batch_data, desc=f"Predicting {batch_file}", leave=False):
                prediction = self.predict(sample["query_text"], sample["image_path"])

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

        return results_df

    def aggregate_predictions(self, predictions_df, validation_df=None, test_df=None):
        """Aggregate predictions for each encounter and question ID."""
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

            if options_en is None and test_df is not None:
                matching_rows = test_df[(test_df['encounter_id'] == encounter_id) & 
                                        (test_df['base_qid'] == base_qid)]
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


class TrainingPipeline:
    """Handles the complete training pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.data_preprocessor = DataPreprocessor(config)
        self.model_manager = ModelManager(config)
        self.inference_manager = InferenceManager(config)
        
    def prepare_training_data(self, use_combined=False, test_mode=False, min_data_size=10, skip_data_prep=False):
        """Prepare training and validation datasets."""
        return self.data_preprocessor.prepare_and_process_datasets(
            skip_data_prep=skip_data_prep,
            use_combined_dataset=use_combined,
            test_mode=test_mode,
            min_data_size=min_data_size
        )
    
    def process_datasets_to_batches(self, train_df=None, val_df=None, use_combined=False, batch_size=100):
        """Process datasets into batch files for training."""
        return self.data_preprocessor.process_all_datasets(
            train_df=train_df,
            val_df=val_df,
            use_combined_dataset=use_combined,
            batch_size=batch_size
        )
    
    def process_test_dataset(self, batch_size=100):
        """Process test dataset into batch files."""
        return self.data_preprocessor.process_test_dataset(batch_size=batch_size)
    
    def inspect_processed_data(self, processed_dir=None, num_samples=3, data_type="val"):
        """Inspect processed data samples."""
        return self.data_preprocessor.inspect_processed_data(
            processed_dir=processed_dir,
            num_samples=num_samples,
            data_type=data_type
        )
    
    def analyze_dataset_tokens(self, dataset_dir=None, processor=None, num_samples=None, data_type="val"):
        """Analyze token usage in the dataset."""
        if processor is None:
            # Load processor for token analysis
            _, processor = self.model_manager.load_model_and_processor()
        
        return self.data_preprocessor.analyze_dataset_tokens(
            dataset_dir=dataset_dir,
            processor=processor,
            num_samples=num_samples,
            data_type=data_type
        )
    
    def create_collate_fn(self, processor, is_qwen, is_llama):
        """Create custom collate function for batching examples."""
        def collate_fn(examples):
            texts = []
            all_images = []
            
            for example in examples:
                text = processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                )
                
                if is_qwen and process_vision_info:
                    image_inputs, _ = process_vision_info(example["messages"])
                    image_input = image_inputs[0] if image_inputs else None
                else:
                    image_input = None
                    for msg in example["messages"]:
                        if msg["role"] == "user":
                            for content in msg["content"]:
                                if isinstance(content, dict) and content.get("type") == "image" and "image" in content:
                                    image_input = content["image"]
                                    break
                
                if image_input is None:
                    image_input = Image.new('RGB', (224, 224), color='black')
                    
                texts.append(text.strip())
                all_images.append(image_input)
            
            batch = processor(
                text=texts,
                images=all_images,
                return_tensors="pt", 
                padding=True
            )
            
            labels = batch["input_ids"].clone()
            
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            
            # Mask special tokens based on model type
            if is_llama:
                for token_id in processor.tokenizer.all_special_ids:
                    labels[labels == token_id] = -100
                    
                # Additional Llama-specific masking
                try:
                    if hasattr(processor, "image_token"):
                        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
                        labels[labels == image_token_id] = -100
                except:
                    pass
                    
                if hasattr(processor, "image_token_index"):
                    labels[labels == processor.image_token_index] = -100
                    
            elif is_qwen:
                # Qwen-specific image tokens
                image_tokens = [151652, 151653, 151655]
                for token_id in image_tokens:
                    labels[labels == token_id] = -100
                
                for token_id in processor.tokenizer.all_special_ids:
                    labels[labels == token_id] = -100
            else:
                # Other models
                try:
                    for special_token in ["boi_token", "eoi_token"]:
                        if special_token in processor.tokenizer.special_tokens_map:
                            token_id = processor.tokenizer.convert_tokens_to_ids(
                                processor.tokenizer.special_tokens_map[special_token]
                            )
                            labels[labels == token_id] = -100
                    
                    labels[labels == 262144] = -100
                except Exception as e:
                    print(f"Warning: Could not mask tokens: {e}")
            
            batch["labels"] = labels
            return batch
        
        return collate_fn
    
    def train(self, use_combined=False, test_mode=False):
        """Run the complete training pipeline."""
        print("Starting training pipeline...")
        
        # Prepare data
        train_df, val_df = self.prepare_training_data(use_combined, test_mode)
        
        # Load model and processor
        model, processor = self.model_manager.load_model_and_processor()
        
        # Create dataset
        if use_combined:
            dataset = MedicalImageDataset(self.config.PROCESSED_COMBINED_DATA_DIR, processor)
        else:
            dataset = MedicalImageDataset(self.config.PROCESSED_TRAIN_DATA_DIR, processor)
            
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("ERROR: Dataset is empty! Check data loading process.")
            return None
        
        # Create collate function
        collate_fn = self.create_collate_fn(processor, self.config.IS_QWEN, self.config.IS_LLAMA)
        
        # Configure LoRA
        if self.config.IS_LLAMA or self.config.IS_QWEN:
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = "all-linear"

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=None if self.config.IS_LLAMA else ["lm_head", "embed_tokens"]
        )
        
        # Configure training
        training_args = SFTConfig(
            output_dir=self.config.MODEL_SAVE_DIRECTORY,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            learning_rate=1e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            push_to_hub=False,
            report_to="tensorboard",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=collate_fn,
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Training completed. Model saved to {self.config.MODEL_SAVE_DIRECTORY}")
        return trainer

    def run_inference(self, use_finetuning=True, test_mode=False, max_samples=None):
        """Run inference on validation or test data."""
        print("Starting inference pipeline...")
        
        # Determine model path
        if use_finetuning:
            # Find latest checkpoint and merge
            checkpoint_pattern = os.path.join(
                self.config.OUTPUT_DIR, "finetuned-model", f"{self.config.MODEL_NAME}_*", "checkpoint-*"
            )
            checkpoint_dirs = glob.glob(checkpoint_pattern)
            
            if not checkpoint_dirs:
                raise FileNotFoundError(f"No checkpoints found for model {self.config.MODEL_NAME}")
            
            checkpoint_dirs = sorted(checkpoint_dirs, 
                                    key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)), 
                                    reverse=True)
            
            checkpoint_path = checkpoint_dirs[0]
            print(f"Creating merged model from latest checkpoint: {checkpoint_path}")
            
            checkpoint_name = os.path.basename(checkpoint_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            merged_dir = os.path.join(
                self.config.OUTPUT_DIR, "merged", f"{self.config.MODEL_NAME}_{checkpoint_name}_{timestamp}"
            )
            
            model_path = self.inference_manager.merge_lora_model(
                checkpoint_path=checkpoint_path,
                token=self.config.HF_TOKEN,
                output_dir=merged_dir
            )
            print(f"Using merged model at {model_path}")
            adapter_path = None
        else:
            model_path = self.config.MODEL_ID
            adapter_path = None
            print(f"Using BASE MODEL ONLY at {model_path} (no fine-tuning applied)")
        
        # Initialize inference
        inference = MedicalImageInference(
            model_path=model_path,
            token=self.config.HF_TOKEN,
            adapter_path=adapter_path
        )
        
        # Determine data directory and output file
        if test_mode:
            data_dir = self.config.PROCESSED_TEST_DATA_DIR
            output_file = os.path.join(
                self.config.OUTPUT_DIR, 
                f"test_predictions_{self.config.MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        else:
            data_dir = self.config.PROCESSED_VAL_DATA_DIR
            output_file = os.path.join(
                self.config.OUTPUT_DIR, 
                f"val_predictions_{self.config.MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        
        print(f"Running inference on {data_dir}")
        predictions_df = inference.batch_predict(
            processed_data_dir=data_dir,
            output_file=output_file,
            max_samples=max_samples
        )
        
        # Aggregate predictions
        print("Aggregating predictions...")
        if test_mode:
            # Load test dataset for aggregation
            test_dataset_path = os.path.join(self.config.OUTPUT_DIR, "test_dataset.csv")
            if os.path.exists(test_dataset_path):
                test_dataset = pd.read_csv(test_dataset_path)
                aggregated_df = inference.aggregate_predictions(predictions_df, test_df=test_dataset)
            else:
                aggregated_df = inference.aggregate_predictions(predictions_df)
        else:
            # Load validation dataset for aggregation
            val_dataset_path = os.path.join(self.config.OUTPUT_DIR, "val_dataset.csv")
            if os.path.exists(val_dataset_path):
                val_dataset = pd.read_csv(val_dataset_path)
                aggregated_df = inference.aggregate_predictions(predictions_df, validation_df=val_dataset)
            else:
                aggregated_df = inference.aggregate_predictions(predictions_df)
        
        # Save aggregated results
        aggregated_file = os.path.join(
            self.config.OUTPUT_DIR,
            f"{'test' if test_mode else 'val'}_aggregated_predictions_{self.config.MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        aggregated_df.to_csv(aggregated_file, index=False)
        
        # Format for evaluation
        formatted_file = os.path.join(
            self.config.OUTPUT_DIR,
            f"{'test' if test_mode else 'val'}_data_cvqa_sys_{self.config.MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        formatted_predictions = self.format_predictions_for_evaluation(aggregated_df, formatted_file)
        
        print(f"Inference complete!")
        print(f"Raw predictions: {output_file}")
        print(f"Aggregated predictions: {aggregated_file}")
        print(f"Formatted for evaluation: {formatted_file}")
        
        return predictions_df, aggregated_df, formatted_predictions
    
    def format_predictions_for_evaluation(self, aggregated_df, output_file):
        """Format predictions for official evaluation."""
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
                
                options = self.data_preprocessor.safe_convert_options(row.get('options_en', []))
                
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


class FineTuningPipeline:
    """
    Main wrapper class for easy import and use of the medical vision pipeline.
    
    Example usage:
        from latest_code.finetuning_pipeline import FineTuningPipeline
        
        # Initialize pipeline
        pipeline = FineTuningPipeline(
            model_name="Qwen2-VL-2B-Instruct",
            base_dir="/path/to/data",
            output_dir="/path/to/outputs"
        )
        
        # Train a model
        trainer = pipeline.train(use_combined=False, test_mode=False)
        
        # Run inference
        predictions = pipeline.run_inference(use_finetuning=True)
    """
    
    def __init__(self, 
                 model_name="Qwen2-VL-2B-Instruct",
                 base_dir=None,
                 output_dir=None,
                 hf_token=None,
                 setup_environment=True,
                 validate_paths=True):
        """
        Initialize the Medical Vision Pipeline.
        
        Args:
            model_name: Name of the model to use (default: "Qwen2-VL-2B-Instruct")
            base_dir: Base directory containing dataset (defaults to current directory)
            output_dir: Output directory for models and results (defaults to base_dir/outputs)
            hf_token: HuggingFace token (defaults to environment variable)
            setup_environment: Whether to setup environment variables (default: True)
            validate_paths: Whether to validate that required paths exist (default: True)
        """
        self.config = Config(
            model_name=model_name,
            base_dir=base_dir,
            output_dir=output_dir,
            hf_token=hf_token,
            setup_environment=setup_environment,
            validate_paths=validate_paths
        )
        self.training_pipeline = TrainingPipeline(self.config)
        
        print(f"FineTuningPipeline initialized with model: {model_name}")
        print(f"Base directory: {self.config.BASE_DIR}")
        print(f"Output directory: {self.config.OUTPUT_DIR}")
    
    def get_available_models(self):
        """Get list of available models."""
        return list(self.config.AVAILABLE_MODELS.keys())
    
    def prepare_data(self, use_combined=False, test_mode=False, min_data_size=10):
        """
        Prepare training and validation datasets.
        
        Args:
            use_combined: Whether to combine train and validation data
            test_mode: Whether to use a small subset for testing
            min_data_size: Minimum data size when in test mode
            
        Returns:
            Tuple of (train_df, val_df)
        """
        return self.training_pipeline.prepare_training_data(
            use_combined=use_combined,
            test_mode=test_mode,
            min_data_size=min_data_size
        )
    
    def train(self, use_combined=False, test_mode=False):
        """
        Train the model.
        
        Args:
            use_combined: Whether to combine train and validation data for training
            test_mode: Whether to run in test mode with small dataset
            
        Returns:
            Trainer object if successful, None if failed
        """
        return self.training_pipeline.train(
            use_combined=use_combined,
            test_mode=test_mode
        )
    
    def run_inference(self, use_finetuning=True, test_mode=False, max_samples=None):
        """
        Run inference on validation or test data.
        
        Args:
            use_finetuning: Whether to use fine-tuned model or base model
            test_mode: Whether to run on test data instead of validation
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Tuple of (predictions_df, aggregated_df, formatted_predictions)
        """
        return self.training_pipeline.run_inference(
            use_finetuning=use_finetuning,
            test_mode=test_mode,
            max_samples=max_samples
        )
    
    def predict_single(self, image_path, query_text, model_path=None, max_new_tokens=100):
        """
        Run prediction on a single image and query.
        
        Args:
            image_path: Path to the image file
            query_text: Query text for the image
            model_path: Path to model (defaults to latest fine-tuned model)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Prediction string
        """
        if model_path is None:
            # Find latest checkpoint and merge
            checkpoint_pattern = os.path.join(
                self.config.OUTPUT_DIR, "finetuned-model", f"{self.config.MODEL_NAME}_*", "checkpoint-*"
            )
            checkpoint_dirs = glob.glob(checkpoint_pattern)
            
            if checkpoint_dirs:
                checkpoint_dirs = sorted(checkpoint_dirs, 
                                        key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)), 
                                        reverse=True)
                checkpoint_path = checkpoint_dirs[0]
                
                # Create merged model
                model_path = self.training_pipeline.inference_manager.merge_lora_model(
                    checkpoint_path=checkpoint_path,
                    token=self.config.HF_TOKEN
                )
            else:
                # Use base model
                model_path = self.config.MODEL_ID
        
        # Initialize inference
        inference = MedicalImageInference(
            model_path=model_path,
            token=self.config.HF_TOKEN
        )
        
        return inference.predict(query_text, image_path, max_new_tokens)
    
    def evaluate_predictions(self, prediction_file, reference_file=None):
        """
        Evaluate predictions against reference.
        
        Args:
            prediction_file: Path to prediction file
            reference_file: Path to reference file (defaults to validation data)
            
        Returns:
            Evaluation results
        """
        if reference_file is None:
            reference_file = self.config.VAL_CVQA_PATH
        
        # Import evaluation script
        from .evaluation_script import MedicalEvaluator
        evaluator = MedicalEvaluator()
        
        return evaluator.evaluate_predictions(reference_file, prediction_file)
    
    def get_config(self):
        """Get the current configuration."""
        return self.config
    
    def print_system_info(self):
        """Print system information."""
        print(f"Python version: {os.sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


def main():
    """Main function to run the training and inference pipeline."""
    print("Medical Image Fine-tuning and Inference Pipeline")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Print system information
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Selected model: {config.SELECTED_MODEL}")
    print(f"Model ID: {config.MODEL_ID}")
    print(f"Is Llama model: {config.IS_LLAMA}")
    print(f"Is Qwen model: {config.IS_QWEN}")
    print(f"Model output directory: {config.MODEL_SAVE_DIRECTORY}")
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    
    # Configuration options - modify as needed
    RUN_TRAINING = True  # Set to False to skip training
    RUN_INFERENCE = True  # Set to False to skip inference
    USE_COMBINED_DATASET = False  # Set to True to combine train and val for training
    TEST_MODE = False  # Set to True for quick testing with small dataset
    USE_FINETUNED_FOR_INFERENCE = True  # Set to False to use base model for inference
    RUN_ON_TEST_DATA = False  # Set to True to run inference on test data instead of validation
    MAX_INFERENCE_SAMPLES = None  # Set to a number to limit inference samples for testing
    
    # Run training if enabled
    if RUN_TRAINING:
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        
        trainer = pipeline.train(use_combined=USE_COMBINED_DATASET, test_mode=TEST_MODE)
        
        if trainer is not None:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed!")
            if not RUN_INFERENCE:
                return 1
    else:
        print("\nSkipping training...")
    
    # Run inference if enabled
    if RUN_INFERENCE:
        print("\n" + "="*50)
        print("STARTING INFERENCE")
        print("="*50)
        
        try:
            predictions_df, aggregated_df, formatted_predictions = pipeline.run_inference(
                use_finetuning=USE_FINETUNED_FOR_INFERENCE,
                test_mode=RUN_ON_TEST_DATA,
                max_samples=MAX_INFERENCE_SAMPLES
            )
            
            print(f"\nInference completed successfully!")
            print(f"Processed {len(predictions_df)} individual predictions")
            print(f"Generated {len(aggregated_df)} aggregated results")
            print(f"Formatted {len(formatted_predictions)} encounters for evaluation")
            
        except Exception as e:
            print(f"\nInference failed: {str(e)}")
            return 1
    else:
        print("\nSkipping inference...")
    
    print("\nPipeline completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
