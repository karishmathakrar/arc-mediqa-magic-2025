"""
Data Preprocessing Module for Medical Image Fine-tuning Pipeline
Extracts and processes raw dataset files into batch files for training.
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
from collections import Counter, defaultdict
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


class DataPreprocessor:
    """Handles comprehensive data preprocessing for medical image datasets."""
    
    def __init__(self, config):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object containing paths and settings
        """
        self.config = config
    
    def safe_convert_options(self, options_str):
        """
        Safely convert a string representation of a list to an actual list.
        
        Args:
            options_str: String representation of options list
            
        Returns:
            List of options
        """
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
    
    def _check_batch_files_exist(self, use_combined_dataset=False, data_type="train"):
        """
        Check if batch files already exist for the current configuration.
        
        Args:
            use_combined_dataset: Whether using combined dataset
            data_type: Type of data ("train", "val", "test")
            
        Returns:
            Boolean indicating if batch files exist
        """
        if data_type == "test":
            check_dir = self.config.PROCESSED_TEST_DATA_DIR
            batch_prefix = "test_batch_"
        elif data_type == "val":
            check_dir = self.config.PROCESSED_VAL_DATA_DIR
            batch_prefix = "val_batch_"
        elif use_combined_dataset:
            check_dir = self.config.PROCESSED_COMBINED_DATA_DIR
            batch_prefix = "batch_"
        else:
            check_dir = self.config.PROCESSED_TRAIN_DATA_DIR
            batch_prefix = "batch_"
        
        if not os.path.exists(check_dir):
            return False
        
        batch_files = [f for f in os.listdir(check_dir) 
                      if f.startswith(batch_prefix) and f.endswith(".pkl")]
        
        return len(batch_files) > 0
    
    def prepare_dataset(self, mode="train"):
        """
        Create a dataset for either training, validation, or test data.
        
        Args:
            mode: Either "train", "val", or "test" to specify which dataset to prepare
            
        Returns:
            DataFrame containing the processed dataset
        """
        print(f"Preparing {mode} dataset...")
        
        if mode == "train":
            json_path = self.config.TRAIN_JSON_PATH
            cvqa_path = self.config.TRAIN_CVQA_PATH
            images_dir = self.config.TRAIN_IMAGES_DIR
            output_filename = "train_dataset_processed.csv"
        elif mode == "val":
            json_path = self.config.VAL_JSON_PATH
            cvqa_path = self.config.VAL_CVQA_PATH
            images_dir = self.config.VAL_IMAGES_DIR
            output_filename = "val_dataset.csv"
        elif mode == "test":
            return self._prepare_test_dataset()
        else:
            raise ValueError("Mode must be either 'train', 'val', or 'test'")
        
        # Load questions definitions
        with open(self.config.QUESTIONS_PATH, 'r') as f:
            questions = json.load(f)
            
        questions_df = pd.json_normalize(questions)[
            ["qid", "question_en", "options_en", "question_type_en", "question_category_en"]
        ]
        
        # Load input data
        input_df = pd.read_json(json_path)
        query_info_df = input_df[
            ["encounter_id", "image_ids", "query_title_en", "query_content_en", "author_id"]
        ]
        
        # Load CVQA data
        with open(cvqa_path, 'r') as f:
            cvqa_data = json.load(f)
        cvqa_df = pd.json_normalize(cvqa_data)
        
        # Reshape CVQA data
        cvqa_long = cvqa_df.melt(
            id_vars=["encounter_id"], 
            var_name="qid", 
            value_name="answer_index"
        )
        
        cvqa_long = cvqa_long[cvqa_long["qid"] != "encounter_id"]
        cvqa_merged = cvqa_long.merge(questions_df, on="qid", how="left")
        
        def get_answer_text(row):
            try:
                return row["options_en"][row["answer_index"]]
            except (IndexError, TypeError):
                return None
        
        cvqa_merged["answer_text"] = cvqa_merged.apply(get_answer_text, axis=1)
        final_df = cvqa_merged.merge(query_info_df, on="encounter_id", how="left")
        final_df['base_qid'] = final_df['qid'].str.extract(r'(CQID\d+)')
        
        # Group by family and get valid answers
        grouped_by_family = final_df.groupby(['encounter_id', 'base_qid']).agg({
            'qid': list,
            'question_en': list,
            'answer_text': list,
            'answer_index': list,
            'image_ids': 'first',
            'options_en': 'first',
            'question_type_en': 'first',
            'question_category_en': 'first',
            'query_title_en': 'first',
            'query_content_en': 'first',
            'author_id': 'first'
        }).reset_index()
        
        def get_valid_answers(row):
            """Extract all valid answers, with special handling for 'Not mentioned'."""
            answers = row['answer_text']
            answer_indices = row['answer_index']

            if all(ans == "Not mentioned" for ans in answers):
                return [["Not mentioned"], [answer_indices[0]]]

            valid_answers = []
            valid_indices = []

            for i, ans in enumerate(answers):
                if ans != "Not mentioned":
                    if isinstance(ans, str):
                        cleaned_ans = ans.strip("'\" ").replace(" (please specify)", "")
                        if cleaned_ans not in valid_answers:
                            valid_answers.append(cleaned_ans)
                            valid_indices.append(answer_indices[i])
                    else:
                        str_ans = str(ans).strip("'\" ")
                        if str_ans not in valid_answers:
                            valid_answers.append(str_ans)
                            valid_indices.append(answer_indices[i])

            return [valid_answers, valid_indices]
        
        grouped_by_family[['valid_answers', 'valid_indices']] = grouped_by_family.apply(
            lambda row: pd.Series(get_valid_answers(row)), axis=1
        )
        
        # Create dataset rows
        dataset_rows = []
        
        for _, row in tqdm(grouped_by_family.iterrows(), desc=f"Creating {mode} dataset"):
            encounter_id = row['encounter_id']
            base_qid = row['base_qid']
            valid_answers = row['valid_answers']
            valid_indices = row['valid_indices']
            image_ids = row['image_ids']
            question_text = row['question_en'][0]
            query_title = row['query_title_en']
            query_content = row['query_content_en']
            author_id = row['author_id']
            options_en = row['options_en']
            question_type_en = row['question_type_en']
            question_category_en = row['question_category_en']
            
            for img_id in image_ids:
                img_path = os.path.join(images_dir, img_id)
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_id} not found at {img_path}")
                    continue
                    
                dataset_rows.append({
                    'encounter_id': encounter_id,
                    'base_qid': base_qid,
                    'image_id': img_id,
                    'image_path': img_path,
                    'valid_answers': valid_answers,
                    'valid_indices': valid_indices,
                    'question_text': question_text,
                    'query_title_en': query_title,
                    'query_content_en': query_content,
                    'author_id': author_id,
                    'options_en': options_en,
                    'question_type_en': question_type_en, 
                    'question_category_en': question_category_en,
                    'is_multi_label': len(valid_answers) > 1
                })
        
        dataset = pd.DataFrame(dataset_rows)
        dataset.to_csv(os.path.join(self.config.OUTPUT_DIR, output_filename), index=False)
        
        print(f"{mode.capitalize()} dataset created with {len(dataset)} entries")
        return dataset
    
    def _prepare_test_dataset(self):
        """
        Create a dataset for test data without ground truth answers.
        Similar structure to prepare_dataset but adapted for test set.
        """
        print("Preparing test dataset...")
        
        json_path = self.config.TEST_JSON_PATH
        images_dir = self.config.TEST_IMAGES_DIR
        output_filename = "test_dataset.csv"
        
        # Load questions definitions
        with open(self.config.QUESTIONS_PATH, 'r') as f:
            questions = json.load(f)
            
        questions_df = pd.json_normalize(questions)[
            ["qid", "question_en", "options_en", "question_type_en", "question_category_en"]
        ]
        
        # Load input data
        input_df = pd.read_json(json_path)
        query_info_df = input_df[
            ["encounter_id", "image_ids", "query_title_en", "query_content_en", "author_id"]
        ]
        
        # Get base question IDs
        base_qids = sorted(list(set([q.split('-')[0] if '-' in q else q for q in questions_df['qid']])))
        
        # Create question representatives for each base QID
        question_representatives = {}
        for base_qid in base_qids:
            matching_questions = questions_df[questions_df['qid'].str.startswith(base_qid)]
            if not matching_questions.empty:
                question_representatives[base_qid] = matching_questions.iloc[0]
        
        dataset_rows = []
        
        for _, row in tqdm(query_info_df.iterrows(), desc="Creating test dataset"):
            encounter_id = row['encounter_id']
            image_ids = row['image_ids']
            query_title = row['query_title_en']
            query_content = row['query_content_en']
            author_id = row['author_id']
            
            for base_qid in base_qids:
                if base_qid in question_representatives:
                    question_info = question_representatives[base_qid]
                    question_text = question_info['question_en']
                    
                    options_en = question_info['options_en']
                    if isinstance(options_en, str):
                        options_en = self.safe_convert_options(options_en)
                    
                    # Clean options
                    cleaned_options = []
                    for opt in options_en:
                        if isinstance(opt, str):
                            cleaned_opt = opt.strip("'\" ").replace(" (please specify)", "")
                            cleaned_options.append(cleaned_opt)
                        else:
                            cleaned_options.append(str(opt).strip("'\" "))
                    
                    options_en = cleaned_options
                    question_type_en = question_info['question_type_en']
                    question_category_en = question_info['question_category_en']
                    
                    for img_id in image_ids:
                        img_path = os.path.join(images_dir, img_id)
                        
                        if not os.path.exists(img_path):
                            print(f"Warning: Image {img_id} not found at {img_path}")
                            continue
                            
                        dataset_rows.append({
                            'encounter_id': encounter_id,
                            'base_qid': base_qid,
                            'image_id': img_id,
                            'image_path': img_path,
                            'question_text': question_text,
                            'query_title_en': query_title,
                            'query_content_en': query_content,
                            'author_id': author_id,
                            'options_en': options_en,
                            'question_type_en': question_type_en, 
                            'question_category_en': question_category_en
                        })
        
        dataset = pd.DataFrame(dataset_rows)
        dataset.to_csv(os.path.join(self.config.OUTPUT_DIR, output_filename), index=False)
        
        print(f"Test dataset created with {len(dataset)} entries")
        return dataset
    
    def process_dataset_batch(self, batch_df, batch_idx, save_dir, images_dir, mode="train"):
        """
        Process a batch of data samples and save them as a pickle file.
        Works for both training and validation/inference datasets.
        
        Args:
            batch_df: DataFrame containing the batch to process
            batch_idx: Index of the batch (for naming the output file)
            save_dir: Directory to save the processed batch
            images_dir: Directory containing the images
            mode: Either "train" or "val"/"inference" to determine processing
            
        Returns:
            Number of successfully processed examples
        """
        os.makedirs(save_dir, exist_ok=True)
        batch_data = []
        
        file_prefix = "batch_" if mode == "train" else "val_batch_"
        if mode == "test":
            file_prefix = "test_batch_"
        
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx}"):
            try:
                image_id = row.get('image_id')
                if not image_id:
                    continue
                    
                if 'image_path' in row and os.path.exists(row['image_path']):
                    image_path = row['image_path']
                else:
                    image_path = os.path.join(images_dir, image_id)
                
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                # Verify image can be loaded
                try:
                    with Image.open(image_path) as img:
                        img.load()
                except Exception as e:
                    print(f"Corrupt or unreadable image at {image_path} — {e}")
                    continue
                
                # Process options
                if 'options_en' in row:
                    options = self.safe_convert_options(row['options_en'])
                    
                    cleaned_options = []
                    for opt in options:
                        if isinstance(opt, str):
                            cleaned_opt = opt.strip("'\" ").replace(" (please specify)", "")
                            cleaned_options.append(cleaned_opt)
                        else:
                            cleaned_options.append(str(opt).strip("'\" "))
                    options = cleaned_options
                else:
                    options = ["Yes", "No", "Not mentioned"]
                    
                options_text = ", ".join(options)
                
                # Create metadata
                metadata = ""
                if 'question_type_en' in row:
                    metadata += f"Type: {row['question_type_en']}"
                    
                if 'question_category_en' in row:
                    metadata += f", Category: {row['question_category_en']}"
                
                # Process question
                question = row.get('question_text', 'What do you see in this image?')
                
                if "Please specify which affected area for each selection" in question:
                    question = question.replace(" Please specify which affected area for each selection.", "")
                
                question = re.sub(r'^\d+\s+', '', question)
                
                # Create clinical context
                query_title = row.get('query_title_en', '')
                query_content = row.get('query_content_en', '')
                
                clinical_context = ""
                if query_title or query_content:
                    clinical_context += "Background Clinical Information (to help with your analysis):\n"
                    if query_title:
                        clinical_context += f"{query_title}\n"
                    if query_content:
                        clinical_context += f"{query_content}\n"

                # Create query text
                query_text = (f"MAIN QUESTION TO ANSWER: {question}\n"
                             f"Question Metadata: {metadata}\n"
                             f"{clinical_context}"
                             f"Available Options (choose from these): {options_text}")
                
                # Create data item
                data_item = {
                    "id": row.get('encounter_id', str(idx)),
                    "qid": row.get('base_qid', ''),
                    "query_text": query_text,
                    "image_path": image_path,
                    "question_type": row.get('question_type_en', ''),
                    "question_category": row.get('question_category_en', '')
                }
                
                # Add answer for training mode
                if mode == "train":
                    if 'valid_answers' in row and row['valid_answers']:
                        answers = row['valid_answers']
                        if isinstance(answers, list):
                            cleaned_answers = []
                            for ans in answers:
                                if isinstance(ans, str):
                                    cleaned_ans = ans.strip("'\" ")
                                    cleaned_ans = cleaned_ans.replace(" (please specify)", "")
                                    cleaned_answers.append(cleaned_ans)
                                else:
                                    cleaned_answers.append(str(ans).strip("'\" "))
                            
                            if len(cleaned_answers) > 1:
                                answer_text = ", ".join(cleaned_answers)
                            elif len(cleaned_answers) == 1:
                                answer_text = cleaned_answers[0]
                            else:
                                answer_text = "Not mentioned"
                        else:
                            if isinstance(answers, str):
                                answer_text = answers.strip("'\" ")
                                answer_text = answer_text.replace(" (please specify)", "")
                            else:
                                answer_text = str(answers).strip("'\" ")
                                
                        # Clean up answer text format
                        if isinstance(answer_text, str) and answer_text.startswith("[") and answer_text.endswith("]"):
                            clean_text = answer_text.strip("[]'")
                            parts = [part.strip() for part in clean_text.split("', '")]
                            answer_text = ", ".join(parts)
                    
                    elif 'multi_label' in row:
                        answer_text = row['multi_label']
                    else:
                        answer_text = "Not mentioned"
                    
                    data_item["answer_text"] = answer_text
                
                # Add image_id for inference modes
                if mode in ["val", "test"]:
                    data_item["image_id"] = os.path.basename(image_path)
                
                batch_data.append(data_item)
            
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                traceback.print_exc()
        
        # Save batch file
        batch_file = os.path.join(save_dir, f"{file_prefix}{batch_idx}.pkl")
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        
        return len(batch_data)
    
    def preprocess_dataset(self, df, batch_size=50, save_dir=None, images_dir=None, mode="train", use_combined_dataset=False):
        """
        Process an entire dataset in batches.
        Works for both training and validation/inference datasets.
        
        Args:
            df: DataFrame containing the dataset
            batch_size: Number of examples per batch
            save_dir: Directory to save processed batches
            images_dir: Directory containing images
            mode: Either "train", "val", or "test"
            use_combined_dataset: Whether using combined train+val dataset
            
        Returns:
            Total number of processed examples
        """
        total_processed = 0
        
        if save_dir is None:
            if mode == "train" and use_combined_dataset:
                save_dir = self.config.PROCESSED_COMBINED_DATA_DIR
            elif mode == "train":
                save_dir = self.config.PROCESSED_TRAIN_DATA_DIR
            elif mode == "test":
                save_dir = self.config.PROCESSED_TEST_DATA_DIR
            else:
                save_dir = self.config.PROCESSED_VAL_DATA_DIR
        
        if images_dir is None:
            if mode == "train":
                images_dir = self.config.TRAIN_IMAGES_DIR
            elif mode == "test":
                images_dir = self.config.TEST_IMAGES_DIR
            else:
                images_dir = self.config.VAL_IMAGES_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_idx = i // batch_size
            
            print(f"Processing batch {batch_idx+1}/{(len(df)-1)//batch_size + 1}")
            processed = self.process_dataset_batch(batch_df, batch_idx, save_dir, images_dir, mode=mode)
            total_processed += processed
            
            gc.collect()
            print(f"Processed {total_processed} examples so far")
        
        return total_processed
    
    def prepare_and_process_datasets(self, skip_data_prep=False, use_combined_dataset=False, test_mode=False, min_data_size=10):
        """
        Prepare and process datasets based on arguments.
        
        Args:
            skip_data_prep: Whether to skip data preparation
            use_combined_dataset: Whether to combine train and val datasets
            test_mode: Whether to run in test mode with small dataset
            min_data_size: Minimum data size for test mode
            
        Returns:
            Tuple containing (train_df, val_df)
        """
        train_df = None
        val_df = None
        
        # Check if batch files already exist
        batch_files_exist = self._check_batch_files_exist(use_combined_dataset)
        
        if skip_data_prep and batch_files_exist:
            print("Skipping data preparation - batch files already exist...")
            
            if os.path.exists(os.path.join(self.config.OUTPUT_DIR, "train_dataset_processed.csv")):
                train_df = pd.read_csv(os.path.join(self.config.OUTPUT_DIR, "train_dataset_processed.csv"))
                print(f"Loaded existing training dataset with {len(train_df)} samples")
            
            if os.path.exists(os.path.join(self.config.OUTPUT_DIR, "val_dataset.csv")):
                val_df = pd.read_csv(os.path.join(self.config.OUTPUT_DIR, "val_dataset.csv"))
                print(f"Loaded existing validation dataset with {len(val_df)} samples")
        
        else:
            if use_combined_dataset:
                print("Creating combined train+val dataset...")
                
                print("Preparing training dataset...")
                train_df = self.prepare_dataset(mode="train")
                
                print("Preparing validation dataset...")
                val_df = self.prepare_dataset(mode="val")
            
                train_df = pd.concat([train_df, val_df], ignore_index=True)
                val_df = None
                
                combined_file = os.path.join(self.config.OUTPUT_DIR, "combined_train_val_dataset.csv")
                train_df.to_csv(combined_file, index=False)
                print(f"Combined dataset saved to {combined_file} with {len(train_df)} samples")
                            
            else: 
                print("Preparing training dataset...")
                train_df = self.prepare_dataset(mode="train")
                
                print("Preparing validation dataset...")
                val_df = self.prepare_dataset(mode="val")
            
            if test_mode:
                print("Running in test mode with a small subset of data...")
                
                if train_df is not None:
                    test_size = min(min_data_size, len(train_df))
                    train_df = train_df.head(test_size)
                    print(f"Using {len(train_df)} training samples for testing")
                
                if val_df is not None:
                    test_size = min(min_data_size, len(val_df))
                    val_df = val_df.head(test_size)
                    print(f"Using {len(val_df)} validation samples for testing")
            
            # Clean up existing processed directories
            if use_combined_dataset:
                if os.path.exists(self.config.PROCESSED_COMBINED_DATA_DIR):
                    shutil.rmtree(self.config.PROCESSED_COMBINED_DATA_DIR)
                os.makedirs(self.config.PROCESSED_COMBINED_DATA_DIR, exist_ok=True)
            else:
                if os.path.exists(self.config.PROCESSED_TRAIN_DATA_DIR):
                    shutil.rmtree(self.config.PROCESSED_TRAIN_DATA_DIR)
                os.makedirs(self.config.PROCESSED_TRAIN_DATA_DIR, exist_ok=True)
            
                if os.path.exists(self.config.PROCESSED_VAL_DATA_DIR):
                    shutil.rmtree(self.config.PROCESSED_VAL_DATA_DIR)
                os.makedirs(self.config.PROCESSED_VAL_DATA_DIR, exist_ok=True)
            
            # Process datasets into batch files
            print("Processing datasets into batch files...")
            self.process_all_datasets(
                train_df=train_df,
                val_df=val_df,
                use_combined_dataset=use_combined_dataset,
                batch_size=100
            )
        
        return train_df, val_df
    
    def process_all_datasets(self, train_df=None, val_df=None, use_combined_dataset=False, batch_size=100):
        """
        Process all datasets (train, val, and optionally test) into batch files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            use_combined_dataset: Whether using combined dataset
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        # Process training data
        if train_df is not None:
            print("Processing training data...")
            total_train = self.preprocess_dataset(
                train_df, 
                batch_size=batch_size, 
                mode="train", 
                use_combined_dataset=use_combined_dataset
            )
            results['train'] = total_train
            print(f"Total processed training examples: {total_train}")
        
        # Process validation data
        if val_df is not None:
            print("Processing validation data...")
            total_val = self.preprocess_dataset(val_df, batch_size=batch_size, mode="val")
            results['val'] = total_val
            print(f"Total processed validation examples: {total_val}")
        
        return results
    
    def process_test_dataset(self, batch_size=100):
        """
        Process test dataset into batch files.
        
        Args:
            batch_size: Batch size for processing
            
        Returns:
            Number of processed test examples
        """
        print("Preparing and processing test dataset...")
        
        # Check if test batch files already exist
        test_batch_files_exist = self._check_batch_files_exist(use_combined_dataset=False, data_type="test")
        
        if test_batch_files_exist:
            print("Test batch files already exist, skipping processing...")
            # Count existing test samples
            batch_files = [f for f in os.listdir(self.config.PROCESSED_TEST_DATA_DIR) 
                          if f.startswith("test_batch_") and f.endswith(".pkl")]
            total_test = 0
            for batch_file in batch_files:
                with open(os.path.join(self.config.PROCESSED_TEST_DATA_DIR, batch_file), 'rb') as f:
                    batch_data = pickle.load(f)
                    total_test += len(batch_data)
            print(f"Found {total_test} existing processed test examples")
            return total_test
        
        # Clean up existing test processed directory
        if os.path.exists(self.config.PROCESSED_TEST_DATA_DIR):
            shutil.rmtree(self.config.PROCESSED_TEST_DATA_DIR)
        os.makedirs(self.config.PROCESSED_TEST_DATA_DIR, exist_ok=True)
        
        # Prepare test dataset
        test_df = self.prepare_dataset(mode="test")
        
        # Process test dataset
        total_test = self.preprocess_dataset(test_df, batch_size=batch_size, mode="test")
        print(f"Total processed test examples: {total_test}")
        
        return total_test
    
    def check_processed_sample(self, use_combined_dataset=False):
        """
        Check a sample of processed data and display it.
        
        Args:
            use_combined_dataset: Whether to check combined dataset
            
        Returns:
            Boolean indicating success
        """
        if use_combined_dataset:
            check_dir = self.config.PROCESSED_COMBINED_DATA_DIR
        else:
            check_dir = self.config.PROCESSED_TRAIN_DATA_DIR
        
        batch_file = os.path.join(check_dir, "batch_0.pkl")
        
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            print(f"\nSample of processed data from {check_dir}:")
            
            print("\nFirst example:")
            sample_data = batch_data[0]
            for key, value in sample_data.items():
                print(f"{key}: {value}")
            
            if len(batch_data) > 1:
                print("\nSecond example:")
                sample_data = batch_data[1]
                for key, value in sample_data.items():
                    print(f"{key}: {value}")
                
            return True
        except Exception as e:
            print(f"Error checking processed data: {e}")
            return False
    
    def inspect_processed_data(self, processed_dir=None, num_samples=3, data_type="val"):
        """
        Inspect the actual inputs being used during inference by examining the
        processed data files.
        
        Args:
            processed_dir: Directory containing processed data
            num_samples: Number of samples to display
            data_type: Type of data ("val", "test", "train")
        """
        if processed_dir is None:
            if data_type == "test":
                processed_dir = self.config.PROCESSED_TEST_DATA_DIR
            elif data_type == "train":
                processed_dir = self.config.PROCESSED_TRAIN_DATA_DIR
            else:
                processed_dir = self.config.PROCESSED_VAL_DATA_DIR
        
        # Determine batch file prefix
        if data_type == "test":
            prefix = "test_batch_"
        elif data_type == "train":
            prefix = "batch_"
        else:
            prefix = "val_batch_"
            
        batch_files = sorted([f for f in os.listdir(processed_dir) 
                             if f.startswith(prefix) and f.endswith(".pkl")])
        
        if not batch_files:
            print(f"No batch files found in {processed_dir}")
            return
        
        print(f"Found {len(batch_files)} batch files in {processed_dir}")
        
        with open(os.path.join(processed_dir, batch_files[0]), 'rb') as f:
            batch_data = pickle.load(f)
        
        print(f"Batch contains {len(batch_data)} samples")
        
        for i, sample in enumerate(batch_data[:num_samples]):
            print(f"\n{'='*80}")
            print(f"SAMPLE {i+1}")
            print(f"{'='*80}")
            
            print(f"ID: {sample['id']}")
            print(f"Question ID: {sample['qid']}")
            
            print("\nQUERY TEXT:")
            print("-" * 80)
            print(sample['query_text'])
            print("-" * 80)
            
            print(f"\nIMAGE PATH: {sample['image_path']}")
            
            if 'answer_text' in sample:
                print(f"EXPECTED ANSWER: {sample['answer_text']}")
            
            try:
                img = Image.open(sample['image_path'])
                print(f"Image dimensions: {img.size[0]}x{img.size[1]}, Format: {img.format}")
            except Exception as e:
                print(f"Could not open image: {e}")
            
            if i >= num_samples - 1:
                break
    
    def analyze_dataset_tokens(self, dataset_dir=None, processor=None, num_samples=None, data_type="val"):
        """
        Analyze token counts in the dataset without running training or inference.
        
        Args:
            dataset_dir: Path to the processed dataset directory
            processor: The processor from the model
            num_samples: Optional limit on number of samples to process
            data_type: Type of data ("val", "test", "train")
            
        Returns:
            Dictionary with token statistics
        """
        if processor is None:
            raise ValueError("Processor must be provided for tokenization")
        
        if dataset_dir is None:
            if data_type == "test":
                dataset_dir = self.config.PROCESSED_TEST_DATA_DIR
            elif data_type == "train":
                dataset_dir = self.config.PROCESSED_TRAIN_DATA_DIR
            else:
                dataset_dir = self.config.PROCESSED_VAL_DATA_DIR
        
        if not os.path.exists(dataset_dir):
            print(f"Directory not found: {dataset_dir}")
            return None
        
        token_stats = {
            "samples": [],
            "summary": {}
        }
        
        # Determine batch file prefix
        if data_type == "test":
            prefix = "test_batch_"
        elif data_type == "train":
            prefix = "batch_"
        else:
            prefix = "val_batch_"
        
        batch_files = sorted([f for f in os.listdir(dataset_dir) 
                             if f.startswith(prefix) and f.endswith(".pkl")])
        
        if not batch_files:
            print(f"No batch files found in {dataset_dir}")
            return None
        
        total_tokens = 0
        max_tokens = 0
        min_tokens = float('inf')
        sample_count = 0
        all_token_counts = []
        
        for batch_file in tqdm(batch_files, desc="Analyzing batches"):
            try:
                with open(os.path.join(dataset_dir, batch_file), 'rb') as f:
                    batch_data = pickle.load(f)
                
                for sample in tqdm(batch_data, desc=f"Analyzing {batch_file}", leave=False):
                    if not isinstance(sample, dict) or "query_text" not in sample:
                        print(f"Warning: Unexpected sample format in {batch_file}")
                        continue
                    
                    # Create system message
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
                                {"type": "text", "text": sample["query_text"]},
                                {"type": "image", "image": "IMAGE_PLACEHOLDER"},
                            ],
                        },
                    ]
                    
                    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
                    tokens = processor.tokenizer.encode(text)
                    token_count = len(tokens)
                    all_token_counts.append(token_count)
                    
                    total_tokens += token_count
                    max_tokens = max(max_tokens, token_count)
                    min_tokens = min(min_tokens, token_count)
                    
                    token_stats["samples"].append({
                        "id": sample.get("id", "unknown"),
                        "qid": sample.get("qid", "unknown"),
                        "image": os.path.basename(sample.get("image_path", "unknown")),
                        "token_count": token_count,
                        "text_length": len(sample["query_text"])
                    })
                    
                    sample_count += 1
                    if num_samples and sample_count >= num_samples:
                        break
                
                if num_samples and sample_count >= num_samples:
                    break
                    
            except Exception as e:
                print(f"Error processing batch {batch_file}: {e}")
                continue
        
        if sample_count == 0:
            print("No samples were successfully processed")
            return None
        
        token_stats["summary"] = {
            "total_samples": sample_count,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / sample_count,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "median_tokens": np.median(all_token_counts),
            "percentile_90": np.percentile(all_token_counts, 90),
            "percentile_99": np.percentile(all_token_counts, 99)
        }
        
        # Save analysis results
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{os.path.basename(dataset_dir)}_token_analysis.json")
        with open(output_path, "w") as f:
            json.dump(token_stats, f, indent=2)
        
        print("\nToken Usage Analysis:")
        print(f"Total samples analyzed: {sample_count}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average tokens per sample: {total_tokens/sample_count:.2f}")
        print(f"Median tokens per sample: {np.median(all_token_counts):.2f}")
        print(f"90th percentile: {np.percentile(all_token_counts, 90):.2f}")
        print(f"99th percentile: {np.percentile(all_token_counts, 99):.2f}")
        print(f"Max tokens in a sample: {max_tokens}")
        print(f"Min tokens in a sample: {min_tokens}")
        print(f"Percentage of 128K context window used (max): {(max_tokens/128000)*100:.2f}%")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        bin_count = min(50, len(set(all_token_counts)))
        n, bins, patches = plt.hist(all_token_counts, bins=bin_count, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.axvline(x=np.mean(all_token_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_token_counts):.1f}')
        plt.axvline(x=np.median(all_token_counts), color='green', linestyle='-', linewidth=2, label=f'Median: {np.median(all_token_counts):.1f}')
        plt.axvline(x=np.percentile(all_token_counts, 90), color='orange', linestyle='-.', linewidth=2, label=f'90th percentile: {np.percentile(all_token_counts, 90):.1f}')
        
        plt.title(f"Distribution of Token Counts (n={sample_count})", fontsize=16)
        plt.xlabel("Token Count", fontsize=14)
        plt.ylabel("Number of Samples", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        stats_text = (
            f"Min: {min_tokens}\n"
            f"Max: {max_tokens}\n"
            f"Mean: {np.mean(all_token_counts):.1f}\n"
            f"Median: {np.median(all_token_counts):.1f}\n"
            f"Std Dev: {np.std(all_token_counts):.1f}\n"
            f"90th %ile: {np.percentile(all_token_counts, 90):.1f}\n"
            f"% of 128K used: {(max_tokens/128000)*100:.2f}%"
        )
        
        plt.text(0.95, 0.95, stats_text, 
                 transform=plt.gca().transAxes, 
                 verticalalignment='top', 
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plt_path = os.path.join(self.config.OUTPUT_DIR, f"{os.path.basename(dataset_dir)}_token_distribution.png")
        plt.savefig(plt_path, dpi=300)
        plt.show()
        
        return token_stats
