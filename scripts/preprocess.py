#!/usr/bin/env python
"""
Dedicated preprocessing script for the MEDIQA project.
Handles data preparation for both training and inference.
"""
import os
import sys
import argparse
import logging
import json
import pandas as pd
import pickle
from tqdm.auto import tqdm
from PIL import Image

# Add parent directory to path to allow absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modules
from data.processor import MedicalDataProcessor
from config.config import *
from utils.utils import print_system_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess data for the MEDIQA project")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="processed_data",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "inference"],
        default="train",
        help="Preprocessing mode (train or inference)"
    )
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default=None,
        help="CSV file with data (will be generated if it doesn't exist)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of encounters to process"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess the data even if already processed"
    )
    
    return parser.parse_args()

def generate_csv(data_dir, csv_output, mode="train"):
    """
    Generate a unified CSV file from the raw JSON data.
    This function combines data from train.json, train_cvqa.json,
    and the questions definitions.
    
    Args:
        data_dir: Directory containing the dataset
        csv_output: Path to save the generated CSV
        mode: Either 'train' or 'inference'
        
    Returns:
        Generated DataFrame
    """
    logger.info(f"Generating CSV file from raw data in {data_dir}")
    
    # Load train.json
    train_json_path = os.path.join(data_dir, "train.json")
    train_df = pd.read_json(train_json_path)
    
    # Filter relevant columns
    train_df = train_df[[
        "encounter_id", "author_id", "image_ids", "responses", 
        "query_title_en", "query_content_en"
    ]]
    
    # Extract English responses
    train_df["responses_en"] = train_df["responses"].apply(
        lambda resp_list: [r["content_en"] for r in resp_list]
    )
    
    # Load CVQA data
    cvqa_path = os.path.join(data_dir, "train_cvqa.json")
    with open(cvqa_path, "r", encoding="utf-8") as f:
        cvqa_data = json.load(f)
    cvqa_df = pd.json_normalize(cvqa_data)
    
    # Melt to get one row per question
    cvqa_long = cvqa_df.melt(id_vars=["encounter_id"], 
                             var_name="qid", 
                             value_name="answer_index")
    
    # Filter out encounter_id rows
    cvqa_long = cvqa_long[cvqa_long["qid"] != "encounter_id"]
    
    # Load question definitions
    questions_path = os.path.join(data_dir, "closedquestions_definitions_imageclef2025.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions_df = pd.json_normalize(questions)[["qid", "question_en", "options_en", "question_type_en", "question_category_en"]]
    
    # Merge CVQA with questions
    cvqa_merged = cvqa_long.merge(questions_df, on="qid", how="left")
    
    # Get answer text
    def get_answer_text(row):
        try:
            return row["options_en"][row["answer_index"]]
        except (IndexError, TypeError):
            return None
    
    cvqa_merged["answer_text"] = cvqa_merged.apply(get_answer_text, axis=1)
    
    # Merge with train data
    final_df = cvqa_merged.merge(train_df, on="encounter_id", how="left")
    
    # Save to CSV
    final_df.to_csv(csv_output, index=False)
    logger.info(f"Saved merged dataframe to {csv_output}")
    
    # Create option maps for evaluation
    option_maps = {}
    for _, question in questions_df.iterrows():
        base_qid = question['qid'].split('-')[0]
        if base_qid not in option_maps:
            options = question['options_en']
            option_dict = {opt: i for i, opt in enumerate(options)}
            if "Not mentioned" not in option_dict:
                option_dict["Not mentioned"] = len(options)
            option_maps[base_qid] = option_dict
    
    # Save option maps
    option_maps_path = os.path.join(data_dir, "option_maps.json")
    with open(option_maps_path, 'w', encoding='utf-8') as f:
        json.dump(option_maps, f, indent=2)
    
    logger.info(f"Saved option mappings to {option_maps_path}")
    
    return final_df

def process_batch(batch_df, batch_idx, save_dir, data_dir, mode="train"):
    """
    Process a batch of data and save to disk.
    
    Args:
        batch_df: DataFrame containing the batch data
        batch_idx: Batch index
        save_dir: Directory to save processed data
        data_dir: Directory containing raw data
        mode: Either 'train' or 'inference'
        
    Returns:
        Number of processed examples
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_data = []
    
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx}"):
        try:
            # Only take the first image from the list
            if not row['image_ids'] or len(row['image_ids']) == 0:
                continue
            
            # Get just the first image path
            image_path = os.path.join(data_dir, "images_train", row['image_ids'][0])
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found at {image_path}")
                continue
            
            # Verify the image is valid
            try:
                with Image.open(image_path) as img:
                    img.load()
            except Exception as e:
                logger.warning(f"Corrupt or unreadable image at {image_path} — {e}")
                continue
            
            # Format options text
            options_text = ", ".join([f"{i+1}. {opt}" for i, opt in enumerate(row['options_en'])])
            
            # Create metadata string
            metadata = f"Type: {row.get('question_type_en', '')}, Category: {row.get('question_category_en', '')}"
            
            # Create the full query text with explicit instructions
            query_text = f"Question: Based on the image, {row['question_en']}\nQuestion Metadata: {metadata}\nOptions: {options_text}"
            
            # Add critical instruction for inference only
            if mode == "inference":
                query_text += "\n\nCRITICAL INSTRUCTION: Only respond with an option if it is **clearly and unambiguously** supported by the image. If the image is unclear, incomplete, or could fit multiple answers, respond with: 'Not mentioned'. You must respond with the **exact text** of one option below. No numbers, no explanation. Given the medical context, err on the side of caution."
            
            batch_data.append({
                "encounter_id": row['encounter_id'],
                "qid": row['qid'],
                "query_text": query_text,
                "image_path": image_path,
                "image_paths": [image_path],  # Keep both for compatibility
                "answer_text": row['answer_text'],
                "answer_index": row.get('answer_index', -1),
                "question_type": row.get('question_type_en', ''),
                "question_category": row.get('question_category_en', ''),
                "options": row['options_en']
            })
        
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
    
    batch_file = os.path.join(save_dir, f"batch_{batch_idx}.pkl")
    with open(batch_file, 'wb') as f:
        pickle.dump(batch_data, f)
    
    return len(batch_data)

def preprocess_dataset(csv_file, data_dir, output_dir, mode="train", limit=None, batch_size=5, reprocess=False):
    """
    Process the dataset by preprocessing the CSV file.
    
    Args:
        csv_file: Path to the CSV file with data
        data_dir: Directory containing the dataset
        output_dir: Directory to save processed data
        mode: Either 'train' or 'inference'
        limit: Limit the number of encounters to process
        batch_size: Number of encounters to process in each batch
        reprocess: Whether to reprocess existing data
        
    Returns:
        Tuple of (total_processed, output_dir)
    """
    # Set default CSV file path if not specified
    if csv_file is None:
        csv_file = os.path.join(data_dir, f"dataset_{mode}.csv")
    
    # Check if output directory exists and has data
    if os.path.exists(output_dir) and not reprocess:
        batch_files = [f for f in os.listdir(output_dir) if f.startswith("batch_") and f.endswith(".pkl")]
        if batch_files:
            logger.info(f"Using existing processed data from {output_dir}")
            # Count total processed examples
            total_processed = 0
            for batch_file in batch_files:
                with open(os.path.join(output_dir, batch_file), 'rb') as f:
                    batch_data = pickle.load(f)
                    total_processed += len(batch_data)
            return total_processed, output_dir
    
    # Clear output directory if reprocessing
    if reprocess and os.path.exists(output_dir):
        import shutil
        logger.info(f"Clearing existing processed data from {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the CSV file exists, create it if not
    if not os.path.exists(csv_file):
        logger.info(f"CSV file not found, generating from raw data: {csv_file}")
        generate_csv(data_dir, csv_file, mode)
    
    # Load the CSV data
    logger.info(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        logger.info(f"Limited dataset to {limit} samples")
    
    # Convert string representations of lists to actual lists
    df['image_ids'] = df['image_ids'].apply(eval)
    df['options_en'] = df['options_en'].apply(eval)
    if 'responses_en' in df.columns:
        df['responses_en'] = df['responses_en'].apply(eval)
    
    # Filter to essential columns
    df = df[['encounter_id', 'qid', 'question_en', 'options_en', 'answer_text', 
             'image_ids', 'question_type_en', 'question_category_en', 'answer_index']]
    
    logger.info(f"Filtered dataframe shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Process the data in batches
    batch_size = min(batch_size, len(df))
    total_processed = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_idx = i // batch_size
        
        logger.info(f"Processing batch {batch_idx+1}/{(len(df)-1)//batch_size + 1}")
        processed = process_batch(batch_df, batch_idx, output_dir, data_dir, mode)
        total_processed += processed
        
        logger.info(f"Processed {total_processed} examples so far")
    
    logger.info(f"Preprocessing complete. Processed {total_processed} examples.")
    return total_processed, output_dir

def main():
    """Main entry point for preprocessing."""
    args = parse_args()
    
    # Print system info
    print_system_info()
    
    output_dir = os.path.join(args.output_dir, args.mode)
    
    # Preprocess the dataset
    total_processed, output_dir = preprocess_dataset(
        args.csv_file,
        args.data_dir,
        output_dir,
        args.mode,
        args.limit,
        args.batch_size,
        args.reprocess
    )
    
    logger.info(f"Preprocessing complete. Processed {total_processed} examples.")
    logger.info(f"Processed data saved to {output_dir}")
    
    # Print a sample for debugging
    processor = MedicalDataProcessor(args.data_dir, args.mode)
    processor.print_example()

if __name__ == "__main__":
    main()