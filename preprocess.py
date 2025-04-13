"""
Data preprocessing script for the MEDIQA project.
Processes the original dataset into a format suitable for training and inference.
"""
import os
import sys
import json
import pickle
import argparse
import logging
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd

import config
from utils import load_json_file, is_valid_image, save_pickle

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
        "--mode", 
        type=str, 
        choices=["train", "inference"],
        default="train",
        help="Processing mode (train or inference)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of examples to process"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=500,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess even if processed files exist"
    )
    parser.add_argument(
        "--use_single_image", 
        action="store_true",
        default=True,
        help="Use only the first image for each encounter"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=config.DATASET_DIR,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=config.PROCESSED_DIR,
        help="Base directory to save processed data (mode will be appended)"
    )
    
    return parser.parse_args()

def prepare_dataset(data_dir):
    """Prepare the dataset by combining data from different sources."""
    logger.info("Combining data from different sources...")

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
    cvqa_data = load_json_file(cvqa_path)
    cvqa_df = pd.json_normalize(cvqa_data)
    
    # Melt to get one row per question
    cvqa_long = cvqa_df.melt(id_vars=["encounter_id"], 
                             var_name="qid", 
                             value_name="answer_index")
    
    # Filter out encounter_id rows
    cvqa_long = cvqa_long[cvqa_long["qid"] != "encounter_id"]
    
    # Load question definitions
    questions_path = os.path.join(config.DATASET_DIR, "closedquestions_definitions_imageclef2025.json")
    questions = load_json_file(questions_path)
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
    option_maps_path = os.path.join(config.DATASET_DIR, "option_maps.json")
    with open(option_maps_path, 'w', encoding='utf-8') as f:
        json.dump(option_maps, f, indent=2)
    
    return final_df

def process_batch(batch_df, batch_idx, output_dir, data_dir, mode="train", use_single_image=True):
    """Process a batch of data and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    batch_data = []
    
    # Debug info
    logger.info(f"Processing batch with data_dir: {data_dir}")
    if not batch_df.empty:
        first_row = batch_df.iloc[0]
        first_img_id = first_row['image_ids'][0] if first_row['image_ids'] else "no_image"
        example_path = os.path.join(data_dir, "images_train", first_img_id)
        logger.info(f"Example image path: {example_path}")
        logger.info(f"Path exists: {os.path.exists(example_path)}")
    
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx}"):
        try:
            # Skip rows with no images
            if not row['image_ids'] or len(row['image_ids']) == 0:
                continue
            
            # Get image path
            img_id = row['image_ids'][0]
#             print(f"{img_id}")  # Debug print
            
            # IMPORTANT: Use IMAGES_DIR from config instead of constructing path
            image_path = os.path.join(config.IMAGES_DIR, img_id)
            
            # Check if file exists before other validation
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            # To this simpler approach:
            try:
                img = Image.open(image_path)
                # Just check if we can access basic properties
                width, height = img.size
                img.close()
            except Exception as e:
                logger.warning(f"Invalid image {image_path}: {e}")
                continue
            
            # Format options text
            options_text = ", ".join([f"{i+1}. {opt}" for i, opt in enumerate(row['options_en'])])
            
            # Create metadata string
            metadata = f"Type: {row.get('question_type_en', '')}, Category: {row.get('question_category_en', '')}"
            
            # Create the full query text
            query_text = f"Question: Based on the image, {row['question_en']}\nQuestion Metadata: {metadata}\nOptions: {options_text}"
            
            # For inference, add critical instruction
            if mode == "inference":
                query_text += "\n\nCRITICAL INSTRUCTION: Only respond with an option if it is **clearly and unambiguously** supported by the image. If the image is unclear, incomplete, or could fit multiple answers, respond with: 'Not mentioned'. You must respond with the **exact text** of one option below. No numbers, no explanation. Given the medical context, err on the side of caution."
            
            # Add to batch data
            batch_data.append({
                "encounter_id": row['encounter_id'],
                "qid": row['qid'],
                "query_text": query_text,
                "image_paths": [image_path],  # List with single image path
                "answer_text": row['answer_text'],
                "answer_index": row.get('answer_index', -1),
                "options": row['options_en'],
                "question_type": row.get('question_type_en', ''),
                "question_category": row.get('question_category_en', '')
            })
            
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
    
    # Save batch data if not empty
    if batch_data:
        batch_file = os.path.join(output_dir, f"batch_{batch_idx}.pkl")
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
    
    return len(batch_data)

def main():
    """Main entry point for preprocessing."""
    args = parse_args()
    
    # Create output directory for the specific mode
#     output_dir = os.path.join(config.PROCESSED_DIR, args.mode)
    output_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data needs to be reprocessed
    if os.path.exists(output_dir) and not args.reprocess:
        batch_files = [f for f in os.listdir(output_dir) if f.startswith("batch_") and f.endswith(".pkl")]
        if batch_files:
            logger.info(f"Using existing processed data from {output_dir}")
            # Count total processed examples
            total_processed = 0
            for batch_file in batch_files:
                with open(os.path.join(output_dir, batch_file), 'rb') as f:
                    batch_data = pickle.load(f)
                    total_processed += len(batch_data)
            logger.info(f"Found {total_processed} processed examples")
            return
    
    # Clear output directory if reprocessing
    if args.reprocess and os.path.exists(output_dir):
        import shutil
        logger.info(f"Clearing existing processed data from {output_dir}")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    data_dir = args.data_dir
    
    # Prepare the dataset
    df = prepare_dataset(args.data_dir)  # Pass data_dir here
    logger.info(f"Prepared dataset with {len(df)} total examples")
    
    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited dataset to {args.limit} examples")
    
    # Process data in batches
    batch_size = min(args.batch_size, len(df))
    total_processed = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_idx = i // batch_size
        
        logger.info(f"Processing batch {batch_idx+1}/{(len(df)-1)//batch_size + 1}")
        processed = process_batch(batch_df, batch_idx, output_dir, data_dir, args.mode, args.use_single_image)
        total_processed += processed
        
        logger.info(f"Processed {total_processed} examples so far")
    
    logger.info(f"Preprocessing complete. Processed {total_processed} examples.")
    logger.info(f"Data saved to {output_dir}")
    
    # When processing an image
    first_row = batch_df.iloc[0]
    first_img_id = first_row['image_ids'][0]
    example_path = os.path.join(data_dir, "images_train", first_img_id)
    logger.info(f"Example image path: {example_path}")
    logger.info(f"Path exists: {os.path.exists(example_path)}")

    # Print sample for verification
    batch_file = os.path.join(output_dir, "batch_0.pkl")
    if os.path.exists(batch_file):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            if batch_data:
                sample = batch_data[0]
                logger.info("\nSample processed example:")
                logger.info(f"Encounter ID: {sample['encounter_id']}")
                logger.info(f"Question ID: {sample['qid']}")
                logger.info(f"Query text: {sample['query_text']}")
                logger.info(f"Image paths: {sample['image_paths']}")
                logger.info(f"Answer text: {sample['answer_text']}")

if __name__ == "__main__":
    main()