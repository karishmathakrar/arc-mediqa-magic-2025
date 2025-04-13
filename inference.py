#!/usr/bin/env python
"""
Main inference script for the MEDIQA project using the built-in processor approach.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import torch
import json
import pickle
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from dotenv import load_dotenv
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on medical images.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="2025_dataset/train",
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default=None,
        help="CSV file with inference data (will be generated if it doesn't exist)"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="data/processed/inference",  # Previously "processed_data_inference"
        help="Directory to save/load processed data"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="merged_model",
        help="Hugging Face model ID or path to local model"
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
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="evaluation_results/results.csv",  # Previously "inference_results.csv"
        help="Path to save inference results CSV"
    )
    parser.add_argument(
        "--json_output", 
        type=str, 
        default="evaluation_results/results.json",  # Previously "data_cvqa_sys.json"
        help="Path to save inference results in JSON format for evaluation"
    )
    parser.add_argument(
        "--indexed_output", 
        type=str, 
        default="evaluation_results/results_indices.json",  # Previously "data_cvqa_sys_indices.json"
        help="Path to save indexed JSON for evaluation"
    )
    parser.add_argument(
        "--option_maps", 
        type=str, 
        default="2025_dataset/train/option_maps.json",
        help="Path to the option maps JSON file"
    )
    parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess the data even if already processed"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=64,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.5,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    
    return parser.parse_args()

def preprocess_dataset(csv_file, data_dir, output_dir="processed_data_inference", limit=None):
    """
    Prepare the dataset by preprocessing the CSV file.
    If CSV file doesn't exist, generate it first from the raw data.
    """
    # Check if the CSV file exists, create it if not
    if not os.path.exists(csv_file):
        logger.info(f"CSV file not found, generating from raw data: {csv_file}")
        # Call the preprocessing script to generate the CSV
        generate_csv(data_dir, csv_file)
    
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
             'image_ids', 'question_type_en', 'question_category_en']]
    
    logger.info(f"Filtered dataframe shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Process the data in batches
    os.makedirs(output_dir, exist_ok=True)
    
    # Batch processing to avoid memory issues
    batch_size = min(500, len(df))
    total_processed = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_idx = i // batch_size
        
        logger.info(f"Processing batch {batch_idx+1}/{(len(df)-1)//batch_size + 1}")
        processed = process_batch(batch_df, batch_idx, output_dir, data_dir)
        total_processed += processed
        
        # Clear memory
        gc.collect()
        
        logger.info(f"Processed {total_processed} examples so far")
    
    return total_processed, output_dir

def generate_csv(data_dir, csv_output):
    """Generate a CSV file from the raw JSON data."""
    logger.info(f"Generating CSV file from raw data in {data_dir}")
    
    # Load train.json
    train_json_path = os.path.join(data_dir, "train.json")
    train_df = pd.read_json(train_json_path)
    
    # Filter relevant columns
    train_df = train_df[[
        "encounter_id", "author_id", "image_ids", "responses", 
        "query_title_en", "query_content_en"
    ]]
    
    # Generate image paths
    def generate_image_paths(image_ids):
        return [os.path.normpath(os.path.join(data_dir, "images_train", img)) for img in image_ids]
    
    train_df["image_paths"] = train_df["image_ids"].apply(generate_image_paths)
    
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

def process_batch(batch_df, batch_idx, save_dir, data_dir):
    """Process a batch of the dataset and save to disk for inference."""
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
            
            # Create the full query text
            query_text = f"Question: Based on the image, {row['question_en']}\nQuestion Metadata: {metadata}\nOptions: {options_text}"
            
            batch_data.append({
                "encounter_id": row['encounter_id'],
                "qid": row['qid'],
                "query_text": query_text,
                "image_path": image_path,
                "answer_text": row['answer_text'],
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

class MedicalImageDataset(Dataset):
    """Dataset class for preprocessed medical data."""
    def __init__(self, data_dir, processor, mode="inference"):
        self.processor = processor
        self.examples = []
        self.mode = mode
        
        for batch_file in sorted(os.listdir(data_dir)):
            if batch_file.startswith("batch_") and batch_file.endswith(".pkl"):
                with open(os.path.join(data_dir, batch_file), 'rb') as f:
                    batch_data = pickle.load(f)
                    self.examples.extend(batch_data)
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_dir}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Open just one image, convert to RGB
        image = Image.open(example['image_path']).convert("RGB")
        
        # Define system message
        system_message = "You are a medical image analysis assistant. Your only task is to examine the provided clinical image and select the exact option text that best describes what you see. Note this is not the full context so if you are unsure or speculate other regions being affected, respond with 'Not mentioned'. You must respond with the full text of one of the provided options, exactly as written. Do not include any additional words or reasoning. Given the medical context, err on the side of caution when uncertain."
        
        # Format as messages
        if self.mode == "inference":
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
            ]
        else:
            # For training mode, include the ground truth answer
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
        
        # Include metadata for later use
        return {
            "messages": messages,
            "metadata": {
                "encounter_id": example['encounter_id'],
                "qid": example['qid'],
                "ground_truth": example.get('answer_text', ''),
                "options": example.get('options', [])
            }
        }

def create_collate_fn(processor):
    """Create a collation function for the DataLoader."""
    def collate_fn(examples):
        texts = []
        images = []
        metadata = []
        
        for example in examples:
            # Extract image from messages
            image_input = None
            for msg in example["messages"]:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if isinstance(content, dict) and content.get("type") == "image" and "image" in content:
                            image_input = content["image"]
                            break
            
            if image_input is None:
                image_input = Image.new('RGB', (224, 224), color='black')
                
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=True, tokenize=False
            )
            
            texts.append(text.strip())
            images.append([image_input])
            metadata.append(example["metadata"])
        
        # Process inputs
        inputs = processor(
            text=texts, 
            images=images,
            return_tensors="pt", 
            padding=True
        )
        
        # Attach metadata
        inputs["metadata"] = metadata
        
        return inputs
    
    return collate_fn

def clean_generated_answer(text):
    """Clean up the generated answer text."""
    lines = text.strip().split('\n')
    model_content = False
    answer_lines = []
    
    for line in lines:
        if '<start_of_turn>model' in line or line.strip() == "model":
            model_content = True
            continue
        if model_content and line.strip() and not line.startswith("<") and not any(tag in line for tag in ["start_of_turn", "end_of_turn"]):
            answer_lines.append(line.strip())
    
    # If we found model content, return it
    if answer_lines:
        return " ".join(answer_lines)
    
    # If all else fails, return the last non-empty line
    for line in reversed(lines):
        if line.strip() and not line.startswith("<") and not any(tag in line for tag in ["start_of_turn", "end_of_turn"]):
            return line.strip()
    
    return ""

def convert_text_to_index(base_qid, text_answer, option_maps):
    """Convert text answer to index using option maps."""
    if base_qid not in option_maps:
        logger.warning(f"No mapping found for question ID: {base_qid}")
        return -1
    
    text_answer = text_answer.strip()
    
    # Remove option number prefix if present (e.g., "1. single spot" -> "single spot")
    text_answer = text_answer.split('. ', 1)[-1] if text_answer.split('. ', 1)[0].isdigit() else text_answer
    
    if text_answer in option_maps[base_qid]:
        return option_maps[base_qid][text_answer]
    else:
        # Try case-insensitive match as fallback
        for key, value in option_maps[base_qid].items():
            if key.lower() == text_answer.lower():
                logger.warning(f"Case mismatch for {base_qid}: '{text_answer}' vs '{key}', using index {value}")
                return value
        
        logger.warning(f"No matching option found for {base_qid}: '{text_answer}'")
        return -1

def run_inference(model, processor, dataset, batch_size=1, max_new_tokens=64, temperature=0.5):
    """Run inference on the dataset."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(processor)
    )
    
    results = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference with no gradient calculation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move tensor inputs to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)
            
            # Define stopping tokens
            stop_token_ids = [
                processor.tokenizer.eos_token_id, 
                processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]
            
            # Generate text
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-5),  # Ensure temperature is positive
                num_beams=1,  # Simple greedy or sampling
                eos_token_id=stop_token_ids
            )
            
            # Decode generated text
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], outputs)]
            generated_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Store results with metadata
            for i, text in enumerate(generated_texts):
                metadata = batch["metadata"][i]
                
                results.append({
                    "encounter_id": metadata["encounter_id"],
                    "qid": metadata["qid"],
                    "generated_answer": text,
                    "ground_truth": metadata["ground_truth"],
                    "cleaned_answer": clean_generated_answer(text)
                })
    
    return results

def save_results(results, output_file, json_output=None, indexed_output=None, option_maps_path=None):
    """Save inference results to files."""
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    logger.info("\nInference Results:")
    logger.info(f"Total examples processed: {len(results_df)}")
    
    # Add cleaned answers if not already present
    if "cleaned_answer" not in results_df.columns:
        results_df["cleaned_answer"] = results_df["generated_answer"].apply(clean_generated_answer)
    
    # Save CSV results
    results_df.to_csv(output_file, index=False)
    logger.info(f"CSV results saved to '{output_file}'")
    
    # Load option maps for index conversion
    if option_maps_path and os.path.exists(option_maps_path):
        with open(option_maps_path, 'r') as f:
            option_maps = json.load(f)
    else:
        option_maps = {}
        logger.warning("Option maps file not found, indices may be incorrect")
    
    # Convert to JSON format for evaluation (using text answers)
    if json_output:
        # Group by encounter_id
        encounter_groups = results_df.groupby('encounter_id')
        output_data = []
        
        for encounter_id, group in encounter_groups:
            # Create entry for this encounter
            encounter_entry = {"encounter_id": encounter_id}
            
            # Add each question's answer
            for _, row in group.iterrows():
                qid = row['qid']
                answer = row['cleaned_answer']
                encounter_entry[qid] = answer
            
            output_data.append(encounter_entry)
        
        # Save as JSON
        with open(json_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"JSON results for evaluation saved to '{json_output}'")
    
    # Convert to indexed JSON format
    if indexed_output:
        # Group by encounter_id
        encounter_groups = results_df.groupby('encounter_id')
        output_data = []
        
        for encounter_id, group in encounter_groups:
            # Create entry for this encounter
            encounter_entry = {"encounter_id": encounter_id}
            
            # Add each question's answer as a numeric index
            for _, row in group.iterrows():
                qid = row['qid']
                base_qid = qid.split("-")[0]
                text_answer = row['cleaned_answer']
                
                # Convert to index
                index = convert_text_to_index(base_qid, text_answer, option_maps)
                
                # Store the index
                encounter_entry[qid] = index
            
            output_data.append(encounter_entry)
        
        # Save as JSON
        with open(indexed_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Indexed JSON results for evaluation saved to '{indexed_output}'")
    
    return results_df

def main():
    """Main entry point."""
    args = parse_args()
    
    # Clear GPU memory before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Set default CSV file path if not specified
    if args.csv_file is None:
        args.csv_file = os.path.join(args.data_dir, "final_df_2.csv")
    
    # Process data
    if os.path.exists(args.processed_dir) and not args.reprocess:
        logger.info(f"Using existing processed data from {args.processed_dir}")
    else:
        if args.reprocess and os.path.exists(args.processed_dir):
            import shutil
            shutil.rmtree(args.processed_dir)
        
        _, args.processed_dir = preprocess_dataset(
            args.csv_file, 
            args.data_dir, 
            output_dir=args.processed_dir, 
            limit=args.limit
        )
    
    # Load HuggingFace token from environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    # Load processor
    logger.info(f"Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
    
    # Load model with quantization
    logger.info(f"Loading model from: {args.model_id}")
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        logger.warning("GPU may not support bfloat16 natively")
    
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add quantization for better memory efficiency
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )
    
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs, token=hf_token)
    
    # Create dataset
    logger.info(f"Creating dataset from: {args.processed_dir}")
    dataset = MedicalImageDataset(args.processed_dir, processor, mode="inference")
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    # Run inference
    logger.info("Starting inference...")
    results = run_inference(
        model=model,
        processor=processor,
        dataset=dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Save results
    save_results(
        results=results,
        output_file=args.output_file,
        json_output=args.json_output,
        indexed_output=args.indexed_output,
        option_maps_path=args.option_maps
    )
    
    logger.info("Inference completed successfully.")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    logger.info("Model unloaded and GPU memory cleared.")

if __name__ == "__main__":
    main()