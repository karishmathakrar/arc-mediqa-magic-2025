#!/usr/bin/env python
"""
Main training script for the MEDIQA project using the built-in processor.
"""
import os
import sys
import argparse
import logging
import torch
import pickle
import json
import pandas as pd
import gc
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune model on medical images.")
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
        help="CSV file with training data (will be generated if it doesn't exist)"
    )
        parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="data/processed/train",  # Previously "processed_data_train"
        help="Directory to save/load processed data"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="google/gemma-3-4b-it",
        help="Hugging Face model ID or path to local model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/fine_tuned",  # Previously "fine_tuned_model"
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--merged_dir", 
        type=str, 
        default="models/merged",  # Previously "merged_model"
        help="Directory to save the merged model"
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
        help="Per-device batch size for training"
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--reprocess", 
        action="store_true",
        help="Reprocess the data even if already processed"
    )
    parser.add_argument(
        "--skip_merge", 
        action="store_true",
        help="Skip merging LoRA weights with base model"
    )
    
    return parser.parse_args()

def preprocess_dataset(csv_file, data_dir, output_dir="processed_data", limit=None):
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
    """Process a batch of the dataset and save to disk."""
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
                "question_category": row.get('question_category_en', '')
            })
        
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
    
    batch_file = os.path.join(save_dir, f"batch_{batch_idx}.pkl")
    with open(batch_file, 'wb') as f:
        pickle.dump(batch_data, f)
    
    return len(batch_data)

class MedicalImageDataset(Dataset):
    """Dataset class for preprocessed medical data."""
    def __init__(self, data_dir, processor):
        self.processor = processor
        self.examples = []
        
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

def create_collate_fn(processor):
    """Create a collation function for the DataLoader."""
    def collate_fn(examples):
        texts = []
        images = []
        
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
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            
            texts.append(text.strip())
            images.append([image_input])
        
        batch = processor(
            text=texts, 
            images=images,
            return_tensors="pt", 
            padding=True
        )
        
        labels = batch["input_ids"].clone()
        
        # Get token IDs for image related tokens
        special_tokens = {
            token: processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map.get(token, "")
            ) for token in ["boi_token", "eoi_token", "image_token"]
        }
        
        # Mask tokens that shouldn't contribute to the loss
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Mask special tokens
        for token_id in special_tokens.values():
            if token_id:
                labels[labels == token_id] = -100
        
        batch["labels"] = labels
        return batch
    
    return collate_fn

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
    logger.info(f"Loading processor for model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
    
    # Load model with quantization
    logger.info(f"Loading model: {args.model_id}")
    
    # Check if GPU supports bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        logger.warning("GPU may not support bfloat16 natively")
    
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
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
    dataset = MedicalImageDataset(args.processed_dir, processor)
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    # Create LoRA config
    logger.info("Creating LoRA configuration")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    
    # Create trainer config
    logger.info("Creating trainer configuration")
    trainer_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=args.lr,
        bf16=True,
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
    
    # Create collate function
    collate_fn = create_collate_fn(processor)
    
    # Initialize trainer
    logger.info("Initializing SFT trainer")
    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Training complete, saving model to {args.output_dir}")
    trainer.save_model()
    
    # Merge and save if not skipped
    if not args.skip_merge:
        logger.info("Freeing up memory for model merging")
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"Loading base model: {args.model_id}")
        from peft import PeftModel
        model = AutoModelForImageTextToText.from_pretrained(args.model_id, low_cpu_mem_usage=True, token=hf_token)
        
        logger.info(f"Loading PEFT adapter from: {args.output_dir}")
        peft_model = PeftModel.from_pretrained(model, args.output_dir)
        
        logger.info("Merging models...")
        merged_model = peft_model.merge_and_unload()
        
        logger.info(f"Saving merged model to: {args.merged_dir}")
        merged_model.save_pretrained(args.merged_dir, safe_serialization=True, max_shard_size="2GB")
        
        # Also save the processor
        logger.info("Saving processor...")
        processor.save_pretrained(args.merged_dir)
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()