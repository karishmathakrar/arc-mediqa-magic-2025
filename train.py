"""
Training script for the MEDIQA project.
Fine-tunes a gemma-3-4b-it model on medical image data.
"""
import os
import sys
import argparse
import logging
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv

import config
from utils import print_system_info, clear_gpu_memory, create_dummy_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model on medical images")
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default=os.path.join(config.PROCESSED_DIR, "train"),
        help="Directory with processed training data"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=config.MODEL_ID,
        help="HuggingFace model ID to fine-tune"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=config.FINETUNED_MODEL_DIR,
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--merged_dir", 
        type=str, 
        default=config.MERGED_MODEL_DIR,
        help="Directory to save merged model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=config.BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=config.GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=config.NUM_TRAIN_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=config.LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--skip_merge", 
        action="store_true",
        help="Skip merging LoRA weights with base model"
    )
    
    return parser.parse_args()

class MedicalImageDataset(Dataset):
    """Dataset for medical image-text pairs."""
    def __init__(self, data_dir, processor):
        self.processor = processor
        self.examples = []
        
        # Load batches from pickle files
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
        
        # Load and convert the first image to RGB
        image = Image.open(example['image_paths'][0]).convert("RGB")
        
        # Format as chat messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": config.SYSTEM_PROMPT}],
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
                image_input = create_dummy_image()
                
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
        
        # Get token IDs for special tokens
        boi_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
        eoi_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["eoi_token"]
        )
        
        # Mask tokens that shouldn't contribute to the loss
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Mask image tokens
        if boi_token_id:
            labels[labels == boi_token_id] = -100
        if eoi_token_id:
            labels[labels == eoi_token_id] = -100
        
        batch["labels"] = labels
        return batch
    
    return collate_fn

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Print system info
    print_system_info()
    
    # Load HuggingFace token from environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.merged_dir, exist_ok=True)
    
    # Check if GPU supports bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        logger.warning("GPU may not support bfloat16 natively")
    
    # Load processor
    logger.info(f"Loading processor for model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
    
    # Load model with quantization
    logger.info(f"Loading model: {args.model_id}")
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
    
    # Debug a sample
    sample_example = dataset[0]
    logger.info(f"Sample example roles: {[m['role'] for m in sample_example['messages']]}")
    
    # Create LoRA config
    logger.info("Creating LoRA configuration")
    peft_config = LoraConfig(
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        r=config.LORA_RANK,
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
        max_grad_norm=config.MAX_GRAD_NORM,
        warmup_ratio=config.WARMUP_RATIO,
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