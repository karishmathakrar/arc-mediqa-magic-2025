"""
Training functionality for the MEDIQA project.
"""
import os
import logging
import torch
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# Import config and utils directly instead of using relative imports
from config.config import *
import utils

logger = logging.getLogger(__name__)

def create_peft_config():
    """
    Create a LoRA configuration for efficient fine-tuning.
    
    Returns:
        LoRA configuration
    """
    return LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_RANK,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

def create_trainer_config(output_dir="fine_tuned_model"):
    """
    Create a trainer configuration.
    
    Args:
        output_dir: Directory to save the model
        
    Returns:
        SFT configuration
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        bf16=True,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,  # Critical for custom datasets
        label_names=["labels"],  # Explicitly setting label_names
    )

def train_model(model, train_dataset, processor, collate_fn=None, output_dir="fine_tuned_model"):
    """
    Train a model using SFTTrainer.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        processor: Processor for tokenization
        collate_fn: Function to collate batch data
        output_dir: Directory to save the model
        
    Returns:
        Trained model path
    """
    utils.ensure_directory(output_dir)
    
    # Create configurations
    peft_config = create_peft_config()
    trainer_config = create_trainer_config(output_dir=output_dir)
    
    logger.info("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Training complete, saving model to {output_dir}")
    trainer.save_model()
    
    return output_dir

def merge_and_save_lora_model(base_model_id, adapter_model_dir, output_dir="merged_model"):
    """
    Merge LoRA weights with base model and save.
    
    Args:
        base_model_id: Base model ID
        adapter_model_dir: Directory with adapter weights
        output_dir: Directory to save the merged model
        
    Returns:
        Output directory path
    """
    utils.ensure_directory(output_dir)
    
    # First load the processor from the adapter model to ensure we match tokenizers
    logger.info(f"Loading processor from adapter: {adapter_model_dir}")
    processor = AutoProcessor.from_pretrained(adapter_model_dir, token=HF_TOKEN)
    
    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id, 
        low_cpu_mem_usage=True, 
        token=HF_TOKEN
    )
    
    # Important: Resize the token embeddings to match what was used during training
    logger.info(f"Resizing token embeddings to match adapter (size: {len(processor.tokenizer)})")
    model.resize_token_embeddings(len(processor.tokenizer))
    
    logger.info(f"Loading PEFT adapter from: {adapter_model_dir}")
    peft_model = PeftModel.from_pretrained(model, adapter_model_dir)
    
    logger.info("Merging models...")
    merged_model = peft_model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    
    # Also save the processor
    logger.info("Saving processor...")
    processor.save_pretrained(output_dir)
    
    return output_dir

# def merge_and_save_lora_model(base_model_id, adapter_model_dir, output_dir="merged_model"):
#     """
#     Merge LoRA weights with base model and save.
    
#     Args:
#         base_model_id: Base model ID
#         adapter_model_dir: Directory with adapter weights
#         output_dir: Directory to save the merged model
        
#     Returns:
#         Output directory path
#     """
#     utils.ensure_directory(output_dir)
    
#     logger.info(f"Loading base model: {base_model_id}")
#     model = AutoModelForImageTextToText.from_pretrained(
#         base_model_id, 
#         low_cpu_mem_usage=True, 
#         token=HF_TOKEN
#     )
    
#     logger.info(f"Loading PEFT adapter from: {adapter_model_dir}")
#     peft_model = PeftModel.from_pretrained(model, adapter_model_dir)
    
#     logger.info("Merging models...")
#     merged_model = peft_model.merge_and_unload()
    
#     logger.info(f"Saving merged model to: {output_dir}")
#     merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    
#     # Also save the processor
#     logger.info("Saving processor...")
#     processor = AutoProcessor.from_pretrained(adapter_model_dir, token=HF_TOKEN)
#     processor.save_pretrained(output_dir)
    
#     return output_dir