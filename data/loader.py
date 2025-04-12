"""
Data loading utilities for the MEDIQA project.
"""
import os
import pickle
import torch
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Use absolute imports instead of relative imports
import utils

logger = logging.getLogger(__name__)

class MedicalImageDataset(Dataset):
    """Dataset class for processed medical data."""
    def __init__(self, data_dir, processor, mode="inference"):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory with processed data
            processor: Processor for tokenization
            mode: Either 'train' or 'inference'
        """
        self.processor = processor
        self.examples = []
        self.mode = mode
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return
            
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
        
        images = [Image.open(path).convert("RGB") for path in example['image_paths']]
        
        if self.mode == "train":
            # Format for training (includes assistant's response)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an AI assistant answering medical questions based on clinical images."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example['query_text']},
                        *[{"type": "image", "image": img} for img in images],
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example['answer_text']}],
                },
            ]
            
            return {"messages": messages}
        
        else:
            # Format for inference with a specific instruction to select an option
            query_text = example['query_text']
            # Make instructions even more forceful
#             query_text += "\n\nSelect one of the available options."
#             query_text += "\n\nCRITICAL INSTRUCTION: Analyze the image and select the most appropriate option from the numbered list above. Your ENTIRE response must be ONLY the single digit (1, 2, 3, etc.) corresponding to the CORRECT option number. DO NOT include any explanation, reasoning, or additional text."
#             query_text += "\n\nCRITICAL INSTRUCTION: Carefully examine the image. If a specific option clearly matches the visual evidence, select its number. If the image does not clearly support any option, respond with the number corresponding to 'Not mentioned'. Respond ONLY with the single digit (1, 2, 3, etc.). DO NOT include explanations or additional text."
#             query_text += "\n\nCRITICAL INSTRUCTION: Carefully examine the image. Choose the most appropriate option from the list below. Your response must be the exact text of one of the options provided. If none are clearly supported by the image, respond with 'Not mentioned'. DO NOT include any explanation or numbering."
            query_text += "\n\nCRITICAL INSTRUCTION: Only respond with an option if it is **clearly and unambiguously** supported by the image. If the image is unclear, incomplete, or could fit multiple answers, respond with: 'Not mentioned'. You must respond with the **exact text** of one option below. No numbers, no explanation. Given the medical context, err on the side of caution."
            messages = [
                {
                    "role": "system",
#                     "content": [{"type": "text", "text": "You are a physician answering medical questions based on clinical images that responds based ONLY on the provided options."}],
#                     "content": [{"type": "text", "text": "You are a medical image analysis assistant. Your only task is to examine the provided images and select the correct option number that best describes what you see. Respond with ONLY the option number (e.g., '2') and nothing else."}],
#                     "content": [{"type": "text", "text": "You are a medical image analysis assistant. Your only task is to examine the provided clinical images and select the correct option number that matches what is visually evident. If the image does not clearly support any of the options, select the number for 'Not mentioned'. Respond ONLY with the option number (e.g., '2'). Do not explain or add any text."}],
                    "content": [{"type": "text", "text": "You are a medical image analysis assistant. Your only task is to examine the provided clinical images and select the exact option text that best describes what you see. Note this is not the full context so if you are unsure or speculate other regions being affected, respond with 'Not mentioned'. You must respond with the full text of one of the provided options, exactly as written. Do not include any additional words or reasoning."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_text},
                        *[{"type": "image", "image": img} for img in images],
                    ],
                },
            ]
            
            return {
                "messages": messages, 
                "encounter_id": example['encounter_id'],
                "qid": example['qid'],
                "ground_truth": example['answer_text']
            }

def create_collate_fn(processor, mode="inference"):
    """
    Create a collate function for the dataloader.
    
    Args:
        processor: HuggingFace processor
        mode: Either 'train' or 'inference'
        
    Returns:
        Collate function
    """
    def collate_fn(examples):
        texts = []
        images_per_example = []
        metadata = []
        
        for example in examples:
            image_inputs = utils.process_vision_info(example["messages"])
            if not image_inputs:
                logger.warning(f"Using dummy image — Example roles: {[m['role'] for m in example['messages']]}")
                image_inputs = [utils.create_dummy_image()]
            
            # Apply chat template - add generation prompt only for inference
            add_generation_prompt = (mode == "inference")
            text = processor.apply_chat_template(
                example["messages"], 
                add_generation_prompt=add_generation_prompt, 
                tokenize=False
            )
            
            # Count image tokens explicitly
            num_image_tokens = text.count("<image>")
            
            # Ensure number of images matches tokens
            if len(image_inputs) < num_image_tokens:
                needed_dummies = num_image_tokens - len(image_inputs)
                image_inputs += [utils.create_dummy_image()] * needed_dummies
            elif len(image_inputs) > num_image_tokens:
                # Never truncate to zero
                image_inputs = image_inputs[:max(1, num_image_tokens)]
            
            texts.append(text.strip())
            images_per_example.append(image_inputs)
            
            if mode == "inference" and hasattr(example, "get"):
                metadata.append({
                    "encounter_id": example.get("encounter_id", ""),
                    "qid": example.get("qid", ""),
                    "ground_truth": example.get("ground_truth", "")
                })
        
        # Use the processor to tokenize inputs
        batch = processor(
            text=texts, 
            images=images_per_example,
            return_tensors="pt", 
            padding=True
        )
        
        if mode == "train":
            # Create labels for training
            labels = batch["input_ids"].clone()
            
            # Get image token ID
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
            
            # Mask tokens that shouldn't contribute to the loss
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            
            # Special token handling
            if hasattr(processor.tokenizer, "bos_token_id"):
                bos_id = processor.tokenizer.bos_token_id
                if bos_id is not None:
                    labels[labels == bos_id] = -100
                    
            # Handle system message
            if "<start_of_turn>system" in processor.tokenizer.get_vocab():
                system_token_id = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>system")
                labels[labels == system_token_id] = -100
            
            batch["labels"] = labels
        else:
            # Add metadata for inference
            batch["metadata"] = metadata
            
        return batch
    
    return collate_fn


def create_dataloader(dataset, processor, batch_size=1, shuffle=False, mode="inference"):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset object
        processor: HuggingFace processor
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        mode: Either 'train' or 'inference'
        
    Returns:
        DataLoader object
    """
    collate_fn = create_collate_fn(processor, mode)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def debug_template_application(examples, processor):
    """
    Debug template application for the given examples.
    
    Args:
        examples: List of dataset examples
        processor: HuggingFace processor
    """
    logger.info("\n=== DEBUGGING TEMPLATE APPLICATION ===")
    
    for i, example in enumerate(examples):
        logger.info(f"\nExample {i}:")
        
        # First, check message structure
        logger.info("Message structure:")
        for j, msg in enumerate(example["messages"]):
            role = msg.get("role", "unknown")
            logger.info(f"  Message {j} role: {role}")
            
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
                
            logger.info(f"  Content types: {[c.get('type') if isinstance(c, dict) else type(c).__name__ for c in content]}")
            
            # Check specifically for image content
            image_count = sum(1 for c in content if isinstance(c, dict) and c.get('type') == 'image')
            logger.info(f"  Image content count: {image_count}")
        
        # Next, check template application
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=True, tokenize=False
        )
        
        # Count image tokens in generated text
        img_token_count = text.count("<image>")
        logger.info(f"  Image tokens in text: {img_token_count}")
        
        # Print text snippet to see if tokens appear
        logger.info(f"  Text preview: {text[:200]}...")
        
        # Check if there's a mismatch
        if img_token_count == 0:
            logger.warning("  WARNING: No image tokens found in text!")
            
            # Examine raw content in more detail
            logger.info("  Detailed content examination:")
            for j, msg in enumerate(example["messages"]):
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        content = [content]
                    
                    for k, item in enumerate(content):
                        if isinstance(item, dict):
                            logger.info(f"    Item {k}: type={item.get('type')}, keys={list(item.keys())}")
                            if item.get('type') == 'image':
                                logger.info(f"    Found image content item")
                            else:
                                logger.info(f"    Item has type={item.get('type')}, not 'image'")