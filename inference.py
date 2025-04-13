"""
Inference script for the MEDIQA project.
Runs inference on both the base model and the fine-tuned model.
"""
import os
import sys
import argparse
import logging
import json
import gc
import torch
import pickle
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from dotenv import load_dotenv

import config
from utils import print_system_info, clear_gpu_memory, clean_generated_text, save_json_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on medical images")
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default=os.path.join(config.PROCESSED_DIR, "inference"),
        help="Directory with processed inference data"
    )
    parser.add_argument(
        "--base_model_id", 
        type=str, 
        default=config.MODEL_ID,
        help="HuggingFace model ID for base model"
    )
    parser.add_argument(
        "--finetuned_model_dir", 
        type=str, 
        default=config.MERGED_MODEL_DIR,
        help="Directory with fine-tuned model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=config.OUTPUTS_DIR,
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=config.MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=config.TEMPERATURE,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for inference"
    )
    parser.add_argument(
        "--skip_base", 
        action="store_true",
        help="Skip inference with base model"
    )
    parser.add_argument(
        "--skip_finetuned", 
        action="store_true",
        help="Skip inference with fine-tuned model"
    )
    
    return parser.parse_args()

def load_model_and_processor(model_path, token=None):
    """Load model and processor for inference."""
    logger.info(f"Loading model and processor from: {model_path}")
    
    # Configure model parameters
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
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, token=token)
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs, token=token)
    
    return model, processor

def load_data(processed_dir, limit=None):
    """Load processed data for inference."""
    examples = []
    
    # Load all batch files
    batch_files = sorted([f for f in os.listdir(processed_dir) if f.startswith("batch_") and f.endswith(".pkl")])
    
    for batch_file in batch_files:
        with open(os.path.join(processed_dir, batch_file), "rb") as f:
            batch_data = pickle.load(f)
            examples.extend(batch_data)
            
            # Break if limit reached
            if limit and len(examples) >= limit:
                examples = examples[:limit]
                break
    
    logger.info(f"Loaded {len(examples)} examples for inference")
    return examples

def run_inference_on_example(example, model, processor, max_new_tokens=64, temperature=0.5):
    """Run inference on a single example."""
    # Load the image
    image = Image.open(example["image_paths"][0]).convert("RGB")
    
    # Format as messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": config.SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": example["query_text"]},
                {"type": "image", "image": image},
            ],
        },
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize the text and process the images
    inputs = processor(
        text=[text],
        images=[[image]],  # Nested list as processor expects
        padding=True,
        return_tensors="pt",
    )
    
    # Move the inputs to the device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate the output
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-5),
        top_p=config.TOP_P,
        eos_token_id=stop_token_ids,
        disable_compile=True
    )
    
    # Trim the generation and decode the output to text
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean the generated text to get just the answer
    cleaned_answer = clean_generated_text(output_text)
    
    return {
        "encounter_id": example["encounter_id"],
        "qid": example["qid"],
        "ground_truth": example["answer_text"],
        "generated_answer": output_text,
        "cleaned_answer": cleaned_answer,
        "options": example["options"]
    }

def run_inference(examples, model, processor, output_file, max_new_tokens=64, temperature=0.5):
    """Run inference on examples and save results."""
    results = []
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Process examples
    for i, example in enumerate(tqdm(examples, desc="Running inference")):
        with torch.no_grad():
            result = run_inference_on_example(
                example, model, processor, max_new_tokens, temperature
            )
            results.append(result)
        
        # Log progress periodically
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(examples)} examples")
    
    # Format results for evaluation
    formatted_results = format_results_for_evaluation(results)
    
    # Save results
    save_json_file(formatted_results, output_file)
    logger.info(f"Saved inference results to {output_file}")
    
    return results

def format_results_for_evaluation(results):
    """Format results for evaluation by the conference scripts."""
    # Group by encounter_id
    encounters = {}
    for result in results:
        encounter_id = result["encounter_id"]
        if encounter_id not in encounters:
            encounters[encounter_id] = {"encounter_id": encounter_id}
        
        # Add answer to encounter
        encounters[encounter_id][result["qid"]] = result["cleaned_answer"]
    
    # Convert to list of encounters
    return list(encounters.values())

def main():
    """Main entry point for inference."""
    # Parse arguments
    args = parse_args()
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Print system info
    print_system_info()
    
    # Load HuggingFace token from environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    # Create output directories
    base_output_dir = os.path.join(args.output_dir, "base_model")
    finetuned_output_dir = os.path.join(args.output_dir, "finetuned_model")
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(finetuned_output_dir, exist_ok=True)
    
    # Load data
    examples = load_data(args.processed_dir, args.limit)
    
    if not examples:
        logger.error("No examples found for inference. Exiting.")
        return
    
    # Run inference with base model if not skipped
    if not args.skip_base:
        logger.info(f"Running inference with base model: {args.base_model_id}")
        
        # Load base model and processor
        base_model, base_processor = load_model_and_processor(args.base_model_id, hf_token)
        
        # Run inference
        base_output_file = os.path.join(base_output_dir, "results.json")
        run_inference(
            examples, 
            base_model, 
            base_processor, 
            base_output_file,
            args.max_new_tokens,
            args.temperature
        )
        
        # Clean up
        del base_model
        del base_processor
        clear_gpu_memory()
    
    # Run inference with fine-tuned model if not skipped
    if not args.skip_finetuned:
        logger.info(f"Running inference with fine-tuned model: {args.finetuned_model_dir}")
        
        # Load fine-tuned model and processor
        finetuned_model, finetuned_processor = load_model_and_processor(args.finetuned_model_dir)
        
        # Run inference
        finetuned_output_file = os.path.join(finetuned_output_dir, "results.json")
        run_inference(
            examples, 
            finetuned_model, 
            finetuned_processor, 
            finetuned_output_file,
            args.max_new_tokens,
            args.temperature
        )
        
        # Clean up
        del finetuned_model
        del finetuned_processor
        clear_gpu_memory()
    
    logger.info("Inference completed successfully")

if __name__ == "__main__":
    main()