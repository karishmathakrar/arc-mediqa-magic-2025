"""
Utility script to convert inference results to the format expected by the evaluation script.
"""
import os
import json
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_csv_to_json(csv_file, output_file):
    """
    Convert the CSV results from inference to the JSON format expected by the evaluator.
    
    Args:
        csv_file: Path to the CSV file with inference results
        output_file: Path to save the output JSON file
    """
    # Read CSV file
    logger.info(f"Reading results from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Group by encounter_id
    encounter_groups = df.groupby('encounter_id')
    
    # Prepare the output JSON structure
    output_data = []
    
    for encounter_id, group in encounter_groups:
        # Create entry for this encounter
        encounter_entry = {"encounter_id": encounter_id}
        
        # Add each question's answer
        for _, row in group.iterrows():
            qid = row['qid']
            # Use the generated answer, cleaning up any system/model prompts
            answer = clean_generated_answer(row['generated_answer'])
            encounter_entry[qid] = answer
        
        output_data.append(encounter_entry)
    
    # Save as JSON
    logger.info(f"Saving converted results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Conversion complete. Processed {len(output_data)} encounters.")
    return output_data

def clean_generated_answer(text):
    """
    Clean up the generated answer to extract just the answer text.
    
    Args:
        text: Generated text from the model
        
    Returns:
        Cleaned answer text
    """
    # Remove system/user/model prompts
    lines = text.strip().split('\n')
    model_content = False
    answer_lines = []
    
    for line in lines:
        if line.strip() == "model":
            model_content = True
            continue
        if model_content and line.strip() and not line.startswith("<"):
            answer_lines.append(line.strip())
    
    # If we found model content, return it
    if answer_lines:
        return " ".join(answer_lines)
    
    # Otherwise, try to find the most likely answer by looking for text after CRITICAL INSTRUCTION
    for i, line in enumerate(lines):
        if "CRITICAL INSTRUCTION" in line and i+1 < len(lines):
            # Look ahead for non-empty lines that aren't tags
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith("<"):
                    return lines[j].strip()
    
    # If all else fails, return the last non-empty line
    for line in reversed(lines):
        if line.strip() and not line.startswith("<"):
            return line.strip()
    
    # If we got here, there's no useful text
    return ""

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert inference results to evaluation format")
    parser.add_argument(
        "--input_file", 
        type=str,
        required=True,
        help="Path to the CSV file with inference results"
    )
    parser.add_argument(
        "--output_file", 
        type=str,
        required=True,
        help="Path to save the output JSON file"
    )
    
    args = parser.parse_args()
    convert_csv_to_json(args.input_file, args.output_file)
