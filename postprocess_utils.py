"""
Utility functions for post-processing model outputs for evaluation.
"""
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define option mappings for each question category
OPTION_MAPS = {
    # Site (How much of body is affected)
    "CQID010": {
        "single spot": 0,
        "limited area": 1,
        "widespread": 2,
        "Not mentioned": 3
    },
    # Site Location (Where is the affected area)
    "CQID011": {
        "head": 0,
        "neck": 1,
        "upper extremities": 2,
        "lower extremities": 3,
        "chest/abdomen": 4,
        "back": 5,
        "other (please specify)": 6,
        "Not mentioned": 7
    },
    # Size (How large are the affected areas)
    "CQID012": {
        "size of thumb nail": 0,
        "size of palm": 1,
        "larger area": 2,
        "Not mentioned": 3
    },
    # Onset (When did the patient first notice)
    "CQID015": {
        "within hours": 0,
        "within days": 1,
        "within weeks": 2,
        "within months": 3,
        "over a year": 4,
        "multiple years": 5,
        "Not mentioned": 6
    },
    # Skin Description (What label best describes)
    "CQID020": {
        "raised or bumpy": 0,
        "flat": 1,
        "skin loss or sunken": 2,
        "thick or raised": 3,
        "thin or close to the surface": 4,
        "warty": 5,
        "crust": 6,
        "scab": 7,
        "weeping": 8,
        "Not mentioned": 9
    },
    # Itch (Is there any associated itching)
    "CQID025": {
        "yes": 0,
        "no": 1,
        "Not mentioned": 2
    },
    # Lesion Color (What is the color)
    "CQID034": {
        "normal skin color": 0,
        "pink": 1,
        "red": 2,
        "brown": 3,
        "blue": 4,
        "purple": 5,
        "black": 6,
        "white": 7,
        "combination (please specify)": 8,
        "hyperpigmentation": 9,
        "hypopigmentation": 10,
        "Not mentioned": 11
    },
    # Lesion Count (How many skin lesions)
    "CQID035": {
        "single": 0,
        "multiple (please specify)": 1,
        "Not mentioned": 2
    },
    # Texture (What is the skin lesion texture)
    "CQID036": {
        "smooth": 0,
        "rough": 1,
        "Not mentioned": 2
    }
}

def clean_generated_answer(text):
    """
    Clean up the generated answer to extract just the answer text.
    
    Args:
        text: Generated text from the model
        
    Returns:
        Cleaned answer text
    """
    print(f"DEBUG - Processing text: {text[:100]}...")  # Print first 100 chars
    
    # Remove system/user/model prompts
    lines = text.strip().split('\n')
    print(f"DEBUG - Split into {len(lines)} lines")
    
    # Print the last few lines to see what's there
    if len(lines) > 5:
        print(f"DEBUG - Last 5 lines: {lines[-5:]}")
    else:
        print(f"DEBUG - All lines: {lines}")
    
    # Look for lines after "model" appears
    after_model = False
    answer_lines = []
    
    for line in lines:
        if after_model and line.strip() and not line.startswith("<"):
            answer_lines.append(line.strip())
            print(f"DEBUG - Found answer line: {line.strip()}")
        
        if "model" in line:
            after_model = True
            print(f"DEBUG - Found 'model' in line: {line}")
    
    # If we found lines after "model", return them
    if answer_lines:
        result = " ".join(answer_lines)
        print(f"DEBUG - Returning result: {result}")
        return result
    
    # If all else fails, return the last non-empty line
    for line in reversed(lines):
        if line.strip() and not line.startswith("<"):
            print(f"DEBUG - Returning last non-empty line: {line.strip()}")
            return line.strip()
    
    # If we got here, there's no useful text
    print("DEBUG - No useful text found, returning empty string")
    return ""

def convert_text_to_index(base_qid, text_answer):
    """
    Convert a text answer to its corresponding numeric index.
    
    Args:
        base_qid: Base question ID (e.g., 'CQID010')
        text_answer: Text answer to convert
        
    Returns:
        Numeric index for the answer, or -1 if not found
    """
    if base_qid not in OPTION_MAPS:
        logger.warning(f"No mapping found for question ID: {base_qid}")
        return -1
        
    text_answer = text_answer.strip()
    if text_answer in OPTION_MAPS[base_qid]:
        return OPTION_MAPS[base_qid][text_answer]
    else:
        # Try case-insensitive match as fallback
        for key, value in OPTION_MAPS[base_qid].items():
            if key.lower() == text_answer.lower():
                logger.warning(f"Case mismatch for {base_qid}: '{text_answer}' vs '{key}', using index {value}")
                return value
                
        logger.warning(f"No matching option found for {base_qid}: '{text_answer}'")
        return -1

def convert_csv_to_indexed_json(csv_file, output_file):
    """
    Convert the CSV results to indexed JSON format for evaluation.
    
    Args:
        csv_file: Path to the CSV file with inference results
        output_file: Path to save the output JSON file
    """
    import pandas as pd
    
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
        
        # Add each question's answer as a numeric index
        for _, row in group.iterrows():
            qid = row['qid']
            base_qid, suffix = qid.split("-")
            
            # Clean and extract the answer
            generated_text = row['generated_answer']
            text_answer = clean_generated_answer(generated_text)
            
            # Convert to index
            index = convert_text_to_index(base_qid, text_answer)
            
            # Store the index
            encounter_entry[qid] = index
        
        output_data.append(encounter_entry)
    
    # Save as JSON
    logger.info(f"Saving indexed results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Conversion complete. Processed {len(output_data)} encounters.")
    return output_data

def group_by_base_qid(input_file, output_file):
    """
    Group questions by their base QID for the evaluation format.
    This is an optional step if the evaluation requires grouped questions.
    
    Args:
        input_file: Path to the input JSON file with individual questions
        output_file: Path to save the output JSON file with grouped questions
    """
    # Load input JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize output
    output_data = []
    
    # Process each encounter
    for encounter in data:
        encounter_id = encounter["encounter_id"]
        new_encounter = {"encounter_id": encounter_id}
        
        # Group by base QID
        qid_groups = {}
        
        for key, value in encounter.items():
            if key == "encounter_id":
                continue
                
            if "-" in key:
                base_qid, _ = key.split("-")
                
                if base_qid not in qid_groups:
                    qid_groups[base_qid] = []
                    
                qid_groups[base_qid].append(value)
            else:
                # Already a base QID (shouldn't happen but just in case)
                new_encounter[key] = value
        
        # Add grouped values to the output
        for base_qid, values in qid_groups.items():
            new_encounter[base_qid] = values
        
        output_data.append(new_encounter)
    
    # Save output JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Grouped JSON saved to {output_file}")
    return output_data
