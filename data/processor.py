"""
Data processing classes for the MEDIQA project.
"""
import os
import json
import pickle
from PIL import Image
from tqdm.auto import tqdm
import logging
import pandas as pd

# Use absolute imports instead of relative imports
import utils

logger = logging.getLogger(__name__)

class MedicalDataProcessor:
    """Process medical data for both training and inference."""
    
    def __init__(self, base_dir, mode="train"):
        """
        Initialize the processor.
        
        Args:
            base_dir: Base directory for the dataset
            mode: Either 'train' or 'inference'
        """
        self.base_dir = base_dir
        self.mode = mode
        self.images_dir = os.path.join(base_dir, "images_train")
        
        # Load question definitions
        questions_path = os.path.join(base_dir, "closedquestions_definitions_imageclef2025.json")
        self.questions = utils.load_json_file(questions_path)
        
        # Create question lookup dictionary for fast access
        self.questions_lookup = {q["qid"]: q for q in self.questions}
        
        # Get encounter data
        train_json_path = os.path.join(base_dir, "train.json")
        self.train_data = utils.load_json_file(train_json_path)
            
        # Create encounter lookup for faster access
        self.encounter_lookup = {item["encounter_id"]: item for item in self.train_data}
        
        # Load CVQA data (answers)
        cvqa_path = os.path.join(base_dir, "train_cvqa.json")
        self.cvqa_data = utils.load_json_file(cvqa_path)
    
    def get_encounter_ids(self, limit=None):
        """Get list of encounter IDs, optionally limited."""
        encounter_ids = list(self.encounter_lookup.keys())
        if limit:
            return encounter_ids[:limit]
        return encounter_ids
    
    def get_encounter_data(self, encounter_id):
        """Get data for a specific encounter."""
        return self.encounter_lookup.get(encounter_id)
    
    def get_cvqa_data(self, encounter_id):
        """Get CVQA data for a specific encounter."""
        for item in self.cvqa_data:
            if item["encounter_id"] == encounter_id:
                return item
        return None
    
    def get_question_data(self, qid):
        """Get question data for a specific question ID."""
        return self.questions_lookup.get(qid)
    
    def process_encounter(self, encounter_id):
        """Process a single encounter and its questions on-the-fly."""
        
        # Get encounter data
        encounter_data = self.get_encounter_data(encounter_id)
        if not encounter_data:
            return []
        
        # Get CVQA data (answers)
        cvqa_data = self.get_cvqa_data(encounter_id)
        if not cvqa_data:
            return []
        
        results = []
        
        # Process each question for this encounter
        for qid, answer_index in cvqa_data.items():
            if qid == "encounter_id":
                continue
                
            # Get question data
            question_data = self.get_question_data(qid)
            if not question_data:
                continue
            
            # Create image paths
            image_paths = [os.path.join(self.images_dir, img_id) 
                          for img_id in encounter_data["image_ids"]]
            
            # Verify images exist
            valid_images = []
            for img_path in image_paths:
                if os.path.exists(img_path) and utils.is_valid_image(img_path):
                    valid_images.append(img_path)
                        
            if not valid_images:
                continue
                
            # Get answer text
            options = question_data.get("options_en", [])
            try:
                answer_text = options[answer_index]
            except (IndexError, TypeError):
                answer_text = None
                
            # Format options text
            options_text = ", ".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            
            # Combine context
#             clinical_context = f"{encounter_data.get('query_title_en', '')}: {encounter_data.get('query_content_en', '')}"
#             metadata = f"Type: {question_data.get('question_type_en', '')}, Category: {question_data.get('question_category_en', '')}"
#             query_text = f"Clinical Context: {clinical_context}\nQuestion: Based on the image, {question_data.get('question_en', '')}\nQuestion Metadata: {metadata}\nOptions: {options_text}"
            
            metadata = f"Type: {question_data.get('question_type_en', '')}, Category: {question_data.get('question_category_en', '')}"
            query_text = f"Question: Based on the image, {question_data.get('question_en', '')}\nQuestion Metadata: {metadata}\nOptions: {options_text}"

            
            results.append({
                "encounter_id": encounter_id,
                "qid": qid,
                "query_text": query_text,
                "image_paths": valid_images,
                "answer_text": answer_text,
                "answer_index": answer_index
            })
        
        return results
    
    def process_batch(self, encounter_ids, batch_idx, save_dir):
        """Process a batch of encounters and save to disk."""
        utils.ensure_directory(save_dir)
        batch_data = []
        
        for encounter_id in tqdm(encounter_ids, desc=f"Processing batch {batch_idx}"):
            batch_data.extend(self.process_encounter(encounter_id))
            
        if batch_data:
            batch_file = os.path.join(save_dir, f"batch_{batch_idx}.pkl")
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
                
        return len(batch_data)
    
    def process_dataset(self, batch_size=5, limit=None, save_dir="processed_data"):
        """Process the entire dataset in batches."""
        encounter_ids = self.get_encounter_ids(limit)
        total_processed = 0
        
        utils.ensure_directory(save_dir)
        
        for i in range(0, len(encounter_ids), batch_size):
            batch_encounter_ids = encounter_ids[i:i+batch_size]
            batch_idx = i // batch_size
            
            processed = self.process_batch(batch_encounter_ids, batch_idx, save_dir)
            total_processed += processed
            
            logger.info(f"Processed {total_processed} examples after batch {batch_idx+1}")
            
        return total_processed
    
    def process_from_csv(self, csv_file, batch_size=5, limit=None, save_dir="processed_data"):
        """Process dataset from a CSV file (alternative method)."""
        # Load CSV
        df = pd.read_csv(csv_file)
        
        # Apply limit if specified
        if limit:
            df = df.head(limit)
        
        # Convert string representations of lists to actual lists
        if 'image_ids' in df.columns:
            df['image_ids'] = df['image_ids'].apply(eval)
        
        if 'responses_en' in df.columns:
            df['responses_en'] = df['responses_en'].apply(eval)
            
        # Filter to relevant columns
        df = df[['encounter_id', 'qid', 'question_en', 'options_en', 'answer_text', 'image_ids']]
        
        total_processed = 0
        utils.ensure_directory(save_dir)
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_idx = i // batch_size
            
            processed = self._process_csv_batch(batch_df, batch_idx, save_dir)
            total_processed += processed
            
            logger.info(f"Processed {total_processed} examples after batch {batch_idx+1}")
            
        return total_processed
    
    def _process_csv_batch(self, batch_df, batch_idx, save_dir):
        """Process a batch from CSV data."""
        batch_data = []
        
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx}"):
            try:
                image_paths = [os.path.join(self.images_dir, img_id) for img_id in row['image_ids']]
                
                valid_images = []
                for img_path in image_paths:
                    if os.path.exists(img_path) and utils.is_valid_image(img_path):
                        valid_images.append(img_path)
                        
                if not valid_images:
                    continue
                
                options_text = ", ".join([f"{i+1}. {opt}" for i, opt in enumerate(eval(row['options_en']))])
                query_text = f"Question: {row['question_en']} Options: {options_text}"
                
                batch_data.append({
                    "encounter_id": row['encounter_id'],
                    "qid": row['qid'],
                    "query_text": query_text,
                    "image_paths": valid_images,
                    "answer_text": row['answer_text']
                })
            
            except Exception as e:
                logger.error(f"Error processing row: {e}")
        
        if batch_data:
            batch_file = os.path.join(save_dir, f"batch_{batch_idx}.pkl")
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
        
        return len(batch_data)
    
    def print_example(self, encounter_id=None):
        """Print a sample of what we're providing to the model."""
        if encounter_id is None:
            # Get first encounter ID
            encounter_id = self.get_encounter_ids(limit=1)[0]
        
        # Process the encounter
        results = self.process_encounter(encounter_id)
        
        if not results:
            logger.info("No valid examples found for this encounter")
            return
        
        example = results[0]
        
        logger.info("\n=== EXAMPLE INPUT FOR INFERENCE ===")
        logger.info(f"Encounter ID: {example['encounter_id']}")
        logger.info(f"Question ID: {example['qid']}")
        logger.info(f"System message: You are an AI assistant answering medical questions based on clinical images.")
        logger.info(f"User query: {example['query_text']}")
        logger.info(f"Number of images: {len(example['image_paths'])}")
        logger.info(f"Image paths: {example['image_paths']}")
        logger.info(f"Ground truth answer: {example['answer_text']}")
        
        # Show how it would look formatted
        logger.info("\nThis would be formatted as:")
        formatted_prompt = "<start_of_turn>system\nYou are an AI assistant answering medical questions based on clinical images.\n<end_of_turn>\n<start_of_turn>user\n" + example['query_text'] + "\n<image>" * len(example['image_paths']) + "\n<end_of_turn>\n<start_of_turn>model"
        logger.info(formatted_prompt)
        
        return example