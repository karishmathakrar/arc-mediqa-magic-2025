#!/usr/bin/env python3
"""
Submission Package Creator
Creates submission packages for medical image analysis competition.
"""

import os
import json
import shutil
import zipfile
import datetime


def create_submission_package(input_json, output_dir="outputs", model_id="reasoned_all"):
    """
    Create a submission package from prediction JSON file.
    
    Args:
        input_json: Path to the input JSON file with predictions
        output_dir: Directory to save the submission package
        model_id: Identifier for the model used
    
    Returns:
        Path to the created zip file
    """
    # Create timestamp for unique submission
    submission_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create submission directory structure
    submission_dir = os.path.join(output_dir, f"submission_{model_id}_{submission_timestamp}")
    os.makedirs(submission_dir, exist_ok=True)
    
    # Create empty masks_preds directory (required by competition format)
    submission_masks_dir = os.path.join(submission_dir, "masks_preds")
    os.makedirs(submission_masks_dir, exist_ok=True)
    
    # Copy the JSON file to the submission directory with required name
    dest_json = os.path.join(submission_dir, "data_cvqa_sys.json")
    shutil.copy2(input_json, dest_json)
    
    # Create the zip file
    zip_path = os.path.join(output_dir, f"mysubmission_{model_id}_{submission_timestamp}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(dest_json, arcname="data_cvqa_sys.json")
        # Add the empty directory to the zip
        zipf.writestr("masks_preds/", "")
    
    # Print summary
    print(f"Submission package created at: {zip_path}")
    print(f"Files included:")
    print(f" - data_cvqa_sys.json (copied from {input_json})")
    print(f" - masks_preds/ (empty directory)")
    
    return zip_path


def main():
    """Main function to create submission package."""
    # Default configuration - modify as needed
    input_json = "outputs/test_data_cvqa_sys_diagnosis_based_all_20250503_103452.json"
    output_dir = "outputs"
    model_id = "reasoned_all"
    
    # Check if input file exists
    if not os.path.exists(input_json):
        print(f"Error: Input file {input_json} does not exist!")
        print("Please check the file path and try again.")
        return 1
    
    try:
        # Create the submission package
        zip_path = create_submission_package(input_json, output_dir, model_id)
        print(f"\nSubmission package successfully created: {zip_path}")
        return 0
        
    except Exception as e:
        print(f"Error creating submission package: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
