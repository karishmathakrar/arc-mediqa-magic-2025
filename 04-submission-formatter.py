import os
import json
import shutil
import zipfile
import datetime

input_json = "outputs/test_data_cvqa_sys_diagnosis_based_all_20250503_103452.json"
output_dir = "outputs"
model_id = "reasoned_all"
submission_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_dir, exist_ok=True)
submission_dir = os.path.join(output_dir, f"submission_{model_id}_{submission_timestamp}")
os.makedirs(submission_dir, exist_ok=True)
submission_masks_dir = os.path.join(submission_dir, "masks_preds")
os.makedirs(submission_masks_dir, exist_ok=True)
dest_json = os.path.join(submission_dir, "data_cvqa_sys.json")
shutil.copy2(input_json, dest_json)
zip_path = os.path.join(output_dir, f"mysubmission_{model_id}_{submission_timestamp}.zip")
with zipfile.ZipFile(zip_path, 'w') as zipf:
    zipf.write(dest_json, arcname="data_cvqa_sys.json")
    zipf.write(submission_masks_dir, arcname="masks_preds")
    
print(f"Submission package created at: {zip_path}")
print(f"Files included:")
print(f" - data_cvqa_sys.json (copied from {input_json})")
print(f" - masks_preds/ (empty directory)")