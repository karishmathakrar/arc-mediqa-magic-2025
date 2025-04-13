## Setup Instructions

1. **Download data from this google folder:**
https://drive.google.com/file/d/1VOIgXVJy1c8lWFdY63lCyyztY1PS-Bh1/view?usp=sharing
Create an images folder within the 2024_dataset folder and place the train, test, valid folders within images. 

2. **Make sure you have Python 3.10 installed**  

3. **Create a .venv**
```
source .venv/bin/activate
```

4. **Install packages in requirements.txt**
```bash
pip install -r requirements.txt
```

5. **Create Gemini API key and add the following line to the .env file**
```
API_KEY=<your key here>
```

## Training
To finetune the model:
python train.py --data_dir 2025_dataset/train --output_dir models/fine_tuned_model --limit 1 --reprocess


## Inference

### Combined inference + evaluation

1) chmod +x run_full_evaluation.sh

2) Run with baseline model and no limit:
./run_full_evaluation.sh

OR run with fine-tuned model and limit of 1:
OLD: ./run_full_evaluation.sh --model_path models/fine_tuned_model --limit 1 --reprocess
NEW: ./run_full_evaluation.sh --model_path merged_model --limit 1 --reprocess

OR run with fine-tuned model and custom output directory:
OLD: ./run_full_evaluation.sh --model_path models/fine_tuned_model --output_dir evaluation_results_finetuned --reprocess
NEW: ./run_full_evaluation.sh --model_path merged_model --output_dir evaluation_results_finetuned


### Development/testing + evaluation

1) For Gemma inference using baseline model:
python inference.py --data_dir 2025_dataset/train --limit 1 --reprocess --output_file evaluation_results/inference_results.csv --json_output evaluation_results/data_cvqa_sys.json --indexed_json_output evaluation_results/data_cvqa_sys_indices.json

OR for Gemma inference using finetuned model:
python inference.py --data_dir 2025_dataset/train --model_id merged_model --limit 1 --reprocess --output_file evaluation_results/inference_results_finetuned.csv --json_output evaluation_results/data_cvqa_sys_finetuned.json --indexed_json_output evaluation_results/data_cvqa_sys_indices_finetuned.json

2) Run evaluation on converted results: 
python run_cvqa_eval.py 2025_dataset/train/train_cvqa.json evaluation_results/data_cvqa_sys_indices.json evaluation_results --reprocess

OR python run_cvqa_eval.py 2025_dataset/train/train_cvqa.json evaluation_results/data_cvqa_sys_indices_finetuned.json evaluation_results --reprocess