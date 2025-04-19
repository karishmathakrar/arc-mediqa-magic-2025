# MEDIQA-Magic: Medical Image Question Answering

This repository contains code for the MEDIQA challenge, which involves answering medical questions based on clinical images using multi-modal language models.

## Project Structure

```
mediqa-magic-v2/
├── config.py               # Configuration settings
├── preprocess.py           # Data preprocessing script
├── train.py                # Training script
├── inference.py            # Inference script
├── evaluate.py             # Evaluation script
├── utils.py                # Utility functions
├── run.py                  # One-click pipeline script
├── evaluation/             # Conference-provided evaluation scripts
├── 2025_dataset/           # Raw data (not included in repo)
├── processed_data/         # Processed data (generated)
└── outputs/                # Model outputs and evaluation results
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mediqa-magic-v2.git
cd mediqa-magic-v2
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your HuggingFace token:
```
HF_TOKEN=your_huggingface_token_here

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Workflow

### 1. Preprocess the Data

Preprocess data for training:
```bash
python preprocess.py --mode train
```

Preprocess data for inference:
```bash
python preprocess.py --mode inference
```

Using validation data: 
```bash
python preprocess.py --mode inference --data_dir 2025_dataset/validation --output_dir processed_data/validation
```

Options:
- `--limit N`: Limit to N examples
- `--reprocess`: Force reprocessing even if data exists
- `--batch_size N`: Process in batches of N examples
- `--use_single_image`: Use only the first image for each encounter (default: True)

### 2. Train the Model

Fine-tune the model using LoRA:
```bash
python train.py
```

Options:
- `--model_id MODEL_ID`: Base model ID (default: "google/gemma-3-4b-it")
- `--batch_size N`: Batch size for training
- `--grad_accum N`: Gradient accumulation steps
- `--epochs N`: Number of training epochs
- `--lr RATE`: Learning rate
- `--skip_merge`: Skip merging LoRA weights with base model

### 3. Run Inference

Run inference with both base and fine-tuned models:
```bash
python inference.py
```

To do inference on validation data: 
python inference.py --processed_dir processed_data/validation --output_dir outputs/validation

Options:
- `--skip_base`: Skip inference with base model
- `--skip_finetuned`: Skip inference with fine-tuned model
- `--limit N`: Limit to N examples
- `--max_new_tokens N`: Maximum new tokens to generate
- `--temperature T`: Sampling temperature

### 4. Evaluate Results

Compare the performance of base and fine-tuned models:
```bash
python evaluate.py
```

To do evaluation on validation data: 
python evaluate.py --reference_file 2025_dataset/validation/validation_cvqa.json --base_prediction_file outputs/validation/base_model/results.json --finetuned_prediction_file outputs/validation/finetuned_model/results.json

Options:
- `--skip_base`: Skip evaluation of base model
- `--skip_finetuned`: Skip evaluation of fine-tuned model


### TLDR of steps above; 
1) python preprocess.py --mode train --reprocess
2) python train.py
3) python preprocess.py --mode inference --data_dir 2025_dataset/validation --output_dir processed_data/validation
4) python inference.py --processed_dir processed_data/validation --output_dir outputs/validation
5) python evaluate.py --reference_file 2025_dataset/validation/validation_cvqa.json --base_prediction_file outputs/validation/base_model/results.json --finetuned_prediction_file outputs/validation/finetuned_model/results.json


### 5. Run the Complete Pipeline

To run the entire pipeline with a single command:
```bash
python run.py --use_validation
```

Options:
- `--skip_preprocessing`: Skip data preprocessing
- `--skip_training`: Skip model training
- `--skip_inference`: Skip model inference
- `--skip_evaluation`: Skip result evaluation
- `--limit N`: Limit to N examples for faster testing


## Output Files

After running the pipeline, you'll find the following outputs:

1. **Processed Data**:
   - `processed_data/train/`: Processed training data
   - `processed_data/inference/`: Processed inference data

2. **Models**:
   - `outputs/finetuned_model/`: Fine-tuned model with LoRA weights
   - `outputs/merged_model/`: Merged model (base + LoRA weights)

3. **Inference Results**:
   - `outputs/base_model/results.json`: Base model predictions
   - `outputs/finetuned_model/results.json`: Fine-tuned model predictions

4. **Evaluation Results**:
   - `outputs/comparison/model_comparison.json`: Detailed comparison metrics
   - `outputs/comparison/model_comparison.md`: Markdown report with tables
   - `outputs/base_model/evaluation/scores_cvqa.json`: Base model evaluation
   - `outputs/finetuned_model/evaluation/scores_cvqa.json`: Fine-tuned model evaluation

## Reference

If you use this code, please cite the original MEDIQA challenge paper:
```
[Citation information will go here]
```