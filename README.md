# MEDIQA-Magic: Medical Image Question Answering

This repository contains code for dermatology image analysis and medical question answering using multi-modal large language models.

## Project Structure

```
mediqa-magic-v2/
├── 01-model-finetuning-inference-base.ipynb  # Core vision-language model training and inference
├── 02-reasoning-layer.ipynb                  # Reasoning layer implementation
├── 03-agentic-rag.ipynb                      # Diagnosis-based knowledge retrieval system
├── knowledge_db/                             # Vector database for dermatology knowledge
├── 2025_dataset/                             # Dataset directories (train/valid/test)
└── outputs/                                  # Model outputs and prediction results
```

## Setup


1. Clone the repository and navigate to the directory (will be made publicly available after May 30th):
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

3. Create a `.env` file with your API tokens:
```
HF_TOKEN=your_huggingface_token_here
API_KEY=your_gemini_api_key_here
```

## Pipeline Components

### 1. Model Fine-tuning and Inference Base (01-model-finetuning-inference-base.ipynb)

This notebook contains:
- Dataset processing and preparation
- Vision-language model loading (Llama, Gemma, Qwen)
- LoRA fine-tuning configuration
- Training implementation
- Inference functionality for both base and fine-tuned models
- Prediction aggregation and formatting

### 2. Reasoning Layer (02-reasoning-layer.ipynb)

This notebook implements:
- Image analysis for dermatological features
- Clinical context extraction from patient data
- Evidence integration from visual and textual sources
- Reasoning engine to determine accurate answers
- Final prediction formatting for evaluation

### 3. Agentic RAG System (03-agentic-rag.ipynb)

This notebook provides:
- Knowledge base management with LanceDB
- Diagnosis extraction from clinical data
- Query generation based on diagnoses
- Hybrid search with semantic and keyword components
- Self-reflection mechanisms for answer improvement
- Confidence-based revision cycles

## Running the Pipeline

You can run the components in order through Jupyter notebook environments. Each notebook is designed to be self-contained but builds upon the outputs of previous components:

1. First run `01-model-finetuning-inference-base.ipynb` to train models and generate base predictions
2. Then run `02-reasoning-layer.ipynb` to apply reasoning to model predictions
3. Finally run `03-agentic-rag.ipynb` to incorporate medical knowledge and reasoning

## Output Files

The pipeline produces these key outputs:

- Fine-tuned model checkpoints in `outputs/finetuned-model/`
- Merged models in `outputs/merged/`
- Prediction files in `outputs/` with prefixes:
  - `aggregated_predictions_` (validation set)
  - `aggregated_test_predictions_` (test set)
  - `data_cvqa_sys_` (formatted for evaluation)

## Requirements

Key dependencies include:
- PyTorch and transformers for model training
- Sentence-transformers and LanceDB for vector search
- Google's Generative AI for reasoning components
- Various utilities for data processing and evaluation

Refer to requirements.txt for the complete dependency list.