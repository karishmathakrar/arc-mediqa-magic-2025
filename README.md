# MEDIQA-Magic 2025: Medical Image Question Answering Pipeline
## Applied Research Competitions at Georgia Tech

A comprehensive medical image analysis pipeline for dermatological question answering using multi-modal large language models, featuring fine-tuning, reasoning layers, and agentic RAG systems.

## 🏗️ Repository Structure

```
arc-mediqa-magic-2025/
├── 📁 Core Pipeline Components
│   ├── finetuning_pipeline/           # Modular fine-tuning pipeline
│   │   ├── pipeline.py                # Complete training and inference pipeline
│   │   ├── finetuning_pipeline_example_usage.py  # Usage examples
│   │   └── __init__.py                # Package initialization
│   ├── reasoning_pipeline/            # Reasoning layer implementation
│   │   ├── reasoning_pipeline.py      # Gemini-based reasoning system
│   │   ├── reasoning_pipeline_example_usage.py   # Usage examples
│   │   └── __init__.py                # Package initialization
│   ├── rag_pipeline/                  # RAG system implementation
│   │   ├── rag_pipeline.py            # Retrieval-Augmented Generation
│   │   ├── rag_pipeline_example_usage.py         # Usage examples
│   │   └── __init__.py                # Package initialization
│   └── evaluation/                    # Official evaluation scripts
│       ├── run_cvqa_eval.py           # CVQA evaluation runner
│       ├── run_segandcvqa_scoring.py  # Combined scoring
│       └── score_cvqa.py              # CVQA scoring utilities
├── 📁 Standalone Scripts
│   ├── data_preprocessor.py           # Data preprocessing utilities
│   ├── evaluation_script.py           # Evaluation utilities
│   └── submission_utility.py          # Submission formatting
├── 📁 Data & Datasets
│   ├── 2025_dataset/                  # Competition dataset
│   │   ├── train/                     # Training data and images
│   │   ├── valid/                     # Validation data and images
│   │   └── test/                      # Test data and images
│   ├── 2024_dataset/                  # Previous year dataset
│   └── outputs/                       # Generated outputs and models
└── 📁 Configuration
    ├── requirements.txt               # Python dependencies
    ├── .env                          # Environment variables
    ├── .gitignore                    # Git ignore rules
    └── .pre-commit-config.yaml       # Pre-commit hooks
```

## 🚀 Quick Start

### Prerequisites

1. **Clone the repository:**
```bash
git clone https://github.com/karishmathakrar/arc-mediqa-magic-2025.git
cd arc-mediqa-magic-2025
```

2. **Set up environment:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure API keys in `.env`:**
```env
HF_TOKEN=your_huggingface_token_here
API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Alternative key name
```

### Basic Usage

1. **Run fine-tuning pipeline:**
```python
from finetuning_pipeline.pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(base_dir="./")
train_df, val_df = pipeline.prepare_data()
trainer = pipeline.train()
predictions = pipeline.run_inference()
```

2. **Run reasoning pipeline:**
```python
from reasoning_pipeline import ReasoningConfig, ReasoningPipeline

config = ReasoningConfig(use_finetuning=True, base_dir="./")
pipeline = ReasoningPipeline(config)
results = pipeline.process_all_encounters()
```

## 📋 Core Components

### 🔧 Fine-tuning Pipeline (`finetuning_pipeline/`)

**Main Files:**
- `pipeline.py` - Complete fine-tuning pipeline with modular components
- `finetuning_pipeline_example_usage.py` - Comprehensive usage examples
- `__init__.py` - Package initialization

**Key Features:**
- Support for multiple vision-language models (see supported models below)
- LoRA fine-tuning with PEFT
- Automated data preprocessing and validation
- Base and fine-tuned model inference
- Prediction aggregation and evaluation formatting
- Token analysis and dataset inspection tools

**Usage Example:**
```python
from finetuning_pipeline.pipeline import FineTuningPipeline

# Initialize pipeline
pipeline = FineTuningPipeline(
    model_name="Qwen2-VL-2B-Instruct",
    base_dir="./",
    output_dir="./outputs"
)

# Train model
trainer = pipeline.train(test_mode=True)

# Run inference
predictions_df, aggregated_df, formatted_predictions = pipeline.run_inference()
```

### 🧠 Reasoning Pipeline (`reasoning_pipeline/`)

**Main Files:**
- `reasoning_pipeline.py` - Gemini-based reasoning system for medical analysis
- `reasoning_pipeline_example_usage.py` - Usage examples and configurations
- `__init__.py` - Package initialization

**Key Features:**
- Structured dermatological image analysis
- Clinical context extraction and processing
- Multi-evidence reasoning for question answering
- Support for both validation and test datasets
- Configurable Gemini model selection
- Intermediate result saving with customizable frequency

**Usage Example:**
```python
from reasoning_pipeline import ReasoningConfig, ReasoningPipeline

# Configure pipeline
config = ReasoningConfig(
    use_finetuning=True,
    use_test_dataset=False,
    base_dir="./",
    output_dir="./outputs"
)

# Process encounters
pipeline = ReasoningPipeline(config)
results = pipeline.process_all_encounters()
```

### 🔍 RAG Pipeline (`rag_pipeline/`)

**Main Files:**
- `rag_pipeline.py` - Retrieval-Augmented Generation system
- `rag_pipeline_example_usage.py` - Usage examples
- `__init__.py` - Package initialization

**Key Features:**
- Vector database integration with LanceDB
- Diagnosis-based knowledge retrieval
- Hybrid semantic and keyword search
- Self-reflection and answer refinement
- Medical knowledge base management

## 🛠️ Standalone Scripts

### `data_preprocessor.py`
**Purpose:** Data preprocessing utilities and transformations

**Key Features:**
- Dataset cleaning and normalization
- Image path validation and fixing
- Question and option processing
- Batch file generation for inference

### `evaluation_script.py`
**Purpose:** Model evaluation and scoring utilities

**Key Features:**
- Prediction accuracy calculation
- Performance metrics computation
- Result comparison and analysis
- Official evaluation format validation

### `submission_utility.py`
**Purpose:** Format predictions for competition submission

**Key Features:**
- Official submission format generation
- Prediction validation and error checking
- Multiple model result aggregation
- Submission file packaging

## 🤖 Supported Models

The pipeline supports the following vision-language models:

### Llama Models
- **`llama-3.2-11b-vision`** - Meta Llama 3.2 11B Vision Instruct
  - Model ID: `meta-llama/Llama-3.2-11B-Vision-Instruct`
  - Memory: ~22GB VRAM required

### Gemma Models  
- **`gemma-3-4b-it`** - Google Gemma 3 4B Instruct
  - Model ID: `google/gemma-3-4b-it`
  - Memory: ~8GB VRAM required
- **`gemma-3-12b-it`** - Google Gemma 3 12B Instruct
  - Model ID: `google/gemma-3-12b-it`
  - Memory: ~24GB VRAM required

### Qwen Models
- **`Qwen2-VL-2B-Instruct`** - Qwen2 Vision-Language 2B (Default)
  - Model ID: `Qwen/Qwen2-VL-2B-Instruct`
  - Memory: ~4GB VRAM required
- **`Qwen2-VL-7B-Instruct`** - Qwen2 Vision-Language 7B
  - Model ID: `Qwen/Qwen2-VL-7B-Instruct`
  - Memory: ~14GB VRAM required
- **`Qwen2.5-VL-3B-Instruct`** - Qwen2.5 Vision-Language 3B
  - Model ID: `Qwen/Qwen2.5-VL-3B-Instruct`
  - Memory: ~6GB VRAM required
- **`Qwen2.5-VL-7B-Instruct`** - Qwen2.5 Vision-Language 7B
  - Model ID: `Qwen/Qwen2.5-VL-7B-Instruct`
  - Memory: ~14GB VRAM required

**Model Selection:**
```python
# Available models can be retrieved programmatically
from finetuning_pipeline.pipeline import FineTuningPipeline
pipeline = FineTuningPipeline()
available_models = pipeline.get_available_models()
print(available_models)
```

## 📊 Data Structure

### Dataset Organization
```
2025_dataset/
├── train/
│   ├── train.json              # Training questions and metadata
│   ├── train_cvqa.json         # CVQA format training data
│   ├── option_maps.json        # Answer option mappings
│   ├── closedquestions_definitions_imageclef2025.json  # Question definitions
│   └── images_train/           # Training images
├── valid/
│   ├── valid.json              # Validation questions
│   ├── valid_cvqa.json         # CVQA format validation data
│   └── images_valid/           # Validation images
└── test/
    ├── test.json               # Test questions
    └── images_test/            # Test images (if available)
```

### Output Structure
```
outputs/
├── processed_val/              # Processed validation batches
├── processed_val_fixed/        # Fixed image path batches
├── processed_train_data-{model}-V3/     # Processed training data
├── processed_combined_data-{model}-V3/  # Combined train+val data
├── processed_test_data-{model}-V3/      # Processed test data
├── val_dataset.csv            # Generated validation dataset
├── finetuned-model/           # Fine-tuned model checkpoints
├── merged/                    # Merged model files
├── *_predictions_*.csv        # Raw prediction files
├── *_aggregated_*.csv         # Aggregated predictions
└── *_data_cvqa_sys_*.json     # Evaluation-ready predictions
```

## 🔧 Configuration

### Environment Variables (`.env`)
```env
# Hugging Face token for model access
HF_TOKEN=your_huggingface_token_here

# Gemini API key for reasoning pipeline
API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Alternative

# Optional: Custom model cache directory
HF_HOME=/path/to/huggingface/cache
```

### Model Configuration
Models are automatically configured based on selection. The pipeline handles:
- Quantization settings (4-bit with NF4)
- Attention mechanisms (Flash Attention 2 where supported)
- Device mapping for multi-GPU setups
- Memory optimization settings

## 📈 Pipeline Workflows

### 1. Complete Training and Inference Workflow
```python
# 1. Initialize pipeline
from finetuning_pipeline.pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(
    model_name="Qwen2-VL-2B-Instruct",
    base_dir="./",
    validate_paths=True
)

# 2. Prepare data
train_df, val_df = pipeline.prepare_data(use_combined=False)

# 3. Train model
trainer = pipeline.train(test_mode=False)

# 4. Run inference
predictions_df, aggregated_df, formatted_predictions = pipeline.run_inference()
```

### 2. Reasoning-Enhanced Workflow
```python
# 1. Generate base predictions (from above)
# 2. Apply reasoning layer
from reasoning_pipeline import ReasoningConfig, ReasoningPipeline

config = ReasoningConfig(
    use_finetuning=True,
    base_dir="./",
    output_dir="./outputs"
)

reasoning_pipeline = ReasoningPipeline(config)
reasoning_results = reasoning_pipeline.process_all_encounters()
```

### 3. RAG-Enhanced Workflow
```python
# 1. Generate predictions and reasoning (from above)
# 2. Apply RAG system
from rag_pipeline.rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline(base_dir="./")
enhanced_results = rag_pipeline.process_with_knowledge_retrieval(reasoning_results)
```

## 🧪 Testing and Validation

### Running Tests
```bash
# Test fine-tuning pipeline
python finetuning_pipeline/finetuning_pipeline_example_usage.py

# Test reasoning pipeline
python reasoning_pipeline/reasoning_pipeline_example_usage.py

# Test RAG pipeline
python rag_pipeline/rag_pipeline_example_usage.py
```

### Validation Checks
- Image path validation and automatic fixing
- Dataset integrity verification
- Model output format validation
- Prediction consistency checks

## 📋 Requirements

### Core Dependencies
```
# Deep Learning Framework
torch>=2.6.0
transformers @ git+https://github.com/huggingface/transformers@b6d65e40b256d98d9621707762b94bc8ad83b7a7
accelerate>=1.4.0
peft>=0.14.0
trl>=0.15.2
bitsandbytes>=0.45.3

# Vision-Language Models
qwen-vl-utils>=0.0.11
flash_attn>=2.7.4

# Data Processing
datasets>=3.3.2
pandas>=2.2.3
numpy>=2.1.2
pillow>=11.1.0

# Vector Search & RAG
lancedb>=0.22.0
sentence-transformers>=4.1.0
FlagEmbedding>=1.3.4

# API Integration
google-generativeai>=0.8.4
google-genai>=1.10.0

# Utilities
tqdm>=4.67.1
python-dotenv>=1.1.0
matplotlib>=3.10.1
seaborn>=0.13.2
```

### Hardware Requirements
- **GPU:** NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for larger models)
- **RAM:** 32GB+ system RAM recommended
- **Storage:** 50GB+ free space for models and datasets
- **CUDA:** Compatible CUDA installation (12.4+ recommended)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size in training configuration
   - Use gradient checkpointing (enabled by default)
   - Try smaller models (e.g., Qwen2-VL-2B instead of 7B)
   - Enable CPU offloading for large models

2. **Image Path Errors:**
   - Use the data preprocessing utilities in `data_preprocessor.py`
   - Check image directory structure matches expected format
   - Verify dataset paths in configuration

3. **API Rate Limits:**
   - Monitor API usage and quotas in Google Cloud Console
   - Consider adding delays between requests for large batches
   - Use the reasoning pipeline's built-in retry mechanisms

4. **Model Loading Issues:**
   - Verify HuggingFace token permissions for gated models
   - Check model availability and access rights
   - Ensure sufficient disk space for model cache
   - Clear HuggingFace cache if corrupted: `rm -rf ~/.cache/huggingface/`

5. **Transformers Version Issues:**
   - The pipeline uses a specific transformers commit for compatibility
   - Reinstall if needed: `pip install git+https://github.com/huggingface/transformers@b6d65e40b256d98d9621707762b94bc8ad83b7a7`

### Debug Mode
Enable debug logging by setting environment variables:
```bash
export TRANSFORMERS_VERBOSITY=debug
export DATASETS_VERBOSITY=debug
export CUDA_LAUNCH_BLOCKING=1  # For CUDA debugging
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the example usage files for implementation details
- Review the troubleshooting section above

## 🏆 Competition Results

This pipeline was developed for the MEDIQA 2025 competition, focusing on dermatological image analysis and medical question answering. The modular design allows for easy experimentation with different model combinations and reasoning strategies.
