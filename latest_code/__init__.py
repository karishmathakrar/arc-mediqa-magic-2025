"""
Medical Vision AI Pipeline Package

This package provides modular components for medical image analysis including:
- Fine-tuning vision-language models
- Reasoning and analysis pipelines  
- RAG-based knowledge retrieval
- Evaluation and submission utilities
"""

from .finetuning_pipeline import FineTuningPipeline

__version__ = "1.0.0"
__all__ = [
    "FineTuningPipeline"
]

# For backward compatibility, also export individual components
try:
    from .evaluation_script import MedicalEvaluator
    __all__.append("MedicalEvaluator")
except ImportError:
    pass
