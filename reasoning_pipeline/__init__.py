"""
Reasoning Pipeline Package

This package provides a comprehensive reasoning-based medical analysis pipeline
that uses Gemini to analyze medical images and clinical context with structured reasoning.
"""

# Import main classes for easy access
from .reasoning_pipeline import (
    Args,
    DataLoader,
    DataProcessor,
    AgenticRAGData,
    AnalysisService,
    DermatologyPipeline,
    ReasoningConfig,
    ReasoningPipeline,
    run_all_encounters_pipeline,
    run_single_encounter_pipeline,
    main
)

__version__ = "1.0.0"
__author__ = "Medical Vision Team"

__all__ = [
    "Args",
    "DataLoader",
    "DataProcessor",
    "AgenticRAGData",
    "AnalysisService",
    "DermatologyPipeline",
    "ReasoningConfig",
    "ReasoningPipeline",
    "run_all_encounters_pipeline",
    "run_single_encounter_pipeline",
    "main"
]
