#!/usr/bin/env python3
"""
Diagnosis-based RAG Medical Analysis Pipeline
Advanced agentic system with knowledge retrieval, self-reflection, and multi-cycle reasoning.
"""

import os
import glob
import pandas as pd
import ast
import re
from collections import defaultdict
import json
import datetime
import time
import traceback
from PIL import Image
from dotenv import load_dotenv
from google import genai
import numpy as np
import lancedb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch
import random
from typing import Dict, List, Tuple, Optional, Any


class Args:
    """Configuration arguments for the RAG pipeline."""
    
    def __init__(self, use_finetuning=True, use_test_dataset=True, base_dir=None, output_dir=None,
                 model_predictions_dir=None, images_dir=None, dataset_path=None, gemini_model=None,
                 max_reflection_cycles=None, confidence_threshold=None, knowledge_db_path=None,
                 embedding_model=None, cross_encoder_model=None, vector_dimension=None,
                 top_k_semantic=None, top_k_keyword=None, top_k_hybrid=None, top_k_rerank=None,
                 dataset_name_huggingface=None, question_type_retrieval_config=None, default_rag_config=None):
        """
        Initialize arguments with options for dataset and model type.
        
        Parameters:
        - use_finetuning: Whether to use the fine-tuned model predictions (True) or base model predictions (False)
        - use_test_dataset: Whether to use the test dataset (True) or validation dataset (False)
        - base_dir: Base directory for the project (defaults to current working directory)
        - output_dir: Output directory for results (defaults to base_dir/outputs)
        - model_predictions_dir: Directory containing model predictions (defaults to output_dir/05022025)
        - images_dir: Directory containing images (auto-determined based on dataset if not provided)
        - dataset_path: Path to dataset CSV file (auto-determined based on dataset if not provided)
        - gemini_model: Gemini model to use (defaults to gemini-2.0-flash-exp-2025-01-29)
        - max_reflection_cycles: Maximum number of reflection cycles (defaults to 2)
        - confidence_threshold: Confidence threshold for reflection (defaults to 0.75)
        - knowledge_db_path: Path to knowledge database (defaults to base_dir/knowledge_db)
        - embedding_model: Embedding model for semantic search (defaults to BioBERT)
        - cross_encoder_model: Cross-encoder model for reranking (defaults to ms-marco-MiniLM)
        - vector_dimension: Vector dimension for embeddings (defaults to 768)
        - top_k_semantic: Top K results for semantic search (defaults to 7)
        - top_k_keyword: Top K results for keyword search (defaults to 7)
        - top_k_hybrid: Top K results for hybrid search (defaults to 10)
        - top_k_rerank: Top K results after reranking (defaults to 5)
        - dataset_name_huggingface: HuggingFace dataset name (defaults to Skin_diseases_and_care)
        - question_type_retrieval_config: Configuration for question type retrieval (defaults to predefined config)
        - default_rag_config: Default RAG configuration (defaults to {"use_rag": True, "weight": 0.4})
        """
        self.use_finetuning = use_finetuning
        self.use_test_dataset = use_test_dataset
        
        # Set base directory
        self.base_dir = base_dir or os.getcwd()
        
        # Set output directory
        self.output_dir = output_dir or os.path.join(self.base_dir, "outputs")
        
        # Set model predictions directory
        self.model_predictions_dir = model_predictions_dir or os.path.join(self.output_dir, "05022025")
        
        # Set dataset-specific configurations
        if self.use_test_dataset:
            self.dataset_name = "test"
            self.dataset_path = dataset_path or os.path.join(self.output_dir, "test_dataset.csv")
            self.images_dir = images_dir or os.path.join(self.base_dir, "2025_dataset", "test", "images_test")
            self.prediction_prefix = "aggregated_test_predictions_"
        else:
            self.dataset_name = "validation"
            self.dataset_path = dataset_path or os.path.join(self.output_dir, "val_dataset.csv")
            self.images_dir = images_dir or os.path.join(self.base_dir, "2025_dataset", "valid", "images_valid")
            self.prediction_prefix = "aggregated_predictions_"
        
        self.model_type = "finetuned" if self.use_finetuning else "base"
        
        # Model and processing configuration
        self.gemini_model = gemini_model or "gemini-2.0-flash-exp-2025-01-29"
        self.max_reflection_cycles = max_reflection_cycles or 2
        self.confidence_threshold = confidence_threshold or 0.75
        
        # Knowledge base configuration
        self.knowledge_db_path = knowledge_db_path or os.path.join(self.base_dir, "knowledge_db")
        self.embedding_model = embedding_model or "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        self.cross_encoder_model = cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.vector_dimension = vector_dimension or 768
        self.top_k_semantic = top_k_semantic or 7
        self.top_k_keyword = top_k_keyword or 7
        self.top_k_hybrid = top_k_hybrid or 10
        self.top_k_rerank = top_k_rerank or 5
        
        self.dataset_name_huggingface = dataset_name_huggingface or "brucewayne0459/Skin_diseases_and_care"
        
        # Question type retrieval configuration
        self.question_type_retrieval_config = question_type_retrieval_config or {
            "Site Location": {"use_rag": False, "weight": 0.2},
            "Lesion Color": {"use_rag": False, "weight": 0.2},
            "Size": {"use_rag": False, "weight": 0.1},
            "Skin Description": {"use_rag": True, "weight": 0.3},
            "Onset": {"use_rag": True, "weight": 0.4},
            "Itch": {"use_rag": True, "weight": 0.4},
            "Extent": {"use_rag": False, "weight": 0.2},
            "Treatment": {"use_rag": True, "weight": 0.7},
            "Lesion Evolution": {"use_rag": True, "weight": 0.5},
            "Texture": {"use_rag": True, "weight": 0.3},
            "Lesion Count": {"use_rag": False, "weight": 0.1},
            "Differential": {"use_rag": True, "weight": 0.8},
            "Specific Diagnosis": {"use_rag": True, "weight": 0.8},
        }
        
        self.default_rag_config = default_rag_config or {"use_rag": True, "weight": 0.4}
        
        print(f"\nRAG Pipeline Configuration initialized:")
        print(f"- Base directory: {self.base_dir}")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Using {'test' if self.use_test_dataset else 'validation'} dataset")
        print(f"- Looking for {self.model_type} model predictions")
        print(f"- Dataset path: {self.dataset_path}")
        print(f"- Images directory: {self.images_dir}")
        print(f"- Model predictions directory: {self.model_predictions_dir}")
        print(f"- Prediction file prefix: {self.prediction_prefix}")
        print(f"- Gemini model: {self.gemini_model}")
        print(f"- Knowledge DB path: {self.knowledge_db_path}")
        print(f"- Max reflection cycles: {self.max_reflection_cycles}")
        print(f"- Confidence threshold: {self.confidence_threshold}")


class DataLoader:
    """Handles loading and processing of model predictions and validation data."""
    
    @staticmethod
    def get_latest_aggregated_files(args):
        """Get the latest aggregated prediction files for each model."""
        pattern = os.path.join(args.model_predictions_dir, f"{args.prediction_prefix}*_{args.model_type}_*.csv")
        
        agg_files = glob.glob(pattern)
        
        if len(agg_files) == 0:
            return []
        
        latest_files = {}
        
        for file_path in agg_files:
            file_name = os.path.basename(file_path)
            
            parts = file_name.split(f"_{args.model_type}_")
            if len(parts) != 2:
                print(f"Warning: Unexpected filename format: {file_name}")
                continue
            
            model_part = parts[0].replace(args.prediction_prefix, "")
            model_name = model_part
            
            timestamps = re.findall(r'(\d+)', parts[1])
            if len(timestamps) < 2:
                print(f"Warning: Could not find timestamps in {file_name}")
                continue
            
            timestamp = int(timestamps[1])
            
            if model_name not in latest_files or timestamp > latest_files[model_name]['timestamp']:
                latest_files[model_name] = {
                    'file_path': file_path,
                    'timestamp': timestamp
                }
        
        return [info['file_path'] for _, info in latest_files.items()]
    
    @staticmethod
    def load_all_model_predictions(args):
        """Load all model predictions from aggregated files."""
        latest_files = DataLoader.get_latest_aggregated_files(args)
        
        if not latest_files:
            print("No aggregated prediction files found. Cannot proceed.")
            return {}
        
        model_predictions = {}
        
        for file_path in latest_files:
            file_name = os.path.basename(file_path)
            
            parts = file_name.split(f"_{args.model_type}_")
            if len(parts) != 2:
                print(f"Warning: Unexpected filename format: {file_name}")
                continue
                
            model_name = parts[0].replace(args.prediction_prefix, "")
            
            try:
                df = pd.read_csv(file_path)
                df['model_name'] = model_name
                model_predictions[model_name] = df
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return model_predictions

    @staticmethod
    def load_validation_dataset(args):
        """Load the validation dataset."""
        val_df = pd.read_csv(args.dataset_path)
        
        val_df = DataLoader.process_validation_dataset(val_df)
        
        encounter_question_data = defaultdict(lambda: {
            'images': [],
            'data': None
        })
        
        for _, row in val_df.iterrows():
            encounter_id = row['encounter_id']
            base_qid = row['base_qid']
            key = (encounter_id, base_qid)
            
            if 'image_path' in row and row['image_path']:
                encounter_question_data[key]['images'].append(row['image_path'])
            elif 'image_id' in row and row['image_id']:
                image_path = os.path.join(args.images_dir, row['image_id'])
                encounter_question_data[key]['images'].append(image_path)
            
            if encounter_question_data[key]['data'] is None:
                encounter_question_data[key]['data'] = row.to_dict()
        
        grouped_data = []
        for (encounter_id, base_qid), data in encounter_question_data.items():
            entry = data['data'].copy()
            entry['all_images'] = data['images']
            entry['encounter_id'] = encounter_id
            entry['base_qid'] = base_qid
            grouped_data.append(entry)
        
        return pd.DataFrame(grouped_data)
    
    @staticmethod
    def safe_convert_options(options_str):
        """Safely convert a string representation of a list to an actual list."""
        if not isinstance(options_str, str):
            return options_str
            
        try:
            return ast.literal_eval(options_str)
        except (SyntaxError, ValueError):
            if options_str.startswith('[') and options_str.endswith(']'):
                return [opt.strip().strip("'\"") for opt in options_str[1:-1].split(',')]
            elif ',' in options_str:
                return [opt.strip() for opt in options_str.split(',')]
            else:
                return [options_str]
    
    @staticmethod
    def process_validation_dataset(val_df):
        """Process and clean the validation dataset."""
        if 'options_en' in val_df.columns:
            val_df['options_en'] = val_df['options_en'].apply(DataLoader.safe_convert_options)
            
            def clean_options(options):
                if not isinstance(options, list):
                    return options
                    
                cleaned_options = []
                for opt in options:
                    if isinstance(opt, str):
                        cleaned_opt = opt.strip("'\" ").replace(" (please specify)", "")
                        cleaned_options.append(cleaned_opt)
                    else:
                        cleaned_options.append(str(opt).strip("'\" "))
                return cleaned_options
                
            val_df['options_en_cleaned'] = val_df['options_en'].apply(clean_options)
        
        if 'question_text' in val_df.columns:
            val_df['question_text_cleaned'] = val_df['question_text'].apply(
                lambda q: q.replace(" Please specify which affected area for each selection.", "") 
                          if isinstance(q, str) and "Please specify which affected area for each selection" in q 
                          else q
            )
            
            val_df['question_text_cleaned'] = val_df['question_text_cleaned'].apply(
                lambda q: re.sub(r'^\d+\s+', '', q) if isinstance(q, str) else q
            )
        
        if 'base_qid' not in val_df.columns and 'qid' in val_df.columns:
            val_df['base_qid'] = val_df['qid'].apply(
                lambda q: q.split('-')[0] if isinstance(q, str) and '-' in q else q
            )
        
        return val_df


class DataProcessor:
    """Handles data processing for query creation."""
    
    @staticmethod
    def create_query_context(row, args=None):
        """Create query context from validation data similar to the inference process."""
        question = row.get('question_text_cleaned', row.get('question_text', 'What do you see in this image?'))
        
        metadata = ""
        if 'question_type_en' in row:
            metadata += f"Type: {row['question_type_en']}"
            
        if 'question_category_en' in row:
            metadata += f", Category: {row['question_category_en']}"
        
        query_title = row.get('query_title_en', '')
        query_content = row.get('query_content_en', '')
        
        clinical_context = ""
        if query_title or query_content:
            clinical_context += "Background Clinical Information (to help with your analysis):\n"
            if query_title:
                clinical_context += f"{query_title}\n"
            if query_content:
                clinical_context += f"{query_content}\n"
        
        options = row.get('options_en_cleaned', row.get('options_en', ['Yes', 'No', 'Not mentioned']))
        if isinstance(options, list):
            options_text = ", ".join(options)
        else:
            options_text = str(options)
        
        query_text = (f"MAIN QUESTION TO ANSWER: {question}\n"
                     f"Question Metadata: {metadata}\n"
                     f"{clinical_context}"
                     f"Available Options (choose from these): {options_text}")
        
        return query_text


class AgenticRAGData:
    """Manages combined data for agentic reasoning with RAG."""
    
    def __init__(self, all_models_df, validation_df):
        self.all_models_df = all_models_df
        self.validation_df = validation_df
        
        self.model_predictions = {}
        for (encounter_id, base_qid), group in all_models_df.groupby(['encounter_id', 'base_qid']):
            self.model_predictions[(encounter_id, base_qid)] = group
        
        self.validation_data = {}
        for _, row in validation_df.iterrows():
            self.validation_data[(row['encounter_id'], row['base_qid'])] = row
    
    def get_combined_data(self, encounter_id, base_qid):
        """Retrieve combined data for a specific encounter and question."""
        model_preds = self.model_predictions.get((encounter_id, base_qid), None)
        
        val_data = self.validation_data.get((encounter_id, base_qid), None)
        
        if model_preds is None:
            print(f"No model predictions found for encounter {encounter_id}, question {base_qid}")
            return None
            
        if val_data is None:
            print(f"No validation data found for encounter {encounter_id}, question {base_qid}")
            return None
        
        if 'query_context' not in val_data:
            val_data['query_context'] = DataProcessor.create_query_context(val_data)
        
        model_predictions_dict = {}
        for _, row in model_preds.iterrows():
            model_name = row['model_name']
            
            model_predictions_dict[model_name] = self._process_model_predictions(row)
        
        return {
            'encounter_id': encounter_id,
            'base_qid': base_qid,
            'query_context': val_data['query_context'],
            'images': val_data.get('all_images', []),
            'options': val_data.get('options_en_cleaned', val_data.get('options_en', [])),
            'question_type': val_data.get('question_type_en', ''),
            'question_category': val_data.get('question_category_en', ''),
            'model_predictions': model_predictions_dict
        }
    
    def _process_model_predictions(self, row):
        """Process model predictions from row data."""
        return {
            'model_prediction': row.get('combined_prediction', '')
        }
    
    def get_all_encounter_question_pairs(self):
        """Return a list of all unique encounter_id, base_qid pairs."""
        return list(self.validation_data.keys())
    
    def get_sample_data(self, n=5):
        """Get a sample of combined data for n random encounter-question pairs."""
        import random
        
        all_pairs = self.get_all_encounter_question_pairs()
        sample_pairs = random.sample(all_pairs, min(n, len(all_pairs)))
        
        return [self.get_combined_data(encounter_id, base_qid) for encounter_id, base_qid in sample_pairs]


def parse_json_response(text):
    """Parse JSON from LLM response."""
    cleaned_text = text
    if "```json" in cleaned_text:
        cleaned_text = cleaned_text.split("```json")[1]
    if "```" in cleaned_text:
        cleaned_text = cleaned_text.split("```")[0]
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse as JSON")
        return {"parse_error": "Could not parse as JSON", "raw_text": text}


class KnowledgeBaseManager:
    """Manages the dermatology knowledge base for RAG."""

    def __init__(self, args=None):
        """Initialize the knowledge base manager."""
        self.args = args
        self.embedding_model = SentenceTransformer(args.embedding_model if args else "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        self.cross_encoder = CrossEncoder(args.cross_encoder_model if args else "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.db_path = args.knowledge_db_path if args else os.path.join(os.getcwd(), "knowledge_db")
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)

        self.table_name = "dermatology_knowledge"

        if self.table_name not in self.db.table_names():
            print(f"Knowledge base not found. Creating new knowledge base at {self.db_path}")
            self._initialize_knowledge_base()
        else:
            print(f"Using existing knowledge base at {self.db_path}")
            self.table = self.db.open_table(self.table_name)

        self.tokenized_corpus = []
        self.doc_ids = []
        self._initialize_bm25_index()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with the skin diseases dataset."""
        print("Loading dermatology dataset...")
        dataset_name = self.args.dataset_name_huggingface if self.args else "brucewayne0459/Skin_diseases_and_care"
        dataset = load_dataset(dataset_name)

        data = []

        print("Processing dataset and creating embeddings...")
        for i, item in enumerate(dataset['train']):
            topic = item['Topic']
            information = item['Information']

            combined_text = f"Topic: {topic}\n\nInformation: {information}"

            embedding = self.embedding_model.encode(combined_text)

            data.append({
                "id": i,
                "topic": topic,
                "information": information,
                "combined_text": combined_text,
                "vector": embedding.tolist()
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} documents")

        import pandas as pd
        data_df = pd.DataFrame(data)

        print("Creating vector database...")
        self.table = self.db.create_table(
            self.table_name,
            data=data_df
        )
        print("Knowledge base initialization complete.")
    
    def _initialize_bm25_index(self):
        """Initialize the BM25 index for keyword search without NLTK dependencies."""
        print("Initializing BM25 index...")

        results = self.table.search().limit(10000).to_pandas()

        common_stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "by", "about", "from", "as", "of", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "can", "could", "will",
            "would", "shall", "should", "may", "might", "must", "this", "that", "these",
            "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her"
        }

        for idx, row in results.iterrows():
            doc_text = row['combined_text']
            self.doc_ids.append(row['id'])

            tokens = []
            for token in doc_text.lower().split():
                token = ''.join(c for c in token if c.isalnum())
                if token and token not in common_stopwords:
                    tokens.append(token)

            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index initialization complete.")
    
    def semantic_search(self, query, top_k=None):
        """Perform semantic search using embeddings."""
        if top_k is None:
            top_k = self.args.top_k_semantic if self.args else 7
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.table.search(query_embedding.tolist()).limit(top_k).to_pandas()
        
        return results
    
    def keyword_search(self, query, top_k=None):
        """Perform keyword search using BM25."""
        if top_k is None:
            top_k = self.args.top_k_keyword if self.args else 7

        common_stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "by", "about", "from", "as", "of", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "can", "could", "will",
            "would", "shall", "should", "may", "might", "must", "this", "that", "these",
            "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her"
        }

        query_tokens = []
        for token in query.lower().split():
            token = ''.join(c for c in token if c.isalnum())
            if token and token not in common_stopwords:
                query_tokens.append(token)

        doc_scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                doc_id = self.doc_ids[idx]
                score = doc_scores[idx]
                
                doc = self.table.search().where(f"id = {doc_id}").limit(1).to_pandas()
                
                if not doc.empty:
                    results.append({
                        "id": doc_id,
                        "topic": doc['topic'].iloc[0],
                        "information": doc['information'].iloc[0],
                        "combined_text": doc['combined_text'].iloc[0],
                        "_distance": 1.0 - min(score / 10.0, 1.0)
                    })
        
        return pd.DataFrame(results)
    
    def hybrid_search(self, query, top_k=None):
        """Perform hybrid search combining semantic and keyword search."""
        if top_k is None:
            top_k = self.args.top_k_hybrid if self.args else 10
        
        semantic_results = self.semantic_search(query, top_k=top_k)
        keyword_results = self.keyword_search(query, top_k=top_k)
        
        combined_results = pd.concat([semantic_results, keyword_results])
        combined_results = combined_results.drop_duplicates(subset=['id'])
        
        if len(combined_results) > 0:
            return self.rerank_results(combined_results, query, top_k=min(top_k, len(combined_results)))
        else:
            return pd.DataFrame()
    
    def rerank_results(self, results, query, top_k=None):
        """Rerank search results using a cross-encoder."""
        if top_k is None:
            top_k = self.args.top_k_rerank if self.args else 5
        
        if len(results) == 0:
            return pd.DataFrame()
        
        pairs = [(query, doc) for doc in results['combined_text'].tolist()]
        
        cross_scores = self.cross_encoder.predict(pairs)
        
        results = results.copy()
        results['cross_score'] = cross_scores
        
        results = results.sort_values(by='cross_score', ascending=False).head(top_k)
        
        return results


class DiagnosisExtractor:
    """Extracts potential diagnoses from image analysis and clinical context."""
    
    @staticmethod
    def extract_diagnoses(image_analysis, clinical_context, args=None):
        """
        Extract potential diagnoses from image analysis and clinical context.
        
        Args:
            image_analysis: Structured image analysis containing OVERALL_IMPRESSION
            clinical_context: Structured clinical context analysis
            
        Returns:
            List of dictionaries with diagnoses and confidence scores
        """
        diagnoses = []
        
        if image_analysis and "aggregated_analysis" in image_analysis:
            if "OVERALL_IMPRESSION" in image_analysis["aggregated_analysis"]:
                impression = image_analysis["aggregated_analysis"]["OVERALL_IMPRESSION"]
                if isinstance(impression, str):
                    diagnoses.extend(DiagnosisExtractor._extract_from_text(impression, source="image_analysis", confidence=0.7))
        
        if clinical_context and "structured_clinical_context" in clinical_context:
            if "DIAGNOSTIC_CONSIDERATIONS" in clinical_context["structured_clinical_context"]:
                diagnostic_info = clinical_context["structured_clinical_context"]["DIAGNOSTIC_CONSIDERATIONS"]
                if isinstance(diagnostic_info, str):
                    diagnoses.extend(DiagnosisExtractor._extract_from_text(diagnostic_info, source="clinical_context", confidence=0.6))
        
        if not diagnoses:
            diagnoses = DiagnosisExtractor._suggest_from_features(image_analysis, clinical_context)
            
        return diagnoses
    
    @staticmethod
    def _extract_from_text(text, source, confidence):
        """Extract diagnoses from text."""
        import re
        
        diagnostic_terms = [
            "eczema", "dermatitis", "psoriasis", "acne", "rosacea", "urticaria", 
            "melanoma", "carcinoma", "pemphigus", "pemphigoid", "lupus", "scleroderma",
            "folliculitis", "cellulitis", "impetigo", "tinea", "herpes", "wart",
            "vitiligo", "alopecia", "lichen", "keratosis", "prurigo", "rash"
        ]
        
        diagnoses = []
        
        for term in diagnostic_terms:
            pattern = fr'\b({term})[s\s]\b'
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                diagnoses.append({
                    "diagnosis": match.group(1).capitalize(),
                    "confidence": confidence,
                    "source": source
                })
                
        patterns = [
            r'consistent with\s+([^,.;]+)',
            r'suggestive of\s+([^,.;]+)',
            r'indicative of\s+([^,.;]+)',
            r'compatible with\s+([^,.;]+)',
            r'diagnostic of\s+([^,.;]+)',
            r'likely\s+([^,.;]+)',
            r'probable\s+([^,.;]+)',
            r'possible\s+([^,.;]+)',
            r'suspected\s+([^,.;]+)',
            r'diagnosis of\s+([^,.;]+)',
            r'impression:\s+([^,.;]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                diagnoses.append({
                    "diagnosis": match.group(1).strip().capitalize(),
                    "confidence": confidence * 0.9,
                    "source": source
                })
        
        unique_diagnoses = []
        seen = set()
        for diag in diagnoses:
            if diag["diagnosis"].lower() not in seen:
                seen.add(diag["diagnosis"].lower())
                unique_diagnoses.append(diag)
        
        return unique_diagnoses
    
    @staticmethod
    def _suggest_from_features(image_analysis, clinical_context):
        """Suggest potential diagnoses based on extracted features."""
        diagnoses = []
        features = {}
        
        if image_analysis and "aggregated_analysis" in image_analysis:
            analysis = image_analysis["aggregated_analysis"]
            
            if "SKIN_DESCRIPTION" in analysis:
                features["skin_description"] = analysis["SKIN_DESCRIPTION"]
                
            if "LESION_COLOR" in analysis:
                features["lesion_color"] = analysis["LESION_COLOR"]
                
            if "SITE_LOCATION" in analysis:
                features["site_location"] = analysis["SITE_LOCATION"]
        
        if clinical_context and "structured_clinical_context" in clinical_context:
            context = clinical_context["structured_clinical_context"]
            
            if "SYMPTOMS" in context:
                features["symptoms"] = context["SYMPTOMS"]
                
            if "HISTORY" in context:
                features["history"] = context["HISTORY"]
        
        if features:
            if "hand" in str(features.get("site_location", "")).lower():
                if "scaling" in str(features.get("skin_description", "")).lower():
                    diagnoses.append({
                        "diagnosis": "Hand eczema",
                        "confidence": 0.5,
                        "source": "feature_based"
                    })
                    
            if "red" in str(features.get("lesion_color", "")).lower():
                if "itchy" in str(features.get("symptoms", "")).lower():
                    diagnoses.append({
                        "diagnosis": "Contact dermatitis",
                        "confidence": 0.4,
                        "source": "feature_based"
                    })
        
        if not diagnoses:
            diagnoses.append({
                "diagnosis": "Dermatosis", 
                "confidence": 0.3,
                "source": "fallback"
            })
            
        return diagnoses


class DiagnosisBasedQueryGenerator:
    """Generates search queries based on extracted diagnoses."""
        
    def __init__(self, client, args=None):
        """Initialize the query generator."""
        self.client = client
        self.args = args
    
    def generate_queries(self, question_text, question_type, options, integrated_evidence, diagnoses, num_queries=4):
        """
        Generate search queries based on diagnoses and question type.
        
        Args:
            question_text: The question text
            question_type: Type of question being asked
            options: Available answer options
            integrated_evidence: Integrated evidence from images and clinical context
            diagnoses: List of extracted diagnoses
            num_queries: Number of queries to generate
            
        Returns:
            List of search queries
        """
        sorted_diagnoses = sorted(diagnoses, key=lambda x: x.get('confidence', 0), reverse=True)
        
        question_specific_queries = self._generate_question_specific_queries(
            question_text, 
            question_type, 
            options, 
            sorted_diagnoses
        )
        
        diagnosis_specific_queries = self._generate_diagnosis_specific_queries(
            question_type,
            sorted_diagnoses
        )
        
        all_queries = question_specific_queries + diagnosis_specific_queries
        
        unique_queries = []
        seen = set()
        for query in all_queries:
            if query.lower() not in seen:
                seen.add(query.lower())
                unique_queries.append(query)
        
        return unique_queries[:num_queries]
    
    def _generate_question_specific_queries(self, question_text, question_type, options, diagnoses):
        """Generate queries specific to the question type."""
        queries = []
        
        classification_types = ["Site Location", "Lesion Color", "Size", "Extent", "Lesion Count"]
        if question_type in classification_types:
            classification_terms = ", ".join([opt for opt in options if opt.lower() != "not mentioned"])
            queries.append(f"dermatology {question_type.lower()} classification {classification_terms}")
            
            if len(options) > 2:
                queries.append(f"how to distinguish between {classification_terms} in dermatology")
                
            if question_type == "Extent":
                queries.append("definition of widespread vs limited area skin condition dermatology")
        
        if question_type in ["Differential", "Specific Diagnosis"]:
            if diagnoses:
                top_diagnosis = diagnoses[0]["diagnosis"]
                queries.append(f"{top_diagnosis} diagnostic criteria dermatology")
                
                diagnoses_list = ", ".join([d["diagnosis"] for d in diagnoses[:3]])
                queries.append(f"differential diagnosis {diagnoses_list}")
        
        if question_type == "Treatment":
            if diagnoses:
                top_diagnosis = diagnoses[0]["diagnosis"]
                queries.append(f"{top_diagnosis} treatment options dermatology")
                
                body_site = self._extract_body_site(question_text)
                if body_site:
                    queries.append(f"{top_diagnosis} {body_site} treatment guidelines")
        
        return queries
    
    def _generate_diagnosis_specific_queries(self, question_type, diagnoses):
        """Generate queries that connect diagnoses with the question type."""
        queries = []
        
        if not diagnoses:
            return queries
            
        for diagnosis in diagnoses[:2]:
            diag_name = diagnosis["diagnosis"]
            
            if question_type in ["Site Location", "Extent"]:
                queries.append(f"{diag_name} typical distribution pattern dermatology")
                queries.append(f"{diag_name} localized versus widespread presentation")
                
            elif question_type == "Lesion Color":
                queries.append(f"{diag_name} typical color appearance dermatology")
                
            elif question_type == "Texture":
                queries.append(f"{diag_name} texture characteristics dermatology")
                
            elif question_type == "Itch":
                queries.append(f"is {diag_name} itchy dermatology")
                
            elif question_type == "Onset":
                queries.append(f"{diag_name} typical onset and progression")
                
            else:
                queries.append(f"{diag_name} {question_type.lower()} dermatology")
                
        return queries
    
    def _extract_body_site(self, question_text):
        """Extract body site from question text."""
        import re
        
        body_parts = [
            "hand", "foot", "arm", "leg", "face", "back", "chest", "abdomen",
            "scalp", "neck", "finger", "toe", "elbow", "knee", "shoulder",
            "palm", "sole", "trunk", "extremity", "head"
        ]
        
        for part in body_parts:
            if re.search(r'\b' + part + r'[s]?\b', question_text.lower()):
                return part
                
        return None


class DiagnosisBasedKnowledgeRetriever:
    """Retrieves knowledge from the dermatology knowledge base using diagnosis-based approach."""
    
    def __init__(self, kb_manager, query_generator, diagnosis_extractor, args=None):
        """
        Initialize the knowledge retriever.
        
        Args:
            kb_manager: KnowledgeBaseManager instance
            query_generator: DiagnosisBasedQueryGenerator instance
            diagnosis_extractor: DiagnosisExtractor instance
            args: Configuration arguments
        """
        self.kb_manager = kb_manager
        self.query_generator = query_generator
        self.diagnosis_extractor = diagnosis_extractor
        self.args = args
    
    def retrieve_knowledge(self, question_text, question_type, options, image_analysis, clinical_context, integrated_evidence):
        """
        Retrieve relevant knowledge for a dermatological question using diagnoses.
        
        Args:
            question_text: The question text
            question_type: Type of question being asked
            options: Available answer options
            image_analysis: Structured image analysis
            clinical_context: Structured clinical context
            integrated_evidence: Integrated evidence from images and clinical context
            
        Returns:
            Dictionary with retrieved knowledge
        """
        if self.args:
            rag_config = self.args.question_type_retrieval_config.get(
                question_type, self.args.default_rag_config
            )
        else:
            default_config = {"use_rag": True, "weight": 0.4}
            rag_config = {
                "Site Location": {"use_rag": False, "weight": 0.2},
                "Lesion Color": {"use_rag": False, "weight": 0.2},
                "Size": {"use_rag": False, "weight": 0.1},
            }.get(question_type, default_config)
        
        if not rag_config["use_rag"]:
            return {
                "retrieved": False,
                "reason": f"RAG not enabled for question type: {question_type}",
                "results": []
            }
        
        diagnoses = self.diagnosis_extractor.extract_diagnoses(image_analysis, clinical_context)
        
        queries = self.query_generator.generate_queries(
            question_text, 
            question_type, 
            options, 
            integrated_evidence,
            diagnoses
        )
        
        if not queries:
            return {
                "retrieved": False,
                "reason": "Failed to generate search queries",
                "results": []
            }
        
        all_results = []
        
        for query in queries:
            results = self.kb_manager.hybrid_search(query)
            
            if not results.empty:
                for _, row in results.iterrows():
                    relevance_score = float(row.get('cross_score', 1.0 - row.get('_distance', 0.5)))
                    
                    if relevance_score > 0:
                        all_results.append({
                            "query": query,
                            "topic": row['topic'],
                            "information": row['information'],
                            "relevance_score": relevance_score,
                            "diagnoses": [d["diagnosis"] for d in diagnoses[:3]]
                        })
        
        unique_results = []
        seen_topics = set()
        
        for result in sorted(all_results, key=lambda x: x['relevance_score'], reverse=True):
            if result['topic'] not in seen_topics:
                unique_results.append(result)
                seen_topics.add(result['topic'])
        
        top_k = self.args.top_k_rerank if self.args else 5
        
        return {
            "retrieved": len(unique_results) > 0,
            "queries": queries,
            "diagnoses": diagnoses,
            "results": unique_results[:top_k]
        }


class AgenticDermatologyPipeline:
    """Main pipeline for agentic dermatology analysis with diagnosis-based retrieval."""
    
    def __init__(self, api_key=None, args=None):
        if api_key is None:
            api_key = os.getenv("API_KEY")
        
        self.client = genai.Client(api_key=api_key)
        self.args = args
        
        print("Initializing knowledge base...")
        self.kb_manager = KnowledgeBaseManager(args)
        
        self.diagnosis_extractor = DiagnosisExtractor()
        
        self.query_generator = DiagnosisBasedQueryGenerator(self.client, args)
        
        self.knowledge_retriever = DiagnosisBasedKnowledgeRetriever(
            self.kb_manager,
            self.query_generator,
            self.diagnosis_extractor,
            args
        )
    
    def analyze_images(self, images: List[str], query_context: str) -> Dict[str, Any]:
        """Analyze images and extract structured information."""
        if not images:
            return {"error": "No images provided"}
        
        prompt = f"""
        Analyze these dermatology images and provide structured output:
        
        Context: {query_context}
        
        Please provide a detailed analysis in the following JSON format:
        {{
            "individual_images": [
                {{
                    "image_index": 0,
                    "observations": {{
                        "SITE_LOCATION": "affected body parts",
                        "LESION_COLOR": "colors observed",
                        "SKIN_DESCRIPTION": "texture, appearance details",
                        "LESION_COUNT": "number of lesions",
                        "SIZE": "approximate sizes"
                    }}
                }}
            ],
            "aggregated_analysis": {{
                "SITE_LOCATION": "all affected areas combined",
                "LESION_COLOR": "all colors observed",
                "SKIN_DESCRIPTION": "overall skin appearance",
                "LESION_COUNT": "total count",
                "SIZE": "size range",
                "OVERALL_IMPRESSION": "clinical impression and possible diagnoses"
            }}
        }}
        """
        
        try:
            # Load images
            image_parts = []
            for img_path in images[:5]:  # Limit to 5 images
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    image_parts.append(img)
            
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=[prompt] + image_parts
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error analyzing images: {e}")
            return {"error": str(e)}
    
    def extract_clinical_context(self, query_context: str) -> Dict[str, Any]:
        """Extract structured clinical context from query."""
        prompt = f"""
        Extract structured clinical information from this query:
        
        {query_context}
        
        Provide output in JSON format:
        {{
            "structured_clinical_context": {{
                "HISTORY": "relevant history mentioned",
                "SYMPTOMS": "symptoms described",
                "DURATION": "time course mentioned",
                "MEDICATIONS": "any medications mentioned",
                "DIAGNOSTIC_CONSIDERATIONS": "possible diagnoses based on context"
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=prompt
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error extracting clinical context: {e}")
            return {"error": str(e)}
    
    def integrate_evidence(self, image_analysis: Dict, clinical_context: Dict, 
                          model_predictions: Dict, retrieved_knowledge: Dict) -> Dict[str, Any]:
        """Integrate all evidence sources."""
        prompt = f"""
        Integrate the following evidence sources for dermatological analysis:
        
        1. Image Analysis:
        {json.dumps(image_analysis, indent=2)}
        
        2. Clinical Context:
        {json.dumps(clinical_context, indent=2)}
        
        3. Model Predictions:
        {json.dumps(model_predictions, indent=2)}
        
        4. Retrieved Knowledge:
        {json.dumps(retrieved_knowledge, indent=2)}
        
        Provide integrated analysis in JSON format:
        {{
            "integrated_findings": {{
                "primary_features": "key features from all sources",
                "diagnostic_confidence": "confidence level based on evidence",
                "supporting_evidence": "evidence supporting diagnosis",
                "conflicting_evidence": "any conflicts between sources",
                "recommended_answer": "most likely answer based on all evidence"
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=prompt
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error integrating evidence: {e}")
            return {"error": str(e)}
    
    def reason_with_reflection(self, question_text: str, options: List[str], 
                              integrated_evidence: Dict, cycle: int = 0) -> Dict[str, Any]:
        """Perform reasoning with self-reflection."""
        prompt = f"""
        Question: {question_text}
        Options: {', '.join(options)}
        
        Integrated Evidence:
        {json.dumps(integrated_evidence, indent=2)}
        
        Reasoning Cycle: {cycle + 1}
        
        Analyze this dermatology question step by step:
        
        1. What is the question specifically asking?
        2. What evidence supports each option?
        3. Which option is most strongly supported?
        4. What is your confidence level (0-1)?
        
        Provide your analysis in JSON format:
        {{
            "reasoning": {{
                "question_analysis": "what the question asks",
                "option_evidence": {{
                    "option1": "evidence for/against",
                    "option2": "evidence for/against"
                }},
                "conclusion": "selected answer",
                "confidence": 0.85,
                "areas_of_uncertainty": ["list of uncertainties"]
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=prompt
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error in reasoning: {e}")
            return {"error": str(e)}
    
    def self_reflect(self, reasoning: Dict, integrated_evidence: Dict) -> Dict[str, Any]:
        """Perform self-reflection on reasoning."""
        prompt = f"""
        Review this reasoning for potential errors or improvements:
        
        Reasoning:
        {json.dumps(reasoning, indent=2)}
        
        Evidence:
        {json.dumps(integrated_evidence, indent=2)}
        
        Identify:
        1. Any logical errors or inconsistencies
        2. Missing considerations
        3. Whether confidence level is appropriate
        4. Suggestions for improvement
        
        Provide reflection in JSON format:
        {{
            "reflection": {{
                "logical_errors": ["list of errors"],
                "missing_considerations": ["what was missed"],
                "confidence_assessment": "is confidence appropriate?",
                "improvement_suggestions": ["suggestions"],
                "should_revise": true/false
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=prompt
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return {"error": str(e)}
    
    def process_question(self, sample_data: Dict) -> Dict[str, Any]:
        """Process a single question through the complete pipeline."""
        question_text = sample_data['query_context'].split("MAIN QUESTION TO ANSWER:")[1].split("\n")[0].strip()
        question_type = sample_data['question_type']
        options = sample_data['options']
        model_predictions = sample_data['model_predictions']
        images = sample_data.get('images', [])
        
        # Step 1: Analyze images
        image_analysis = self.analyze_images(images, sample_data['query_context'])
        
        # Step 2: Extract clinical context
        clinical_context = self.extract_clinical_context(sample_data['query_context'])
        
        # Step 3: Initial integration
        initial_integration = self.integrate_evidence(
            image_analysis, clinical_context, model_predictions, {}
        )
        
        # Step 4: Retrieve knowledge based on diagnoses
        retrieved_knowledge = self.knowledge_retriever.retrieve_knowledge(
            question_text, question_type, options,
            image_analysis, clinical_context, initial_integration
        )
        
        # Step 5: Final integration with retrieved knowledge
        integrated_evidence = self.integrate_evidence(
            image_analysis, clinical_context, model_predictions, retrieved_knowledge
        )
        
        # Step 6: Reasoning with reflection cycles
        best_reasoning = None
        best_confidence = 0
        
        for cycle in range(self.args.max_reflection_cycles if self.args else 2):
            reasoning = self.reason_with_reflection(
                question_text, options, integrated_evidence, cycle
            )
            
            if "reasoning" in reasoning and "confidence" in reasoning["reasoning"]:
                current_confidence = reasoning["reasoning"]["confidence"]
                
                if current_confidence > best_confidence:
                    best_reasoning = reasoning
                    best_confidence = current_confidence
                
                # Reflect on reasoning
                reflection = self.self_reflect(reasoning, integrated_evidence)
                
                # Check if we should stop
                if (reflection.get("reflection", {}).get("should_revise", True) == False or 
                    current_confidence >= (self.args.confidence_threshold if self.args else 0.75)):
                    break
        
        # Extract final answer
        final_answer = "Not mentioned"
        if best_reasoning and "reasoning" in best_reasoning:
            final_answer = best_reasoning["reasoning"].get("conclusion", "Not mentioned")
        
        return {
            "question_text": question_text,
            "question_type": question_type,
            "options": options,
            "final_answer": final_answer,
            "confidence": best_confidence,
            "reasoning": best_reasoning,
            "retrieved_knowledge": retrieved_knowledge,
            "image_analysis": image_analysis,
            "clinical_context": clinical_context,
            "integrated_evidence": integrated_evidence
        }
    
    def process_single_encounter(self, agentic_data, encounter_id):
        """
        Process a single encounter with all its questions using the agentic pipeline.

        Args:
            agentic_data: AgenticRAGData instance containing all encounter data
            encounter_id: The specific encounter ID to process

        Returns:
            Dictionary with all questions processed with agentic reasoning for this encounter
        """
        all_pairs = agentic_data.get_all_encounter_question_pairs()
        encounter_pairs = [pair for pair in all_pairs if pair[0] == encounter_id]

        if not encounter_pairs:
            print(f"No data found for encounter {encounter_id}")
            return None

        print(f"Processing {len(encounter_pairs)} questions for encounter {encounter_id}")

        encounter_results = {encounter_id: {}}

        # Process all questions for this encounter
        for i, (encounter_id, base_qid) in enumerate(encounter_pairs):
            print(f"Processing question {i+1}/{len(encounter_pairs)}: {base_qid}")

            sample_data = agentic_data.get_combined_data(encounter_id, base_qid)
            if not sample_data:
                print(f"Warning: No data found for {encounter_id}, {base_qid}")
                continue

            try:
                # Process through the complete pipeline
                result = self.process_question(sample_data)
                
                encounter_results[encounter_id][base_qid] = {
                    "query_context": sample_data['query_context'],
                    "options": sample_data['options'],
                    "model_predictions": sample_data['model_predictions'],
                    "final_answer": result["final_answer"],
                    "confidence": result["confidence"],
                    "reasoning": result.get("reasoning", {}),
                    "retrieved_knowledge": result.get("retrieved_knowledge", {})
                }
                
            except Exception as e:
                print(f"Error processing {encounter_id}, {base_qid}: {e}")
                encounter_results[encounter_id][base_qid] = {
                    "query_context": sample_data['query_context'],
                    "options": sample_data['options'],
                    "model_predictions": sample_data['model_predictions'],
                    "final_answer": "Not mentioned",
                    "error": str(e)
                }

        # Save intermediate results
        output_file = os.path.join(
            self.args.output_dir if self.args else os.getcwd(), 
            f"diagnosis_based_rag_results_{encounter_id}.json"
        )
        
        with open(output_file, "w") as f:
            json.dump(encounter_results, f, indent=2)

        print(f"Processed all {len(encounter_pairs)} questions for encounter {encounter_id}")
        return encounter_results
    
    def format_results_for_evaluation(self, encounter_results, output_file):
        """Format results for official evaluation."""
        QIDS = [
            "CQID010-001",
            "CQID011-001", "CQID011-002", "CQID011-003", "CQID011-004", "CQID011-005", "CQID011-006",
            "CQID012-001", "CQID012-002", "CQID012-003", "CQID012-004", "CQID012-005", "CQID012-006",
            "CQID015-001",
            "CQID020-001", "CQID020-002", "CQID020-003", "CQID020-004", "CQID020-005", 
            "CQID020-006", "CQID020-007", "CQID020-008", "CQID020-009",
            "CQID025-001",
            "CQID034-001",
            "CQID035-001",
            "CQID036-001",
        ]
        
        qid_variants = {}
        for qid in QIDS:
            base_qid, variant = qid.split('-')
            if base_qid not in qid_variants:
                qid_variants[base_qid] = []
            qid_variants[base_qid].append(qid)
        
        required_base_qids = set(qid.split('-')[0] for qid in QIDS)
        
        formatted_predictions = []
        for encounter_id, questions in encounter_results.items():
            encounter_base_qids = set(questions.keys())
            if not required_base_qids.issubset(encounter_base_qids):
                print(f"Skipping encounter {encounter_id} - missing required questions")
                continue
            
            pred_entry = {'encounter_id': encounter_id}
            
            for base_qid, question_data in questions.items():
                if base_qid not in qid_variants:
                    continue
                
                final_answer = question_data['final_answer']
                options = question_data['options']
                
                not_mentioned_index = self._find_not_mentioned_index(options)
                
                self._process_answers(
                    pred_entry, 
                    base_qid, 
                    final_answer, 
                    options, 
                    qid_variants, 
                    not_mentioned_index
                )
            
            formatted_predictions.append(pred_entry)
        
        with open(output_file, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)
        
        print(f"Formatted predictions saved to {output_file} ({len(formatted_predictions)} complete encounters)")
        return formatted_predictions
    
    def _find_not_mentioned_index(self, options):
        """Find the index of 'Not mentioned' in options."""
        for i, opt in enumerate(options):
            if opt.lower() == "not mentioned":
                return i
        return len(options) - 1
    
    def _process_answers(self, pred_entry, base_qid, final_answer, options, qid_variants, not_mentioned_index):
        """Process answers and add to prediction entry."""
        if ',' in final_answer:
            answer_parts = [part.strip() for part in final_answer.split(',')]
            answer_indices = []
            
            for part in answer_parts:
                found = False
                for i, opt in enumerate(options):
                    if part.lower() == opt.lower():
                        answer_indices.append(i)
                        found = True
                        break
                
                if not found:
                    answer_indices.append(not_mentioned_index)
            
            available_variants = qid_variants[base_qid]
            
            for i, idx in enumerate(answer_indices):
                if i < len(available_variants):
                    pred_entry[available_variants[i]] = idx
            
            for i in range(len(answer_indices), len(available_variants)):
                pred_entry[available_variants[i]] = not_mentioned_index
            
        else:
            answer_index = not_mentioned_index
            
            for i, opt in enumerate(options):
                if final_answer.lower() == opt.lower():
                    answer_index = i
                    break
            
            pred_entry[qid_variants[base_qid][0]] = answer_index
            
            if len(qid_variants[base_qid]) > 1:
                for i in range(1, len(qid_variants[base_qid])):
                    pred_entry[qid_variants[base_qid][i]] = not_mentioned_index


def run_diagnosis_based_pipeline_all_encounters(args=None):
    """Run the diagnosis-based pipeline for all available encounters."""
    if args is None:
        args = Args(use_finetuning=True, use_test_dataset=True)
        
    # Load data
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    if not model_predictions_dict:
        print("No model predictions found. Exiting.")
        return
        
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    validation_df = DataLoader.load_validation_dataset(args)
    
    # Create agentic data manager
    agentic_data = AgenticRAGData(all_models_df, validation_df)
    
    # Initialize pipeline
    pipeline = AgenticDermatologyPipeline(args=args)
    
    # Get all unique encounters
    all_pairs = agentic_data.get_all_encounter_question_pairs()
    unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
    print(f"Found {len(unique_encounter_ids)} unique encounters to process")
    
    # Process all encounters
    all_encounter_results = {}
    
    for i, encounter_id in enumerate(unique_encounter_ids):
        print(f"\nProcessing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
        
        try:
            encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
            if encounter_results:
                all_encounter_results.update(encounter_results)
                
            # Save intermediate results periodically
            if (i+1) % 5 == 0 or (i+1) == len(unique_encounter_ids):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                intermediate_file = os.path.join(
                    args.output_dir,
                    f"diagnosis_rag_intermediate_results_{timestamp}.json"
                )
                with open(intermediate_file, "w") as f:
                    json.dump(all_encounter_results, f, indent=2)
                print(f"Saved intermediate results to {intermediate_file}")
                
        except Exception as e:
            print(f"Error processing encounter {encounter_id}: {e}")
            traceback.print_exc()
            continue
    
    # Format and save final results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    formatted_output_file = os.path.join(
        args.output_dir,
        f"diagnosis_rag_formatted_predictions_{timestamp}.json"
    )
    
    formatted_predictions = pipeline.format_results_for_evaluation(
        all_encounter_results, 
        formatted_output_file
    )
    
    # Save complete results
    complete_output_file = os.path.join(
        args.output_dir,
        f"diagnosis_rag_complete_results_{timestamp}.json"
    )
    with open(complete_output_file, "w") as f:
        json.dump(all_encounter_results, f, indent=2)
    
    print(f"\nPipeline completed!")
    print(f"Complete results saved to: {complete_output_file}")
    print(f"Formatted predictions saved to: {formatted_output_file}")
    print(f"Total encounters processed: {len(all_encounter_results)}")
    print(f"Total complete encounters for evaluation: {len(formatted_predictions)}")
    
    return all_encounter_results, formatted_predictions


def run_diagnosis_based_pipeline_sample(args=None, num_samples=3):
    """Run the pipeline on a sample of encounters for testing."""
    if args is None:
        args = Args(use_finetuning=True, use_test_dataset=True)
    
    # Load data
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    if not model_predictions_dict:
        print("No model predictions found. Exiting.")
        return
        
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    validation_df = DataLoader.load_validation_dataset(args)
    
    # Create agentic data manager
    agentic_data = AgenticRAGData(all_models_df, validation_df)
    
    # Initialize pipeline
    pipeline = AgenticDermatologyPipeline(args=args)
    
    # Get sample encounters
    all_pairs = agentic_data.get_all_encounter_question_pairs()
    unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
    
    # Sample random encounters
    sample_encounter_ids = random.sample(unique_encounter_ids, min(num_samples, len(unique_encounter_ids)))
    print(f"Testing on {len(sample_encounter_ids)} sample encounters: {sample_encounter_ids}")
    
    # Process sample encounters
    sample_results = {}
    
    for encounter_id in sample_encounter_ids:
        print(f"\nProcessing sample encounter: {encounter_id}")
        
        try:
            encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
            if encounter_results:
                sample_results.update(encounter_results)
                
        except Exception as e:
            print(f"Error processing encounter {encounter_id}: {e}")
            traceback.print_exc()
    
    # Save sample results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sample_output_file = os.path.join(
        args.output_dir,
        f"diagnosis_rag_sample_results_{timestamp}.json"
    )
    
    with open(sample_output_file, "w") as f:
        json.dump(sample_results, f, indent=2)
    
    print(f"\nSample results saved to: {sample_output_file}")
    print(f"Processed {len(sample_results)} encounters")
    
    return sample_results


# Parameterizable Wrapper Classes
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for the RAG Pipeline."""
    
    # Model and dataset configuration
    use_finetuning: bool = True
    use_test_dataset: bool = True
    gemini_model: str = "gemini-2.0-flash-exp-2025-01-29"
    
    # Directory paths
    base_dir: Optional[str] = None
    output_dir: Optional[str] = None
    model_predictions_dir: Optional[str] = None
    images_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    
    # API configuration
    api_key: Optional[str] = None
    
    # Processing options
    max_reflection_cycles: int = 2
    confidence_threshold: float = 0.75
    save_intermediate_results: bool = True
    intermediate_save_frequency: int = 5
    
    # Knowledge base configuration
    knowledge_db_path: Optional[str] = None
    embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_dimension: int = 768
    top_k_semantic: int = 7
    top_k_keyword: int = 7
    top_k_hybrid: int = 10
    top_k_rerank: int = 5
    
    # Dataset configuration
    dataset_name_huggingface: str = "brucewayne0459/Skin_diseases_and_care"
    
    # Question type retrieval configuration
    question_type_retrieval_config: Optional[Dict[str, Dict[str, Any]]] = None
    default_rag_config: Optional[Dict[str, Any]] = None
    
    def to_rag_args(self) -> Args:
        """Convert to Args format."""
        # Set default configurations if not provided
        if self.question_type_retrieval_config is None:
            self.question_type_retrieval_config = {
                "Site Location": {"use_rag": False, "weight": 0.2},
                "Lesion Color": {"use_rag": False, "weight": 0.2},
                "Size": {"use_rag": False, "weight": 0.1},
                "Skin Description": {"use_rag": True, "weight": 0.3},
                "Onset": {"use_rag": True, "weight": 0.4},
                "Itch": {"use_rag": True, "weight": 0.4},
                "Extent": {"use_rag": False, "weight": 0.2},
                "Treatment": {"use_rag": True, "weight": 0.7},
                "Lesion Evolution": {"use_rag": True, "weight": 0.5},
                "Texture": {"use_rag": True, "weight": 0.3},
                "Lesion Count": {"use_rag": False, "weight": 0.1},
                "Differential": {"use_rag": True, "weight": 0.8},
                "Specific Diagnosis": {"use_rag": True, "weight": 0.8},
            }
        
        if self.default_rag_config is None:
            self.default_rag_config = {"use_rag": True, "weight": 0.4}
        
        # Pass all configuration parameters directly to Args constructor
        args = Args(
            use_finetuning=self.use_finetuning,
            use_test_dataset=self.use_test_dataset,
            base_dir=self.base_dir,
            output_dir=self.output_dir,
            model_predictions_dir=self.model_predictions_dir,
            images_dir=self.images_dir,
            dataset_path=self.dataset_path,
            gemini_model=self.gemini_model,
            max_reflection_cycles=self.max_reflection_cycles,
            confidence_threshold=self.confidence_threshold,
            knowledge_db_path=self.knowledge_db_path,
            embedding_model=self.embedding_model,
            cross_encoder_model=self.cross_encoder_model,
            vector_dimension=self.vector_dimension,
            top_k_semantic=self.top_k_semantic,
            top_k_keyword=self.top_k_keyword,
            top_k_hybrid=self.top_k_hybrid,
            top_k_rerank=self.top_k_rerank,
            dataset_name_huggingface=self.dataset_name_huggingface,
            question_type_retrieval_config=self.question_type_retrieval_config,
            default_rag_config=self.default_rag_config
        )
        
        return args


class RAGPipeline:
    """
    Main wrapper class for the diagnosis-based RAG medical analysis pipeline.
    
    This class provides a clean, parameterizable interface for running medical image
    analysis with knowledge retrieval, self-reflection, and multi-cycle reasoning.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or RAGConfig()
        self.args = self.config.to_rag_args()
        self.pipeline = None
        self.agentic_data = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the pipeline components."""
        if self._initialized:
            return
            
        # Load environment variables for API key
        load_dotenv()
        
        # Initialize the main pipeline
        self.pipeline = AgenticDermatologyPipeline(
            api_key=self.config.api_key,
            args=self.args
        )
        
        # Load data
        model_predictions_dict = DataLoader.load_all_model_predictions(self.args)
        
        if not model_predictions_dict:
            raise ValueError("No model predictions found. Please check your configuration.")
            
        all_models_df = self._concat_model_predictions(model_predictions_dict)
        dataset_df = DataLoader.load_validation_dataset(self.args)
        
        self.agentic_data = AgenticRAGData(all_models_df, dataset_df)
        self._initialized = True
        
    def _concat_model_predictions(self, model_predictions_dict: Dict) -> Any:
        """Safely concatenate model predictions."""
        if not model_predictions_dict:
            raise ValueError("No model predictions to concatenate")
            
        return pd.concat(model_predictions_dict.values(), ignore_index=True)
        
    def process_single_encounter(self, encounter_id: str) -> Dict[str, Any]:
        """
        Process a single encounter with RAG analysis.
        
        Args:
            encounter_id: The encounter ID to process
            
        Returns:
            Dictionary containing the processed results
        """
        if not self._initialized:
            self.initialize()
            
        encounter_results = self.pipeline.process_single_encounter(self.agentic_data, encounter_id)
        
        if not encounter_results:
            raise ValueError(f"No results generated for encounter {encounter_id}")
            
        return encounter_results
        
    def process_all_encounters(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process all available encounters with RAG analysis.
        
        Returns:
            Tuple of (complete_results, formatted_predictions)
        """
        if not self._initialized:
            self.initialize()
            
        all_pairs = self.agentic_data.get_all_encounter_question_pairs()
        unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
        
        print(f"Found {len(unique_encounter_ids)} unique encounters to process")
        
        all_encounter_results = {}
        
        for i, encounter_id in enumerate(unique_encounter_ids):
            print(f"Processing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
            
            try:
                encounter_results = self.pipeline.process_single_encounter(self.agentic_data, encounter_id)
                if encounter_results:
                    all_encounter_results.update(encounter_results)
                
                # Save intermediate results if configured
                if (self.config.save_intermediate_results and 
                    ((i+1) % self.config.intermediate_save_frequency == 0 or (i+1) == len(unique_encounter_ids))):
                    self._save_intermediate_results(all_encounter_results, i+1, len(unique_encounter_ids))
                    
            except Exception as e:
                print(f"Error processing encounter {encounter_id}: {e}")
                continue
        
        # Format and save final results
        return self._format_and_save_final_results(all_encounter_results, unique_encounter_ids)
    
    def process_sample_encounters(self, num_samples: int = 3) -> Dict[str, Any]:
        """
        Process a sample of encounters for testing.
        
        Args:
            num_samples: Number of encounters to sample
            
        Returns:
            Dictionary containing the processed sample results
        """
        if not self._initialized:
            self.initialize()
            
        all_pairs = self.agentic_data.get_all_encounter_question_pairs()
        unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
        
        # Sample random encounters
        sample_encounter_ids = random.sample(unique_encounter_ids, min(num_samples, len(unique_encounter_ids)))
        print(f"Processing {len(sample_encounter_ids)} sample encounters: {sample_encounter_ids}")
        
        sample_results = {}
        
        for encounter_id in sample_encounter_ids:
            print(f"Processing sample encounter: {encounter_id}")
            
            try:
                encounter_results = self.pipeline.process_single_encounter(self.agentic_data, encounter_id)
                if encounter_results:
                    sample_results.update(encounter_results)
                    
            except Exception as e:
                print(f"Error processing encounter {encounter_id}: {e}")
                continue
        
        # Save sample results
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        sample_output_file = os.path.join(
            self.args.output_dir,
            f"rag_sample_results_{timestamp}.json"
        )
        
        with open(sample_output_file, "w") as f:
            json.dump(sample_results, f, indent=2)
        
        print(f"Sample results saved to: {sample_output_file}")
        print(f"Processed {len(sample_results)} encounters")
        
        return sample_results
    
    def _save_intermediate_results(self, results: Dict, current: int, total: int) -> None:
        """Save intermediate results during processing."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        intermediate_output_file = os.path.join(
            self.args.output_dir, 
            f"rag_intermediate_results_{current}_of_{total}_{timestamp}.json"
        )
        with open(intermediate_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved intermediate results after processing {current} encounters")
    
    def _format_and_save_final_results(self, all_encounter_results: Dict, unique_encounter_ids: List) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Format and save final results."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete results
        complete_output_file = os.path.join(
            self.args.output_dir, 
            f"{self.args.dataset_name}_data_cvqa_rag_complete_{timestamp}.json"
        )
        with open(complete_output_file, 'w') as f:
            json.dump(all_encounter_results, f, indent=2)
        
        # Format for evaluation
        formatted_output_file = os.path.join(
            self.args.output_dir, 
            f"{self.args.dataset_name}_data_cvqa_rag_formatted_{timestamp}.json"
        )
        
        formatted_predictions = self.pipeline.format_results_for_evaluation(all_encounter_results, formatted_output_file)
        
        print(f"Complete results saved to: {complete_output_file}")
        print(f"Formatted predictions saved to: {formatted_output_file}")
        print(f"Processed {len(formatted_predictions)} encounters successfully")
        
        return all_encounter_results, formatted_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnosis-based RAG Medical Analysis Pipeline")
    parser.add_argument("--mode", choices=["all", "sample", "single"], default="sample",
                       help="Run mode: all encounters, sample, or single encounter")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to process in sample mode")
    parser.add_argument("--encounter_id", type=str,
                       help="Specific encounter ID to process in single mode")
    parser.add_argument("--use_test", action="store_true", default=True,
                       help="Use test dataset (default: True)")
    parser.add_argument("--use_finetuned", action="store_true", default=True,
                       help="Use fine-tuned model predictions (default: True)")
    
    cmd_args = parser.parse_args()
    
    # Initialize configuration
    args = Args(use_finetuning=cmd_args.use_finetuned, use_test_dataset=cmd_args.use_test)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment variables
    load_dotenv()
    
    if cmd_args.mode == "all":
        print("Running pipeline on all encounters...")
        run_diagnosis_based_pipeline_all_encounters(args)
        
    elif cmd_args.mode == "sample":
        print(f"Running pipeline on {cmd_args.num_samples} sample encounters...")
        run_diagnosis_based_pipeline_sample(args, num_samples=cmd_args.num_samples)
        
    elif cmd_args.mode == "single":
        if not cmd_args.encounter_id:
            print("Error: --encounter_id required for single mode")
            exit(1)
            
        print(f"Running pipeline on single encounter: {cmd_args.encounter_id}")
        
        # Load data
        model_predictions_dict = DataLoader.load_all_model_predictions(args)
        if not model_predictions_dict:
            print("No model predictions found. Exiting.")
            exit(1)
            
        all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
        validation_df = DataLoader.load_validation_dataset(args)
        
        # Create agentic data manager
        agentic_data = AgenticRAGData(all_models_df, validation_df)
        
        # Initialize pipeline
        pipeline = AgenticDermatologyPipeline(args=args)
        
        # Process single encounter
        results = pipeline.process_single_encounter(agentic_data, cmd_args.encounter_id)
        
        if results:
            # Format and save results
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                args.output_dir,
                f"diagnosis_rag_single_{cmd_args.encounter_id}_{timestamp}.json"
            )
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
                
            print(f"Results saved to: {output_file}")
        else:
            print(f"No results for encounter {cmd_args.encounter_id}")
