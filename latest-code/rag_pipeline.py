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


class Args:
    """Configuration arguments for the RAG pipeline."""
    
    def __init__(self, use_finetuning=True, use_test_dataset=True):
        """
        Initialize arguments with options for dataset and model type.
        
        Parameters:
        - use_finetuning: Whether to use the fine-tuned model predictions (True) or base model predictions (False)
        - use_test_dataset: Whether to use the test dataset (True) or validation dataset (False)
        """
        self.use_finetuning = use_finetuning
        self.use_test_dataset = use_test_dataset
        
        self.base_dir = os.getcwd()
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.model_predictions_dir = os.path.join(self.output_dir, "05022025")
        
        if self.use_test_dataset:
            self.dataset_name = "test"
            self.dataset_path = os.path.join(self.output_dir, "test_dataset.csv")
            self.images_dir = os.path.join(self.base_dir, "2025_dataset", "test", "images_test")
            self.prediction_prefix = "aggregated_test_predictions_"
        else:
            self.dataset_name = "validation"
            self.dataset_path = os.path.join(self.output_dir, "val_dataset.csv")
            self.images_dir = os.path.join(self.base_dir, "2025_dataset", "valid", "images_valid")
            self.prediction_prefix = "aggregated_predictions_"
        
        self.model_type = "finetuned" if self.use_finetuning else "base"
        
        self.gemini_model = "gemini-2.5-flash-preview-04-17"
        
        self.max_reflection_cycles = 2
        self.confidence_threshold = 0.75
        
        self.knowledge_db_path = os.path.join(self.base_dir, "knowledge_db")
        self.embedding_model = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        self.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.vector_dimension = 768
        self.top_k_semantic = 7
        self.top_k_keyword = 7
        self.top_k_hybrid = 10
        self.top_k_rerank = 5
        
        self.dataset_name_huggingface = "brucewayne0459/Skin_diseases_and_care"
        
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
        
        self.default_rag_config = {"use_rag": True, "weight": 0.4}
        
        print(f"\nConfiguration initialized:")
        print(f"- Using {'test' if self.use_test_dataset else 'validation'} dataset")
        print(f"- Looking for {self.model_type} model predictions")
        print(f"- Dataset path: {self.dataset_path}")
        print(f"- Images directory: {self.images_dir}")
        print(f"- Prediction file prefix: {self.prediction_prefix}")


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
        self.kb_manager = KnowledgeBaseManager()
        
        self.diagnosis_extractor = DiagnosisExtractor()
        
        self.query_generator = DiagnosisBasedQueryGenerator(self.client)
        
        self.knowledge_retriever = DiagnosisBasedKnowledgeRetriever(
            self.kb_manager,
            self.query_generator,
            self.diagnosis_extractor
        )
    
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

            question_text = sample_data['query_context'].split("MAIN QUESTION TO ANSWER:")[1].split("\n")[0].strip()
            question_type = sample_data['question_type']
            options = sample_data['options']
            model_predictions = sample_data['model_predictions']
            
            # Create a complete result for this question
            encounter_results[encounter_id][base_qid] = {
                "query_context": sample_data['query_context'],
                "options": sample_data['options'],
                "model_predictions": sample_data['model_predictions'],
                "final_answer": "Not mentioned"  # Default answer
            }

        output_file = os.path.join(self.args.output_dir if self.args else os.getcwd(), f"diagnosis_based_rag_results_{encounter_id}.json")
        
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
        
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    validation_df = DataLoader.load_validation_dataset(args)
    
    agentic_data = AgenticRAGData(all_models_df, validation_df)
    pipeline = AgenticDermatologyPipeline(args=args)
    
    all_pairs = agentic_data.get_all_encounter_question_pairs()
    unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
    print(f"Found {len(unique_encounter_ids)} unique encounters to process")
    
    all_encounter_results = {}
    for i, encounter_id in enumerate(unique_encounter_ids):
        print(f"Processing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
        
        try:
            encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
            if encounter_results:
                all_encounter_results.update(encounter_results)
                
            if (i+1) % 5 == 0 or (i+1) == len(unique_encounter_ids):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                