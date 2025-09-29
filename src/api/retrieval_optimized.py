import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
import pickle
import re
from collections import defaultdict, Counter
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedSolutionRetriever:
    """
    Highly optimized RAG system with GPU support and minimal computational overhead:
    - Fast TF-IDF with optimized parameters
    - Lightweight entity extraction using regex
    - Simple but effective relationship mapping
    - GPU acceleration support for Apple Silicon and NVIDIA
    - Memory-efficient indexing
    """
    
    def __init__(self, use_gpu: bool = True):
        # Optimized TF-IDF with reduced feature space
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Reduced from 5000
            stop_words='english',
            ngram_range=(1, 2),  # Reduced from (1, 3)
            min_df=2,
            max_df=0.9,
            lowercase=True,
            strip_accents='ascii'
        )
        
        # GPU acceleration setup
        self.use_gpu = use_gpu
        self.device = self._setup_gpu()
        
        # Data storage - optimized for memory
        self.solutions_data = []
        self.solution_vectors = None
        self.solution_texts = []
        self.metadata_index = defaultdict(list)
        
        # Lightweight entity and relationship mapping
        self.entity_map = defaultdict(set)  # entity -> solution_ids
        self.product_category_map = defaultdict(set)  # product -> categories
        self.category_solutions = defaultdict(list)  # category -> solution_ids
        
        # Fast entity patterns (optimized regex)
        self.entity_patterns = {
            'errors': r'\b(?:error|err)[-_]?\w*[-_]?\d+\b',
            'products': r'\b(?:datasync|pro|enterprise|basic|premium)\b',
            'tech_terms': r'\b(?:database|connection|timeout|api|server)\b',
            'codes': r'\b\d{3,4}\b'  # Simple numeric codes
        }
        
        self.is_indexed = False
        
    def _setup_gpu(self) -> str:
        """
        Setup GPU acceleration for Apple Silicon or NVIDIA
        """
        if not self.use_gpu:
            return 'cpu'
            
        try:
            # Check for Apple Silicon GPU (Metal Performance Shaders)
            import platform
            if platform.system() == 'Darwin' and platform.processor() == 'arm':
                try:
                    import tensorflow as tf
                    if tf.config.list_physical_devices('GPU'):
                        logger.info("Using Apple Silicon GPU acceleration")
                        return 'mps'
                except ImportError:
                    pass
            
            # Check for NVIDIA GPU
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name()}")
                    return 'cuda'
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
        
        logger.info("Using CPU acceleration")
        return 'cpu'
    
    def load_or_create_index(self):
        """
        Load existing index or create new one from training data
        """
        index_path = "artifacts/optimized_retrieval_index.pkl"
        
        if os.path.exists(index_path):
            logger.info("Loading existing optimized retrieval index")
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
                self.solutions_data = index_data['solutions_data']
                self.solution_vectors = index_data['solution_vectors']
                self.solution_texts = index_data['solution_texts']
                self.vectorizer = index_data['vectorizer']
                self.metadata_index = index_data['metadata_index']
                self.entity_map = index_data.get('entity_map', defaultdict(set))
                self.product_category_map = index_data.get('product_category_map', defaultdict(set))
                self.category_solutions = index_data.get('category_solutions', defaultdict(list))
                self.is_indexed = True
            logger.info(f"Loaded optimized index with {len(self.solutions_data)} solutions")
        else:
            logger.info("Creating new optimized retrieval index from training data")
            self.create_index_from_data()
    
    def create_index_from_data(self):
        """
        Create optimized retrieval index from training data
        """
        try:
            df = pd.read_csv("data/splits/train.csv")
            logger.info(f"Creating optimized index from {len(df)} support tickets")
            self.index_solutions_optimized(df)
        except FileNotFoundError:
            logger.warning("Training data not found. Creating empty index.")
            self.solutions_data = []
            self.solution_vectors = np.array([]).reshape(0, 0)
            self.solution_texts = []
            self.is_indexed = True
    
    def preprocess_text_fast(self, text: str) -> str:
        """
        Fast text preprocessing
        """
        if pd.isna(text) or text is None:
            return ""
        return str(text).lower().strip()
    
    def extract_entities_fast(self, text: str) -> Dict[str, set]:
        """
        Fast entity extraction using optimized regex
        """
        entities = {}
        text_lower = text.lower()
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = set(re.findall(pattern, text_lower))
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def index_solutions_optimized(self, solutions_df: pd.DataFrame):
        """
        Optimized solution indexing with minimal overhead
        """
        logger.info(f"Indexing {len(solutions_df)} solutions with optimized approach")
        
        # Prepare data in batches for memory efficiency
        batch_size = 1000
        self.solutions_data = []
        self.solution_texts = []
        
        for batch_start in range(0, len(solutions_df), batch_size):
            batch_end = min(batch_start + batch_size, len(solutions_df))
            batch_df = solutions_df.iloc[batch_start:batch_end]
            
            for idx, row in batch_df.iterrows():
                # Fast text processing
                subject = self.preprocess_text_fast(row.get('subject', ''))
                description = self.preprocess_text_fast(row.get('description', ''))
                error_logs = self.preprocess_text_fast(row.get('error_logs', ''))
                
                # Create combined text
                combined_text = f"{subject} {description} {error_logs}"
                self.solution_texts.append(combined_text)
                
                # Fast entity extraction
                entities = self.extract_entities_fast(combined_text)
                
                # Store minimal solution data
                solution_data = {
                    'id': idx,
                    'ticket_id': row.get('ticket_id', f'TK-{idx}'),
                    'product': row.get('product', 'Unknown'),
                    'category': row.get('category', 'Unknown'),
                    'subcategory': row.get('subcategory', 'Unknown'),
                    'priority': row.get('priority', 'medium'),
                    'subject': subject,
                    'description': description[:200],  # Truncate for memory
                    'resolution_helpful': row.get('resolution_helpful', True),
                    'satisfaction_score': row.get('satisfaction_score', 4),
                    'entities': entities
                }
                
                self.solutions_data.append(solution_data)
                
                # Build fast lookup indices
                category = solution_data['category']
                product = solution_data['product']
                
                self.metadata_index['category'].append((idx, category))
                self.metadata_index['product'].append((idx, product))
                self.category_solutions[category].append(idx)
                self.product_category_map[product].add(category)
                
                # Index entities for fast lookup
                for entity_type, entity_set in entities.items():
                    for entity in entity_set:
                        self.entity_map[entity].add(idx)
        
        # Create TF-IDF vectors with GPU acceleration if available
        if self.solution_texts:
            logger.info("Creating TF-IDF vectors...")
            self.solution_vectors = self.vectorizer.fit_transform(self.solution_texts)
            
            # Convert to GPU if available
            if self.device == 'cuda':
                try:
                    import torch
                    self.solution_vectors = torch.sparse_coo_tensor(
                        indices=torch.from_numpy(np.vstack([self.solution_vectors.row, self.solution_vectors.col])),
                        values=torch.from_numpy(self.solution_vectors.data),
                        size=self.solution_vectors.shape
                    ).cuda()
                    logger.info("Moved TF-IDF vectors to CUDA")
                except Exception as e:
                    logger.warning(f"CUDA acceleration failed: {e}")
            
            logger.info(f"Created optimized TF-IDF vectors with shape: {self.solution_vectors.shape}")
        
        self.is_indexed = True
        self.save_index_optimized()
        logger.info(f"Successfully indexed {len(self.solutions_data)} solutions with optimization")
    
    def search_solutions_fast(self, query: str, 
                             category: Optional[str] = None,
                             subcategory: Optional[str] = None,
                             product: Optional[str] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fast hybrid search with minimal computational overhead
        """
        if not self.is_indexed:
            self.load_or_create_index()
        
        if len(self.solutions_data) == 0:
            return []
        
        # Fast preprocessing
        processed_query = self.preprocess_text_fast(query)
        query_entities = self.extract_entities_fast(processed_query)
        
        # 1. Fast TF-IDF search
        keyword_scores = self._fast_keyword_search(processed_query)
        
        # 2. Fast entity matching
        entity_scores = self._fast_entity_search(query_entities)
        
        # 3. Fast category/product boosting
        category_scores = self._fast_category_search(category, product)
        
        # Combine scores efficiently
        combined_scores = self._combine_scores_fast(keyword_scores, entity_scores, category_scores)
        
        # Get top candidates
        top_indices = sorted(combined_scores.keys(), 
                           key=lambda x: combined_scores[x], reverse=True)[:limit * 2]
        
        # Apply filters
        filtered_indices = self._apply_filters_fast(top_indices, category, subcategory, product)
        
        # Build results
        results = []
        for idx in filtered_indices[:limit]:
            if idx >= len(self.solutions_data):
                continue
                
            solution_data = self.solutions_data[idx]
            
            result = {
                'solution_id': solution_data['ticket_id'],
                'title': solution_data['subject'],
                'content': solution_data['description'],
                'relevance_score': float(combined_scores.get(idx, 0.0)),
                'success_rate': self._calculate_success_rate_fast(solution_data),
                'category': solution_data['category'],
                'subcategory': solution_data['subcategory'],
                'product': solution_data['product'],
                'keyword_score': keyword_scores.get(idx, 0.0),
                'entity_score': entity_scores.get(idx, 0.0),
                'category_score': category_scores.get(idx, 0.0)
            }
            results.append(result)
        
        # Fast re-ranking
        results.sort(key=lambda x: x['relevance_score'] * 0.7 + x['success_rate'] * 0.3, reverse=True)
        
        return results
    
    def _fast_keyword_search(self, query: str) -> Dict[int, float]:
        """
        Fast TF-IDF search with GPU acceleration
        """
        query_vector = self.vectorizer.transform([query])
        
        if self.device == 'cuda' and hasattr(self.solution_vectors, 'cuda'):
            # GPU-accelerated similarity computation
            try:
                import torch
                query_tensor = torch.sparse_coo_tensor(
                    indices=torch.from_numpy(np.vstack([query_vector.row, query_vector.col])),
                    values=torch.from_numpy(query_vector.data),
                    size=query_vector.shape
                ).cuda()
                
                similarities = torch.sparse.mm(query_tensor, self.solution_vectors.t()).cpu().numpy().flatten()
            except Exception:
                similarities = cosine_similarity(query_vector, self.solution_vectors).flatten()
        else:
            similarities = cosine_similarity(query_vector, self.solution_vectors).flatten()
        
        return {i: float(score) for i, score in enumerate(similarities) if score > 0.01}
    
    def _fast_entity_search(self, query_entities: Dict[str, set]) -> Dict[int, float]:
        """
        Fast entity-based search
        """
        entity_scores = defaultdict(float)
        
        for entity_type, entities in query_entities.items():
            for entity in entities:
                if entity in self.entity_map:
                    for solution_id in self.entity_map[entity]:
                        entity_scores[solution_id] += 1.0
        
        # Normalize
        if entity_scores:
            max_score = max(entity_scores.values())
            return {k: v / max_score for k, v in entity_scores.items()}
        return {}
    
    def _fast_category_search(self, category: Optional[str], product: Optional[str]) -> Dict[int, float]:
        """
        Fast category/product-based boosting
        """
        category_scores = defaultdict(float)
        
        if category and category in self.category_solutions:
            for solution_id in self.category_solutions[category]:
                category_scores[solution_id] += 0.5
        
        if product:
            # Find solutions in categories related to this product
            related_categories = self.product_category_map.get(product, set())
            for cat in related_categories:
                for solution_id in self.category_solutions.get(cat, []):
                    category_scores[solution_id] += 0.3
        
        return dict(category_scores)
    
    def _combine_scores_fast(self, keyword_scores: Dict[int, float],
                            entity_scores: Dict[int, float],
                            category_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Fast score combination
        """
        all_indices = set()
        all_indices.update(keyword_scores.keys())
        all_indices.update(entity_scores.keys())
        all_indices.update(category_scores.keys())
        
        combined = {}
        for idx in all_indices:
            combined[idx] = (
                0.6 * keyword_scores.get(idx, 0.0) +
                0.25 * entity_scores.get(idx, 0.0) +
                0.15 * category_scores.get(idx, 0.0)
            )
        
        return combined
    
    def _apply_filters_fast(self, indices: List[int], 
                           category: Optional[str] = None,
                           subcategory: Optional[str] = None,
                           product: Optional[str] = None) -> List[int]:
        """
        Fast filtering
        """
        if not any([category, subcategory, product]):
            return indices
        
        filtered = []
        for idx in indices:
            if idx >= len(self.solutions_data):
                continue
                
            solution = self.solutions_data[idx]
            
            if category and solution['category'].lower() != category.lower():
                continue
            if subcategory and solution['subcategory'].lower() != subcategory.lower():
                continue
            if product and solution['product'].lower() != product.lower():
                continue
            
            filtered.append(idx)
        
        return filtered
    
    def _calculate_success_rate_fast(self, solution_data: Dict[str, Any]) -> float:
        """
        Fast success rate calculation
        """
        resolution_helpful = solution_data.get('resolution_helpful', True)
        satisfaction_score = solution_data.get('satisfaction_score', 4)
        
        return 0.7 * (1.0 if resolution_helpful else 0.0) + 0.3 * (satisfaction_score / 5.0)
    
    def save_index_optimized(self):
        """
        Save optimized index
        """
        os.makedirs("artifacts", exist_ok=True)
        index_path = "artifacts/optimized_retrieval_index.pkl"
        
        # Convert GPU tensors back to CPU for saving
        solution_vectors = self.solution_vectors
        if hasattr(self.solution_vectors, 'cpu'):
            solution_vectors = self.solution_vectors.cpu()
        
        index_data = {
            'solutions_data': self.solutions_data,
            'solution_vectors': solution_vectors,
            'solution_texts': self.solution_texts,
            'vectorizer': self.vectorizer,
            'metadata_index': dict(self.metadata_index),
            'entity_map': {k: list(v) for k, v in self.entity_map.items()},
            'product_category_map': {k: list(v) for k, v in self.product_category_map.items()},
            'category_solutions': dict(self.category_solutions)
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Saved optimized retrieval index to {index_path}")

def main():
    """
    Test the optimized retrieval system
    """
    retriever = OptimizedSolutionRetriever(use_gpu=True)
    retriever.load_or_create_index()
    
    # Test search
    query = "database timeout error DataSync Pro performance"
    results = retriever.search_solutions_fast(query, category="Technical Issue", limit=5)
    
    print("\n=== OPTIMIZED RAG RESULTS ===")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['solution_id']}")
        print(f"   Title: {result['title']}")
        print(f"   Relevance: {result['relevance_score']:.4f}")
        print(f"   Success Rate: {result['success_rate']:.4f}")
        print(f"   Keyword: {result['keyword_score']:.4f}, Entity: {result['entity_score']:.4f}")
        print()

if __name__ == "__main__":
    main()