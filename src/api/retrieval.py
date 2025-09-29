import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
import pickle
import re
from collections import defaultdict, Counter
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolutionRetriever:
    """
    Advanced RAG + Graph-RAG retrieval system that combines:
    - Semantic search with embeddings
    - Keyword matching with TF-IDF
    - Entity linking and relationship mapping
    - Graph-based solution discovery
    """
    
    def __init__(self):
        # TF-IDF for keyword matching
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Semantic embeddings model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}. Using TF-IDF only.")
            self.use_embeddings = False
        
        # Data storage
        self.solutions_data = []
        self.solution_vectors = None  # TF-IDF vectors
        self.solution_embeddings = None  # Semantic embeddings
        self.solution_texts = []
        self.metadata_index = defaultdict(list)
        
        # Graph-RAG components
        self.knowledge_graph = nx.Graph()
        self.entity_index = defaultdict(list)  # entity -> solution_ids
        self.relationship_index = defaultdict(list)  # (entity1, entity2) -> solution_ids
        
        # Entity patterns for extraction
        self.entity_patterns = {
            'error_codes': r'\b(?:error|err)[-_]?\w*[-_]?\d+\b|\b\w+[-_]?(?:error|err)\b',
            'product_names': r'\b(?:datasync|pro|enterprise|basic|premium)\b',
            'technical_terms': r'\b(?:database|connection|timeout|api|server|client|sync|backup|restore)\b',
            'version_numbers': r'\bv?\d+\.\d+(?:\.\d+)?\b',
            'status_codes': r'\b(?:200|404|500|503|401|403)\b'
        }
        
        self.is_indexed = False
        
    def load_or_create_index(self):
        """
        Load existing index or create new one from training data
        """
        index_path = "artifacts/retrieval_index.pkl"
        
        if os.path.exists(index_path):
            logger.info("Loading existing retrieval index")
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
                self.solutions_data = index_data['solutions_data']
                self.solution_vectors = index_data['solution_vectors']
                self.solution_texts = index_data['solution_texts']
                self.vectorizer = index_data['vectorizer']
                self.metadata_index = index_data['metadata_index']
                
                # Load new components if available
                self.solution_embeddings = index_data.get('solution_embeddings')
                self.knowledge_graph = index_data.get('knowledge_graph', nx.Graph())
                self.entity_index = defaultdict(list, index_data.get('entity_index', {}))
                self.relationship_index = defaultdict(list, index_data.get('relationship_index', {}))
                self.use_embeddings = index_data.get('use_embeddings', False)
                
                self.is_indexed = True
            logger.info(f"Loaded enhanced index with {len(self.solutions_data)} solutions, "
                       f"{self.knowledge_graph.number_of_nodes()} graph nodes, "
                       f"{len(self.entity_index)} entities")
        else:
            logger.info("Creating new retrieval index from training data")
            self.create_index_from_data()
    
    def create_index_from_data(self):
        """
        Create retrieval index from training data
        """
        try:
            # Load training data as solutions
            df = pd.read_csv("data/splits/train.csv")
            logger.info(f"Creating index from {len(df)} support tickets")
            
            self.index_solutions(df)
            
        except FileNotFoundError:
            logger.warning("Training data not found. Creating empty index.")
            self.solutions_data = []
            self.solution_vectors = np.array([]).reshape(0, 0)
            self.solution_texts = []
            self.is_indexed = True
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better retrieval
        """
        if pd.isna(text) or text is None:
            return ""
        return str(text).lower().strip()
    
    def index_solutions(self, solutions_df: pd.DataFrame):
        """
        Index solutions using hybrid approach:
        - TF-IDF vectors for keyword matching
        - Semantic embeddings for semantic search
        - Entity extraction and graph building
        """
        logger.info(f"Indexing {len(solutions_df)} solutions with hybrid RAG + Graph-RAG")
        
        # Prepare solution data
        self.solutions_data = []
        self.solution_texts = []
        
        for idx, row in solutions_df.iterrows():
            # Combine text fields for vectorization
            subject = self.preprocess_text(row.get('subject', ''))
            description = self.preprocess_text(row.get('description', ''))
            error_logs = self.preprocess_text(row.get('error_logs', ''))
            
            # Create combined text for search
            combined_text = f"{subject} {description} {error_logs}"
            self.solution_texts.append(combined_text)
            
            # Extract entities from text
            entities = self.extract_entities(combined_text)
            
            # Store solution metadata
            solution_data = {
                'id': idx,
                'ticket_id': row.get('ticket_id', f'ticket_{idx}'),
                'product': row.get('product', 'Unknown'),
                'category': row.get('category', 'Unknown'),
                'subcategory': row.get('subcategory', 'Unknown'),
                'priority': row.get('priority', 'medium'),
                'severity': row.get('severity', 'P3'),
                'subject': subject,
                'description': description,
                'error_logs': error_logs,
                'resolution_helpful': row.get('resolution_helpful', True),
                'satisfaction_score': row.get('satisfaction_score', 4),
                'combined_text': combined_text,
                'entities': entities
            }
            
            self.solutions_data.append(solution_data)
            
            # Build metadata index for fast filtering
            self.metadata_index['category'].append((idx, row.get('category', 'Unknown')))
            self.metadata_index['subcategory'].append((idx, row.get('subcategory', 'Unknown')))
            self.metadata_index['product'].append((idx, row.get('product', 'Unknown')))
            
            # Build entity index and knowledge graph
            self._index_entities(idx, entities, solution_data)
        
        # Create TF-IDF vectors
        if self.solution_texts:
            self.solution_vectors = self.vectorizer.fit_transform(self.solution_texts)
            logger.info(f"Created TF-IDF vectors with shape: {self.solution_vectors.shape}")
        else:
            self.solution_vectors = np.array([]).reshape(0, 0)
        
        # Create semantic embeddings
        if self.use_embeddings and self.solution_texts:
            logger.info("Creating semantic embeddings...")
            self.solution_embeddings = self.embedding_model.encode(self.solution_texts)
            logger.info(f"Created semantic embeddings with shape: {self.solution_embeddings.shape}")
        
        # Build relationship mappings
        self._build_relationship_mappings()
        
        self.is_indexed = True
        
        # Save index for future use
        self.save_index()
        
        logger.info(f"Successfully indexed {len(self.solutions_data)} solutions with Graph-RAG")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using regex patterns
        """
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text.lower())
            entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def _index_entities(self, solution_id: int, entities: Dict[str, List[str]], solution_data: Dict[str, Any]):
        """
        Index entities and build knowledge graph
        """
        # Add solution node to graph
        solution_node = f"solution_{solution_id}"
        self.knowledge_graph.add_node(solution_node, 
                                    type='solution',
                                    category=solution_data['category'],
                                    product=solution_data['product'],
                                    priority=solution_data['priority'])
        
        # Add entity nodes and relationships
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity.strip():
                    entity_node = f"{entity_type}_{entity}"
                    
                    # Add entity node
                    self.knowledge_graph.add_node(entity_node, type=entity_type, value=entity)
                    
                    # Add relationship between solution and entity
                    self.knowledge_graph.add_edge(solution_node, entity_node, 
                                                relation='contains')
                    
                    # Index entity for fast lookup
                    self.entity_index[entity].append(solution_id)
        
        # Add product and category as special entities
        product_node = f"product_{solution_data['product']}"
        category_node = f"category_{solution_data['category']}"
        
        self.knowledge_graph.add_node(product_node, type='product', value=solution_data['product'])
        self.knowledge_graph.add_node(category_node, type='category', value=solution_data['category'])
        
        self.knowledge_graph.add_edge(solution_node, product_node, relation='belongs_to')
        self.knowledge_graph.add_edge(solution_node, category_node, relation='categorized_as')
    
    def _build_relationship_mappings(self):
        """
        Build relationship mappings: Products ↔ Issues ↔ Solutions ↔ Historical Tickets
        """
        logger.info("Building relationship mappings...")
        
        # Group solutions by product and category
        product_categories = defaultdict(set)
        category_products = defaultdict(set)
        
        for solution in self.solutions_data:
            product = solution['product']
            category = solution['category']
            
            product_categories[product].add(category)
            category_products[category].add(product)
        
        # Create cross-product relationships in graph
        for product, categories in product_categories.items():
            for category in categories:
                product_node = f"product_{product}"
                category_node = f"category_{category}"
                
                if self.knowledge_graph.has_node(product_node) and self.knowledge_graph.has_node(category_node):
                    self.knowledge_graph.add_edge(product_node, category_node, 
                                                relation='has_issues_in')
        
        # Build entity co-occurrence relationships
        for i, solution1 in enumerate(self.solutions_data):
            for j, solution2 in enumerate(self.solutions_data[i+1:], i+1):
                if i != j:
                    # Find common entities
                    common_entities = self._find_common_entities(solution1['entities'], solution2['entities'])
                    
                    if common_entities:
                        solution1_node = f"solution_{solution1['id']}"
                        solution2_node = f"solution_{solution2['id']}"
                        
                        # Add similarity edge if solutions share entities
                        if (self.knowledge_graph.has_node(solution1_node) and 
                            self.knowledge_graph.has_node(solution2_node)):
                            self.knowledge_graph.add_edge(solution1_node, solution2_node,
                                                        relation='similar_to',
                                                        common_entities=len(common_entities))
        
        logger.info(f"Built knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes and {self.knowledge_graph.number_of_edges()} edges")
    
    def _find_common_entities(self, entities1: Dict[str, List[str]], entities2: Dict[str, List[str]]) -> List[str]:
        """
        Find common entities between two solutions
        """
        common = []
        for entity_type in entities1:
            if entity_type in entities2:
                common.extend(set(entities1[entity_type]) & set(entities2[entity_type]))
        return common
    
    def save_index(self):
        """
        Save the retrieval index to disk
        """
        os.makedirs("artifacts", exist_ok=True)
        index_path = "artifacts/retrieval_index.pkl"
        
        index_data = {
            'solutions_data': self.solutions_data,
            'solution_vectors': self.solution_vectors,
            'solution_embeddings': self.solution_embeddings if self.use_embeddings else None,
            'solution_texts': self.solution_texts,
            'vectorizer': self.vectorizer,
            'metadata_index': self.metadata_index,
            'knowledge_graph': self.knowledge_graph,
            'entity_index': dict(self.entity_index),
            'relationship_index': dict(self.relationship_index),
            'use_embeddings': self.use_embeddings
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Saved enhanced retrieval index to {index_path}")
    
    def search_solutions(self, query: str, 
                        category: Optional[str] = None,
                        subcategory: Optional[str] = None,
                        product: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Advanced hybrid search combining:
        - Keyword matching (TF-IDF)
        - Semantic search (embeddings)
        - Entity-based retrieval
        - Graph-based discovery
        """
        logger.info(f"Searching solutions for query: {query}")
        
        # Ensure index is loaded
        if not self.is_indexed:
            self.load_or_create_index()
        
        if len(self.solutions_data) == 0:
            logger.warning("No solutions indexed")
            return []
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Extract entities from query
        query_entities = self.extract_entities(processed_query)
        
        # 1. Keyword-based search (TF-IDF)
        keyword_scores = self._keyword_search(processed_query)
        
        # 2. Semantic search (if embeddings available)
        semantic_scores = self._semantic_search(processed_query) if self.use_embeddings else {}
        
        # 3. Entity-based search
        entity_scores = self._entity_search(query_entities)
        
        # 4. Graph-based discovery
        graph_scores = self._graph_search(query_entities, category, product)
        
        # Combine all scores
        combined_scores = self._combine_search_scores(
            keyword_scores, semantic_scores, entity_scores, graph_scores
        )
        
        # Get candidate indices sorted by combined score
        candidate_indices = sorted(combined_scores.keys(), 
                                 key=lambda x: combined_scores[x], reverse=True)
        
        # Apply metadata filters
        filtered_indices = self._apply_filters(
            candidate_indices, category, subcategory, product
        )
        
        # Get top results
        top_indices = filtered_indices[:limit * 2]  # Get more candidates for re-ranking
        
        # Build results with enhanced scoring
        solutions = []
        for idx in top_indices:
            if idx >= len(self.solutions_data):
                continue
                
            solution_data = self.solutions_data[idx]
            
            solution = {
                'solution_id': solution_data['ticket_id'],
                'title': solution_data['subject'],
                'content': solution_data['description'][:200] + "..." if len(solution_data['description']) > 200 else solution_data['description'],
                'relevance_score': float(combined_scores.get(idx, 0.0)),
                'success_rate': self._calculate_success_rate(solution_data),
                'category': solution_data['category'],
                'subcategory': solution_data['subcategory'],
                'product': solution_data['product'],
                'entities': solution_data.get('entities', {}),
                'keyword_score': keyword_scores.get(idx, 0.0),
                'semantic_score': semantic_scores.get(idx, 0.0),
                'entity_score': entity_scores.get(idx, 0.0),
                'graph_score': graph_scores.get(idx, 0.0)
            }
            solutions.append(solution)
        
        # Re-rank based on success rates and category relevance
        solutions = self._re_rank_solutions_enhanced(solutions, category)
        
        # Return top results
        final_results = solutions[:limit]
        
        logger.info(f"Found {len(final_results)} relevant solutions using hybrid RAG + Graph-RAG")
        return final_results
    
    def _keyword_search(self, query: str) -> Dict[int, float]:
        """
        Keyword-based search using TF-IDF
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.solution_vectors).flatten()
        
        return {i: float(score) for i, score in enumerate(similarities)}
    
    def _semantic_search(self, query: str) -> Dict[int, float]:
        """
        Semantic search using embeddings
        """
        if not self.use_embeddings or self.solution_embeddings is None:
            return {}
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.solution_embeddings).flatten()
        
        return {i: float(score) for i, score in enumerate(similarities)}
    
    def _entity_search(self, query_entities: Dict[str, List[str]]) -> Dict[int, float]:
        """
        Entity-based search using extracted entities
        """
        entity_scores = defaultdict(float)
        
        for entity_type, entities in query_entities.items():
            for entity in entities:
                if entity in self.entity_index:
                    for solution_id in self.entity_index[entity]:
                        entity_scores[solution_id] += 1.0
        
        # Normalize scores
        max_score = max(entity_scores.values()) if entity_scores else 1.0
        return {k: v / max_score for k, v in entity_scores.items()}
    
    def _graph_search(self, query_entities: Dict[str, List[str]], 
                     category: Optional[str] = None, 
                     product: Optional[str] = None) -> Dict[int, float]:
        """
        Graph-based search using knowledge graph relationships
        """
        graph_scores = defaultdict(float)
        
        # Find solutions connected to query entities through graph
        query_nodes = []
        
        # Add entity nodes from query
        for entity_type, entities in query_entities.items():
            for entity in entities:
                entity_node = f"{entity_type}_{entity}"
                if self.knowledge_graph.has_node(entity_node):
                    query_nodes.append(entity_node)
        
        # Add category and product nodes if specified
        if category:
            category_node = f"category_{category}"
            if self.knowledge_graph.has_node(category_node):
                query_nodes.append(category_node)
        
        if product:
            product_node = f"product_{product}"
            if self.knowledge_graph.has_node(product_node):
                query_nodes.append(product_node)
        
        # Find connected solutions through graph traversal
        for query_node in query_nodes:
            # Get neighbors (solutions connected to this entity)
            neighbors = list(self.knowledge_graph.neighbors(query_node))
            
            for neighbor in neighbors:
                if neighbor.startswith('solution_'):
                    solution_id = int(neighbor.split('_')[1])
                    
                    # Score based on edge type and distance
                    edge_data = self.knowledge_graph.get_edge_data(query_node, neighbor)
                    relation = edge_data.get('relation', 'unknown')
                    
                    if relation == 'contains':
                        graph_scores[solution_id] += 1.0
                    elif relation == 'belongs_to':
                        graph_scores[solution_id] += 0.8
                    elif relation == 'categorized_as':
                        graph_scores[solution_id] += 0.6
                    else:
                        graph_scores[solution_id] += 0.3
        
        # Normalize scores
        max_score = max(graph_scores.values()) if graph_scores else 1.0
        return {k: v / max_score for k, v in graph_scores.items()}
    
    def _combine_search_scores(self, keyword_scores: Dict[int, float],
                              semantic_scores: Dict[int, float],
                              entity_scores: Dict[int, float],
                              graph_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Combine different search scores with weights
        """
        combined_scores = defaultdict(float)
        
        # Get all solution indices
        all_indices = set()
        all_indices.update(keyword_scores.keys())
        all_indices.update(semantic_scores.keys())
        all_indices.update(entity_scores.keys())
        all_indices.update(graph_scores.keys())
        
        # Combine with weights
        weights = {
            'keyword': 0.3,
            'semantic': 0.3 if self.use_embeddings else 0.0,
            'entity': 0.2,
            'graph': 0.2
        }
        
        # Adjust weights if semantic search is not available
        if not self.use_embeddings:
            weights['keyword'] = 0.5
            weights['entity'] = 0.25
            weights['graph'] = 0.25
        
        for idx in all_indices:
            combined_scores[idx] = (
                weights['keyword'] * keyword_scores.get(idx, 0.0) +
                weights['semantic'] * semantic_scores.get(idx, 0.0) +
                weights['entity'] * entity_scores.get(idx, 0.0) +
                weights['graph'] * graph_scores.get(idx, 0.0)
            )
        
        return dict(combined_scores)
    
    def _apply_filters(self, candidate_indices: np.ndarray, 
                      category: Optional[str] = None,
                      subcategory: Optional[str] = None,
                      product: Optional[str] = None) -> List[int]:
        """
        Apply metadata filters to candidate indices
        """
        filtered_indices = []
        
        for idx in candidate_indices:
            if idx >= len(self.solutions_data):
                continue
                
            solution = self.solutions_data[idx]
            
            # Apply filters
            if category and solution['category'].lower() != category.lower():
                continue
            if subcategory and solution['subcategory'].lower() != subcategory.lower():
                continue
            if product and solution['product'].lower() != product.lower():
                continue
            
            filtered_indices.append(idx)
        
        return filtered_indices
    
    def _calculate_success_rate(self, solution_data: Dict[str, Any]) -> float:
        """
        Calculate success rate based on resolution helpfulness and satisfaction score
        """
        resolution_helpful = solution_data.get('resolution_helpful', True)
        satisfaction_score = solution_data.get('satisfaction_score', 4)
        
        # Normalize satisfaction score (1-5 scale) to 0-1
        normalized_satisfaction = satisfaction_score / 5.0 if satisfaction_score > 0 else 0.8
        
        # Weighted combination
        success_rate = 0.7 * (1.0 if resolution_helpful else 0.0) + 0.3 * normalized_satisfaction
        return success_rate
    
    def _re_rank_solutions_enhanced(self, solutions: List[Dict[str, Any]], 
                                   category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced re-ranking based on multiple factors:
        - Relevance scores from different search methods
        - Success rates and resolution quality
        - Category relevance boost
        - Entity matching bonus
        """
        for solution in solutions:
            # Base relevance score (already combined from multiple methods)
            base_score = solution['relevance_score']
            
            # Success rate component
            success_component = solution['success_rate']
            
            # Category relevance boost
            category_boost = 0.0
            if category and solution['category'].lower() == category.lower():
                category_boost = 0.1
            
            # Entity matching bonus (if solution has many matching entities)
            entity_bonus = 0.0
            if 'entity_score' in solution and solution['entity_score'] > 0:
                entity_bonus = min(0.1, solution['entity_score'] * 0.05)
            
            # Graph connectivity bonus
            graph_bonus = 0.0
            if 'graph_score' in solution and solution['graph_score'] > 0:
                graph_bonus = min(0.05, solution['graph_score'] * 0.02)
            
            # Final combined score
            combined_score = (
                0.5 * base_score +           # Relevance from search methods
                0.3 * success_component +    # Historical success rate
                0.1 * category_boost +       # Category match bonus
                0.05 * entity_bonus +        # Entity match bonus
                0.05 * graph_bonus           # Graph connectivity bonus
            )
            
            solution['final_score'] = combined_score
        
        # Sort by final score
        solutions.sort(key=lambda x: x['final_score'], reverse=True)
        
        return solutions
    
    def _re_rank_solutions(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Legacy re-ranking method for backward compatibility
        """
        return self._re_rank_solutions_enhanced(solutions)

def main():
    """
    Main function to demonstrate solution retrieval
    """
    # Initialize retriever
    retriever = SolutionRetriever()
    
    # Load or create index
    retriever.load_or_create_index()
    
    # Search example
    query = "Database sync failing with timeout error"
    results = retriever.search_solutions(query, limit=5)
    
    print("\n=== SOLUTION RETRIEVAL RESULTS ===")
    for i, result in enumerate(results):
        print(f"{i+1}. Solution ID: {result['solution_id']}")
        print(f"   Title: {result['title']}")
        print(f"   Content: {result['content'][:100]}...")
        print(f"   Relevance Score: {result['relevance_score']:.4f}")
        print(f"   Success Rate: {result['success_rate']:.4f}")
        print(f"   Combined Score: {result.get('combined_score', 0):.4f}")
        print()

if __name__ == "__main__":
    main()
