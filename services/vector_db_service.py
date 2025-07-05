import os
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import re

from config.config import Config

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

logger = logging.getLogger(__name__)

class VectorDBService:
    """Vector database service using ChromaDB for log similarity search and RAG"""
    
    def __init__(self, collection_name: str = None):
        self.config = Config.VECTOR_DB_CONFIG
        self.collection_name = collection_name or self.config['collection_name']
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        
        # Validate embedding dimensions
        self._validate_embedding_dimensions()
        
        # Initialize ChromaDB
        self._initialize_chroma_db()
        
        logger.info(f"VectorDBService initialized with collection: {self.collection_name}")
        logger.info(f"Embedding model: {self.config['embedding_model']} ({self.config['embedding_dimensions']} dimensions)")
        logger.info(f"Distance metric: {self.config['distance_metric']}")
    
    def _validate_embedding_dimensions(self):
        """Validate that the embedding model matches the configured dimensions"""
        try:
            # Get actual embedding dimensions from the model
            test_embedding = self.embedding_model.encode(["test"], convert_to_tensor=False)
            actual_dimensions = test_embedding.shape[1]
            configured_dimensions = self.config['embedding_dimensions']
            
            if actual_dimensions != configured_dimensions:
                logger.warning(f"Embedding model dimensions mismatch: configured {configured_dimensions}, actual {actual_dimensions}")
                # Update the configuration to match the actual model
                self.config['embedding_dimensions'] = actual_dimensions
                logger.info(f"Updated embedding dimensions to {actual_dimensions}")
                
        except Exception as e:
            logger.error(f"Failed to validate embedding dimensions: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Chunk text into smaller pieces for better embedding"""
        chunk_size = chunk_size or self.config['chunk_size']
        chunk_overlap = chunk_overlap or self.config['chunk_overlap']
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                # Find the last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to maximum length"""
        max_length = max_length or self.config['max_text_length']
        
        if len(text) <= max_length:
            return text
        
        # Truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If we can find a space in the last 20%
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def _initialize_chroma_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory if it doesn't exist
            persist_dir = self.config['persist_directory']
            os.makedirs(persist_dir, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception as e:
                # If there's a schema error, try to reset the database
                if "no such column" in str(e) or "schema" in str(e).lower():
                    logger.warning(f"ChromaDB schema error detected: {e}")
                    logger.info("Attempting to reset ChromaDB database...")
                    self._reset_chroma_db()
                    # Try to create collection again
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "OpenStack log embeddings for RCA"}
                    )
                    logger.info(f"Created new collection after reset: {self.collection_name}")
                else:
                    # Regular collection creation
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "OpenStack log embeddings for RCA"}
                    )
                    logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _reset_chroma_db(self):
        """Reset ChromaDB database to fix schema issues"""
        try:
            # Delete the collection if it exists
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass  # Collection might not exist
            
            # Delete the entire database directory to reset schema
            import shutil
            persist_dir = self.config['persist_directory']
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                logger.info(f"Reset ChromaDB database directory: {persist_dir}")
            
            # Recreate the directory
            os.makedirs(persist_dir, exist_ok=True)
            
            # Reinitialize the client
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info("ChromaDB database reset completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            raise
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _prepare_log_text(self, log_entry: Dict) -> str:
        """Prepare log text for embedding by combining relevant fields"""
        message = log_entry.get('message', '')
        service = log_entry.get('service_type', '')
        level = log_entry.get('level', '')
        instance_id = log_entry.get('instance_id', '')
        
        # Combine fields for better semantic understanding
        log_text = f"{service} {level}: {message}"
        if instance_id and pd.notna(instance_id):
            log_text += f" [Instance: {instance_id}]"
        
        # Truncate text if it exceeds maximum length
        log_text = self._truncate_text(log_text)
        
        return log_text
    
    def add_logs(self, logs_df: pd.DataFrame, enable_chunking: bool = False) -> int:
        """Add logs to the vector database"""
        if logs_df.empty:
            logger.warning("No logs to add to vector database")
            return 0
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in logs_df.iterrows():
                # Prepare log text
                log_text = self._prepare_log_text(row.to_dict())
                
                # Handle chunking for long texts
                if enable_chunking and len(log_text) > self.config['chunk_size']:
                    chunks = self._chunk_text(log_text)
                    logger.debug(f"Chunked log {idx} into {len(chunks)} pieces")
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        # Create unique ID for chunk
                        chunk_id = f"log_{idx}_chunk_{chunk_idx}_{hash(chunk) % 1000000}"
                        
                        # Prepare metadata for chunk
                        metadata = {
                            'timestamp': str(row.get('timestamp', '')),
                            'service_type': str(row.get('service_type', '')),
                            'level': str(row.get('level', '')),
                            'instance_id': str(row.get('instance_id', '')),
                            'original_index': str(idx),
                            'chunk_index': str(chunk_idx),
                            'total_chunks': str(len(chunks)),
                            'is_chunked': 'true'
                        }
                        
                        documents.append(chunk)
                        metadatas.append(metadata)
                        ids.append(chunk_id)
                else:
                    # Create unique ID for single log entry
                    log_id = f"log_{idx}_{hash(log_text) % 1000000}"
                    
                    # Prepare metadata
                    metadata = {
                        'timestamp': str(row.get('timestamp', '')),
                        'service_type': str(row.get('service_type', '')),
                        'level': str(row.get('level', '')),
                        'instance_id': str(row.get('instance_id', '')),
                        'original_index': str(idx),
                        'chunk_index': '0',
                        'total_chunks': '1',
                        'is_chunked': 'false'
                    }
                    
                    documents.append(log_text)
                    metadatas.append(metadata)
                    ids.append(log_id)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self._generate_embeddings(documents)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector database")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to add logs to vector database: {e}")
            raise
    
    def search_similar_logs(self, query: str, top_k: int = None, 
                          filter_metadata: Dict = None, 
                          include_chunks: bool = True) -> List[Dict]:
        """Search for logs similar to the query"""
        try:
            top_k = top_k or self.config['top_k_results']
            
            # Truncate query if needed
            query = self._truncate_text(query)
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Prepare where clause
            where_clause = filter_metadata or {}
            if not include_chunks:
                where_clause['is_chunked'] = 'false'
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            similar_logs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Calculate similarity score (convert distance to similarity)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # ChromaDB uses cosine distance
                    
                    similar_logs.append({
                        'document': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'similarity': similarity,
                        'id': results['ids'][0][i]
                    })
                
                # Sort by similarity (highest first)
                similar_logs.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(similar_logs)} similar logs for query: {query[:50]}...")
            return similar_logs
            
        except Exception as e:
            logger.error(f"Failed to search similar logs: {e}")
            return []
    
    def get_context_for_issue(self, issue_description: str, top_k: int = None) -> str:
        """Get historical context for an issue"""
        try:
            similar_logs = self.search_similar_logs(issue_description, top_k=top_k)
            
            if not similar_logs:
                return ""
            
            # Format historical context
            context_parts = []
            for i, log in enumerate(similar_logs[:5]):  # Top 5 most similar
                context_parts.append(f"Historical Log {i+1} (Similarity: {1-log['distance']:.3f}):")
                context_parts.append(f"  {log['document']}")
                context_parts.append(f"  Service: {log['metadata']['service_type']}")
                context_parts.append(f"  Level: {log['metadata']['level']}")
                context_parts.append(f"  Timestamp: {log['metadata']['timestamp']}")
                context_parts.append("")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get historical context: {e}")
            return ""
    
    def get_similarity_scores(self, logs_df: pd.DataFrame, issue_description: str) -> List[float]:
        """Get similarity scores for logs against an issue description"""
        try:
            if logs_df.empty:
                return []
            
            # Prepare log texts
            log_texts = []
            for _, row in logs_df.iterrows():
                log_text = self._prepare_log_text(row.to_dict())
                log_texts.append(log_text)
            
            # Generate embeddings
            log_embeddings = self._generate_embeddings(log_texts)
            query_embedding = self._generate_embeddings([issue_description])[0]
            
            # Calculate cosine similarities
            similarities = []
            for log_emb in log_embeddings:
                similarity = self._cosine_similarity(query_embedding, log_emb)
                similarities.append(similarity)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity scores: {e}")
            return [0.0] * len(logs_df)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def get_config_info(self) -> Dict:
        """Get vector database configuration information"""
        return {
            'embedding_model': self.config['embedding_model'],
            'embedding_dimensions': self.config['embedding_dimensions'],
            'distance_metric': self.config['distance_metric'],
            'chunk_size': self.config['chunk_size'],
            'chunk_overlap': self.config['chunk_overlap'],
            'max_text_length': self.config['max_text_length'],
            'similarity_threshold': self.config['similarity_threshold'],
            'top_k_results': self.config['top_k_results'],
            'collection_name': self.collection_name
        }
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            results = self.collection.get()
            
            total_documents = len(results['documents']) if results['documents'] else 0
            
            # Count chunked vs non-chunked documents
            chunked_count = 0
            non_chunked_count = 0
            
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata.get('is_chunked') == 'true':
                        chunked_count += 1
                    else:
                        non_chunked_count += 1
            
            stats = {
                'total_documents': total_documents,
                'chunked_documents': chunked_count,
                'non_chunked_documents': non_chunked_count,
                'collection_name': self.collection_name,
                'embedding_model': self.config['embedding_model'],
                'embedding_dimensions': self.config['embedding_dimensions'],
                'distance_metric': self.config['distance_metric']
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_chroma_db()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise 