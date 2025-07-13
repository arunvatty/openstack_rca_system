import os
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config.config import Config

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Suppress ChromaDB warnings about duplicate embeddings
import warnings
warnings.filterwarnings("ignore", message=".*existing embedding ID.*", category=UserWarning)

logger = logging.getLogger(__name__)

class InMemoryVectorDB:
    """Fallback in-memory vector database when ChromaDB is not available"""
    
    def __init__(self, collection_name: str = "openstack_logs"):
        self.collection_name = collection_name
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        logger.info(f"InMemoryVectorDB initialized: {collection_name}")
    
    def add(self, embeddings, documents, metadatas, ids):
        """Add documents to in-memory storage"""
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        logger.info(f"Added {len(documents)} documents to in-memory storage")
    
    def query(self, query_embeddings, n_results=20, where=None):
        """Query documents using cosine similarity"""
        if not self.embeddings:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
        
        # Calculate similarities
        similarities = cosine_similarity(query_embeddings, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        # Format results
        results = {
            'documents': [[self.documents[i] for i in top_indices]],
            'metadatas': [[self.metadatas[i] for i in top_indices]],
            'distances': [[1 - similarities[i] for i in top_indices]],  # Convert similarity to distance
            'ids': [[self.ids[i] for i in top_indices]]
        }
        
        return results
    
    def get(self):
        """Get all documents"""
        return {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
    
    def delete_collection(self, name):
        """Delete collection (clear in-memory data)"""
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        logger.info(f"Cleared in-memory collection: {name}")

class VectorDBService:
    """Vector database service using ChromaDB for log similarity search and RAG"""
    
    def __init__(self, collection_name: str = None):
        self.config = Config.VECTOR_DB_CONFIG
        self.collection_name = collection_name or self.config['collection_name']
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        
        # Validate embedding dimensions
        self._validate_embedding_dimensions()
        
        # Initialize database (ChromaDB with fallback to in-memory)
        self._initialize_database()
        
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
    
    def _reset_chroma_db(self):
        """Reset ChromaDB database completely by deleting and recreating"""
        try:
            # Close any existing connections
            if hasattr(self, 'client') and self.client:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception as e:
                    logger.info(f"Collection {self.collection_name} was already deleted or didn't exist")
            
            # Completely remove the ChromaDB directory to fix schema issues
            import shutil
            persist_dir = self.config['persist_directory']
            if os.path.exists(persist_dir):
                logger.info(f"Removing ChromaDB directory: {persist_dir}")
                shutil.rmtree(persist_dir)
                logger.info("✅ ChromaDB directory removed completely")
            
            # Recreate the directory
            os.makedirs(persist_dir, exist_ok=True)
            logger.info(f"✅ Recreated ChromaDB directory: {persist_dir}")
            
            # Reinitialize ChromaDB client
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "OpenStack log embeddings for RCA"}
            )
            logger.info(f"✅ Recreated ChromaDB collection: {self.collection_name}")
            self.use_chromadb = True
            
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            logger.info("Falling back to in-memory vector database")
            self._initialize_in_memory_db()
    
    def _initialize_database(self):
        """Initialize database with ChromaDB fallback to in-memory"""
        self.use_chromadb = False
        
        # Try ChromaDB first
        try:
            import chromadb
            from chromadb.config import Settings
            
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
            
            # Try to get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
                self.use_chromadb = True
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["no such column", "schema", "migration", "version", "collections.topic"]):
                    logger.error(f"ChromaDB schema error detected: {e}")
                    logger.error("❌ ChromaDB schema error: Please run 'main.py --mode vector-db --action reset' to manually reset the database.")
                    logger.info("Falling back to in-memory vector database for this session.")
                    self._initialize_in_memory_db()
                else:
                    # Try to create new collection
                    try:
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            metadata={"description": "OpenStack log embeddings for RCA"}
                        )
                        logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                        self.use_chromadb = True
                    except Exception as create_error:
                        logger.warning(f"Failed to create ChromaDB collection: {create_error}")
                        logger.info("Falling back to in-memory vector database")
                        self._initialize_in_memory_db()
                        
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}")
            logger.info("Using in-memory vector database")
            self._initialize_in_memory_db()
    
    def _initialize_in_memory_db(self):
        """Initialize in-memory vector database"""
        try:
            # Create in-memory collection
            self.collection = InMemoryVectorDB(self.collection_name)
            self.client = None
            self.use_chromadb = False
            logger.info(f"Initialized in-memory vector database: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize in-memory database: {e}")
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
            
            # Get existing IDs to avoid duplicates
            existing_ids = set()
            try:
                existing_results = self.collection.get()
                if existing_results['ids']:
                    existing_ids = set(existing_results['ids'])
            except Exception:
                pass  # Collection might be empty
            
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
                        
                        # Skip if ID already exists
                        if chunk_id in existing_ids:
                            continue
                        
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
                        existing_ids.add(chunk_id)
                else:
                    # Create unique ID for single log entry
                    log_id = f"log_{idx}_{hash(log_text) % 1000000}"
                    
                    # Skip if ID already exists
                    if log_id in existing_ids:
                        continue
                    
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
                    existing_ids.add(log_id)
            
            if not documents:
                logger.info("All documents already exist in vector database")
                return 0
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} new documents...")
            embeddings = self._generate_embeddings(documents)
            
            # Add to collection with error handling for duplicates
            try:
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully added {len(documents)} documents to vector database")
            except Exception as e:
                if "existing embedding ID" in str(e).lower():
                    # Suppress duplicate warnings and log a summary
                    logger.info(f"Some documents already existed. Added {len(documents)} new documents to vector database")
                else:
                    raise
            
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
            
            # Prepare where clause - only include if not empty
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata
            elif not include_chunks:
                where_clause = {'is_chunked': 'false'}
            
            # Search in collection
            search_kwargs = {
                'query_embeddings': [query_embedding],
                'n_results': top_k
            }
            
            # Only add where clause if it's not None
            if where_clause:
                search_kwargs['where'] = where_clause
            
            results = self.collection.query(**search_kwargs)
            
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
                'distance_metric': self.config['distance_metric'],
                'database_type': 'chromadb' if self.use_chromadb else 'in_memory'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            if self.use_chromadb and self.client:
                self.client.delete_collection(self.collection_name)
                # Recreate collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "OpenStack log embeddings for RCA"}
                )
            else:
                # Clear in-memory collection
                self.collection.delete_collection(self.collection_name)
            
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def is_chromadb_available(self) -> bool:
        """Check if ChromaDB is being used"""
        return self.use_chromadb
    
    def get_database_type(self) -> str:
        """Get the type of database being used"""
        return 'chromadb' if self.use_chromadb else 'in_memory' 