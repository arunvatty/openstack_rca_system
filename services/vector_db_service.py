#!/usr/bin/env python3

import os
import sys
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    @staticmethod
    def static_reset_chroma_db(config):
        """Static method to reset ChromaDB database completely by deleting and recreating the directory."""
        try:
            import shutil
            import os
            persist_dir = config['persist_directory']
            if os.path.exists(persist_dir):
                logger.info(f"Removing ChromaDB directory: {persist_dir}")
                shutil.rmtree(persist_dir)
                logger.info("✅ ChromaDB directory removed completely")
            os.makedirs(persist_dir, exist_ok=True)
            logger.info(f"✅ Recreated ChromaDB directory: {persist_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            return False
    
    def _initialize_database(self):
        """Initialize database with ChromaDB fallback to in-memory"""
        self.use_chromadb = False
        
        # Try ChromaDB first
        try:
            import chromadb
            
            # Create persist directory if it doesn't exist
            persist_dir = self.config['persist_directory']
            os.makedirs(persist_dir, exist_ok=True)
            
            # Initialize ChromaDB client with new non-deprecated syntax
            self.client = chromadb.PersistentClient(path=persist_dir)
            
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
        """Add logs to the vector database with robust deduplication"""
        if logs_df.empty:
            logger.warning("No logs to add to vector database")
            return 0
        
        logger.info(f"Starting VectorDB ingestion: {len(logs_df)} logs to process")
        
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
                    logger.info(f"Found {len(existing_ids)} existing documents in VectorDB")
            except Exception:
                logger.info("No existing documents found in VectorDB")
                pass  # Collection might be empty
            
            # Track duplicates for reporting
            duplicates_found = 0
            processed_count = 0
            
            for idx, row in logs_df.iterrows():
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(logs_df)} logs...")
                
                # Prepare log text
                log_text = self._prepare_log_text(row.to_dict())
                
                # Create robust primary key using multiple fields
                timestamp = str(row.get('timestamp', ''))
                service_type = str(row.get('service_type', ''))
                level = str(row.get('level', ''))
                message = str(row.get('message', ''))
                instance_id = str(row.get('instance_id', ''))
                
                # Create a unique identifier that includes all identifying fields
                # Include the DataFrame index to ensure uniqueness
                primary_key_parts = [
                    str(idx),  # DataFrame index for uniqueness
                    timestamp,
                    service_type,
                    level,
                    message[:100],  # First 100 chars of message
                    instance_id
                ]
                primary_key = "_".join(primary_key_parts)
                
                # Create a hash-based ID for ChromaDB
                import hashlib
                log_hash = hashlib.md5(primary_key.encode()).hexdigest()
                
                # Handle chunking for long texts
                if enable_chunking and len(log_text) > self.config['chunk_size']:
                    chunks = self._chunk_text(log_text)
                    logger.debug(f"Chunked log {idx} into {len(chunks)} pieces")
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        # Create unique ID for chunk
                        chunk_id = f"log_{log_hash}_chunk_{chunk_idx}"
                        
                        # Skip if ID already exists
                        if chunk_id in existing_ids:
                            duplicates_found += 1
                            continue
                        
                        # Prepare metadata for chunk
                        metadata = {
                            'timestamp': timestamp,
                            'service_type': service_type,
                            'level': level,
                            'instance_id': instance_id,
                            'original_index': str(idx),
                            'chunk_index': str(chunk_idx),
                            'total_chunks': str(len(chunks)),
                            'is_chunked': 'true',
                            'primary_key': primary_key,
                            'log_hash': log_hash
                        }
                        
                        documents.append(chunk)
                        metadatas.append(metadata)
                        ids.append(chunk_id)
                        existing_ids.add(chunk_id)
                else:
                    # Create unique ID for single log entry
                    log_id = f"log_{log_hash}"
                    
                    # Skip if ID already exists
                    if log_id in existing_ids:
                        duplicates_found += 1
                        if duplicates_found <= 5:  # Log first 5 duplicates for debugging
                            logger.debug(f"Duplicate found for log {idx}: {log_id}")
                        continue
                    
                    # Prepare metadata
                    metadata = {
                        'timestamp': timestamp,
                        'service_type': service_type,
                        'level': level,
                        'instance_id': instance_id,
                        'original_index': str(idx),
                        'chunk_index': '0',
                        'total_chunks': '1',
                        'is_chunked': 'false',
                        'primary_key': primary_key,
                        'log_hash': log_hash
                    }
                    
                    documents.append(log_text)
                    metadatas.append(metadata)
                    ids.append(log_id)
                    existing_ids.add(log_id)
            
            logger.info(f"Processing complete: {len(documents)} new documents, {duplicates_found} duplicates skipped")
            
            if not documents:
                logger.info(f"All documents already exist in vector database (skipped {duplicates_found} duplicates)")
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
                logger.info(f"Successfully added {len(documents)} documents to vector database (skipped {duplicates_found} duplicates)")
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
                    
                    # FIX: Handle ChromaDB distances that can exceed 1.0
                    # ChromaDB can return distances > 1.0 due to normalization issues
                    # Normalize similarity to be non-negative and properly scaled
                    if distance <= 1.0:
                        # Normal case: distance 0-1, similarity 1-0
                        similarity = 1.0 - distance
                    else:
                        # ChromaDB returned distance > 1.0 (normalization issue)
                        # Clamp to 0 similarity (completely dissimilar)
                        similarity = 0.0
                    
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
        """Get historical context for an issue, deduplicating log messages"""
        try:
            similar_logs = self.search_similar_logs(issue_description, top_k=top_k)
            if not similar_logs:
                return ""
            
            # Deduplicate logs by (message, service, level, timestamp)
            seen = set()
            unique_logs = []
            for log in similar_logs:
                key = (
                    log['document'],
                    log['metadata'].get('service_type', ''),
                    log['metadata'].get('level', ''),
                    log['metadata'].get('timestamp', '')
                )
                if key not in seen:
                    seen.add(key)
                    unique_logs.append(log)
                if len(unique_logs) >= (top_k or 5):
                    break

            # Format historical context
            context_parts = []
            for i, log in enumerate(unique_logs):
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


class VectorDBQueryTool:
    """CLI utility for querying the vector database"""
    
    def __init__(self):
        try:
            self.vector_db = VectorDBService()
            logger.info("VectorDBQueryTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VectorDBQueryTool: {e}")
            raise
    
    def search_similar_logs(self, query: str, top_k: int = 10, 
                          filter_metadata: Dict = None, include_chunks: bool = False) -> List[Dict]:
        """Search for logs similar to the query"""
        try:
            similar_logs = self.vector_db.search_similar_logs(
                query, top_k=top_k, filter_metadata=filter_metadata, include_chunks=include_chunks
            )
            return similar_logs
        except Exception as e:
            logger.error(f"Failed to search similar logs: {e}")
            return []
    
    def get_historical_context(self, issue_description: str, top_k: int = 3) -> str:
        """Get historical context for an issue"""
        try:
            context = self.vector_db.get_context_for_issue(issue_description, top_k=top_k)
            return context
        except Exception as e:
            logger.error(f"Failed to get historical context: {e}")
            return ""
    
    def search_by_service(self, service_name: str, top_k: int = 10) -> List[Dict]:
        """Search logs by service name"""
        filter_metadata = {"service": service_name}
        return self.search_similar_logs(f"service {service_name}", top_k=top_k, filter_metadata=filter_metadata)
    
    def search_by_level(self, log_level: str, top_k: int = 10) -> List[Dict]:
        """Search logs by log level"""
        filter_metadata = {"level": log_level.upper()}
        return self.search_similar_logs(f"level {log_level}", top_k=top_k, filter_metadata=filter_metadata)
    
    def search_by_instance(self, instance_id: str, top_k: int = 10) -> List[Dict]:
        """Search logs by instance ID"""
        return self.search_similar_logs(f"instance {instance_id}", top_k=top_k)
    
    def export_collection_to_csv(self, output_file: str) -> bool:
        """Export collection to CSV file"""
        try:
            results = self.vector_db.collection.get()
            if not results['documents']:
                logger.warning("No documents found in collection")
                return False
            
            df = pd.DataFrame({
                'document': results['documents'],
                'metadata': results['metadatas']
            })
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} documents to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
            return False
    
    def get_service_distribution(self) -> Dict:
        """Get distribution of services in the collection"""
        try:
            results = self.vector_db.collection.get()
            services = {}
            for metadata in results['metadatas']:
                service = metadata.get('service', 'unknown')
                services[service] = services.get(service, 0) + 1
            return dict(sorted(services.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Failed to get service distribution: {e}")
            return {}
    
    def get_level_distribution(self) -> Dict:
        """Get distribution of log levels in the collection"""
        try:
            results = self.vector_db.collection.get()
            levels = {}
            for metadata in results['metadatas']:
                level = metadata.get('level', 'unknown')
                levels[level] = levels.get(level, 0) + 1
            return dict(sorted(levels.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Failed to get level distribution: {e}")
            return {}
    
    def clear_collection(self):
        """Clear the collection"""
        self.vector_db.clear_collection()


def print_similar_logs(similar_logs: List[Dict], query_info: str = ""):
    """Print similar logs in a formatted way"""
    if not similar_logs:
        print("❌ No similar logs found")
        return
    
    print(f"\n🔍 Found {len(similar_logs)} similar logs" + (f" for {query_info}" if query_info else ""))
    print("=" * 80)
    
    for i, log in enumerate(similar_logs, 1):
        print(f"\n📄 Log {i}")
        print(f"🎯 Similarity: {log.get('similarity', 0.0):.3f}")
        print(f"📅 Timestamp: {log.get('timestamp', 'N/A')}")
        print(f"🔧 Service: {log.get('service', 'N/A')}")
        print(f"📊 Level: {log.get('level', 'N/A')}")
        print(f"📝 Message: {log.get('message', '')[:200]}{'...' if len(log.get('message', '')) > 200 else ''}")
        print("-" * 80)


def print_distribution(title: str, distribution: Dict):
    """Print distribution data"""
    if not distribution:
        print("❌ No distribution data available")
        return
    
    print(f"\n📊 {title}")
    print("=" * 50)
    total = sum(distribution.values())
    
    for item, count in distribution.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{item:20} | {count:6} ({percentage:5.1f}%)")
    
    print(f"{'Total':20} | {total:6} (100.0%)")


def main():
    """CLI main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vector Database Service with CLI functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get collection statistics
  python3 services/vector_db_service.py --action stats
  
  # Search for similar logs
  python3 services/vector_db_service.py --action search --query "database timeout" --top-k 10
  
  # Get historical context
  python3 services/vector_db_service.py --action context --query "instance launch failed" --top-k 5
  
  # Export collection to CSV
  python3 services/vector_db_service.py --action export --output logs_export.csv
        """
    )
    
    parser.add_argument('--action', required=True,
                       choices=['stats', 'search', 'context', 'service', 'level', 'instance', 
                               'export', 'service-dist', 'level-dist', 'clear'],
                       help='Action to perform')
    
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--service', type=str, help='Service name to filter by')
    parser.add_argument('--level', type=str, help='Log level to filter by')
    parser.add_argument('--instance', type=str, help='Instance ID to filter by')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top results (default: 10)')
    parser.add_argument('--output', type=str, default='vector_db_export.csv', 
                       help='Output file for export (default: vector_db_export.csv)')
    
    args = parser.parse_args()
    
    try:
        tool = VectorDBQueryTool()
        
        if args.action == "stats":
            stats = tool.vector_db.get_collection_stats()
            print("\n📊 Vector Database Statistics")
            print("=" * 50)
            for key, value in stats.items():
                print(f"{key:25} | {value}")
                
        elif args.action == "search":
            if not args.query:
                print("❌ Error: --query is required for search action")
                return
            similar_logs = tool.search_similar_logs(args.query, args.top_k)
            print_similar_logs(similar_logs, f"query='{args.query}'")
            
        elif args.action == "context":
            if not args.query:
                print("❌ Error: --query is required for context action")
                return
            context = tool.get_historical_context(args.query, args.top_k)
            if context:
                print(f"\n📚 Historical Context for: '{args.query}'")
                print("=" * 80)
                print(context)
            else:
                print("❌ No historical context found")
                
        elif args.action == "service":
            if not args.service:
                print("❌ Error: --service is required for service action")
                return
            similar_logs = tool.search_by_service(args.service, args.top_k)
            print_similar_logs(similar_logs, f"service={args.service}")
            
        elif args.action == "level":
            if not args.level:
                print("❌ Error: --level is required for level action")
                return
            similar_logs = tool.search_by_level(args.level, args.top_k)
            print_similar_logs(similar_logs, f"level={args.level}")
            
        elif args.action == "instance":
            if not args.instance:
                print("❌ Error: --instance is required for instance action")
                return
            similar_logs = tool.search_by_instance(args.instance, args.top_k)
            print_similar_logs(similar_logs, f"instance={args.instance}")
            
        elif args.action == "export":
            success = tool.export_collection_to_csv(args.output)
            if success:
                print(f"✅ Successfully exported to {args.output}")
            else:
                print("❌ Failed to export collection")
                
        elif args.action == "service-dist":
            distribution = tool.get_service_distribution()
            print_distribution("Service Distribution", distribution)
            
        elif args.action == "level-dist":
            distribution = tool.get_level_distribution()
            print_distribution("Log Level Distribution", distribution)
            
        elif args.action == "clear":
            confirm = input("⚠️  Are you sure you want to clear the collection? (yes/no): ")
            if confirm.lower() == 'yes':
                tool.clear_collection()
                print("✅ Collection cleared successfully")
            else:
                print("❌ Operation cancelled")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 