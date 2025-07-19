# Vector Database Operations Manual

Complete guide for ChromaDB operations, semantic search, and VectorDB management in the OpenStack RCA system.

## 🚀 Overview

The VectorDB system uses **ChromaDB** with **all-MiniLM-L12-v2** embeddings for semantic log analysis, historical context retrieval, and intelligent similarity search.

### Key Features
- **Semantic search** with 384-dimensional embeddings
- **Historical context** retrieval for RCA analysis
- **Deduplication system** preventing duplicate log ingestion
- **Metadata filtering** for precise queries
- **Scalable architecture** with efficient cosine similarity

## 🏗️ VectorDB Architecture

```
data/vector_db/
├── <collection-id>/
│   ├── data_level0.bin      # Vector embeddings
│   ├── header.bin           # Collection metadata
│   ├── length.bin           # Document lengths
│   └── link_lists.bin       # Hierarchical index
└── chroma.sqlite3           # Metadata database
```

### Components
- **ChromaDB**: Vector database engine
- **Embedding Model**: all-MiniLM-L12-v2 (12-layer transformer)
- **Collection**: openstack_logs (default collection name)
- **Embeddings**: 384-dimensional vectors
- **Distance Metric**: Cosine similarity

## 🔧 VectorDB Management

### 1. Check VectorDB Status
```bash
# Using the VectorDB query utility
cd /path/to/openstack_rca_system
python3 services/vector_db_service.py --action stats

# Output example:
# Collection: openstack_logs
# Total documents: 500
# Embedding model: all-MiniLM-L12-v2
# Dimensions: 384
# Distance metric: cosine
```

### 2. VectorDB Operations via CLI
```bash
# Ingest logs into VectorDB
python3 main.py --mode vector-db --action ingest --logs logs/

# Check VectorDB status
python3 main.py --mode vector-db --action status

# Clean VectorDB collection
python3 main.py --mode vector-db --action clean

# Reset entire VectorDB
python3 main.py --mode vector-db --action reset
```

### 3. Advanced Query Operations
```bash
# Search similar logs
python3 services/vector_db_service.py --action search --query "Database connection timeout" --top-k 10

# Get historical context
python3 services/vector_db_service.py --action context --query "Instance launch failed" --top-k 5

# Filter by service type
python3 services/vector_db_service.py --action service --service nova-compute --top-k 20

# Filter by log level
python3 services/vector_db_service.py --action level --level ERROR --top-k 15

# Export collection to CSV
python3 services/vector_db_service.py --action export --output logs_export.csv
```

## 🔍 VectorDB Query Utility

### Available Actions

| Action | Description | Usage Example |
|--------|-------------|---------------|
| `stats` | Show collection statistics | `--action stats` |
| `search` | Semantic similarity search | `--action search --query "error message"` |
| `context` | Historical context retrieval | `--action context --query "issue description"` |
| `service` | Filter by OpenStack service | `--action service --service nova-compute` |
| `level` | Filter by log level | `--action level --level ERROR` |
| `instance` | Filter by instance ID | `--action instance --instance abc123` |
| `export` | Export to CSV format | `--action export --output logs.csv` |
| `service-dist` | Service distribution analysis | `--action service-dist` |
| `level-dist` | Log level distribution | `--action level-dist` |
| `clear` | Clear collection | `--action clear` |
| `config` | Show configuration info | `--action config` |

### Query Examples

#### 1. Semantic Search
```bash
# Find logs similar to specific error
python3 services/vector_db_service.py \
  --action search \
  --query "No valid host was found. Disk space exhausted" \
  --top-k 10

# Search with metadata filters
python3 services/vector_db_service.py \
  --action search \
  --query "connection timeout" \
  --level ERROR \
  --service nova-compute \
  --top-k 15
```

#### 2. Historical Context
```bash
# Get context for RCA analysis
python3 services/vector_db_service.py \
  --action context \
  --query "Instance spawn failures with resource allocation" \
  --top-k 10

# Context with specific similarity threshold
python3 services/vector_db_service.py \
  --action context \
  --query "Memory allocation errors" \
  --similarity-threshold 0.8
```

#### 3. Data Analysis
```bash
# Service distribution
python3 services/vector_db_service.py --action service-dist
# Output: nova-compute: 45.2%, nova-api: 24.1%, nova-scheduler: 18.3%

# Log level distribution
python3 services/vector_db_service.py --action level-dist
# Output: INFO: 60.3%, ERROR: 24.1%, WARNING: 15.6%

# Export filtered data
python3 services/vector_db_service.py \
  --action export \
  --level ERROR \
  --service nova-compute \
  --output error_logs.csv
```

## 🔄 VectorDB in RCA Workflow

### 1. Log Ingestion Process
```python
# Automatic during training
python3 main.py --mode train
# → Processes logs → Creates embeddings → Stores in VectorDB

# Manual ingestion
from services.vector_db_service import VectorDBService
vector_db = VectorDBService()
logs_added = vector_db.add_logs(dataframe, enable_chunking=False)
```

### 2. Semantic Search in RCA
```python
# During RCA analysis
similar_logs = vector_db.search_similar_logs(
    query="Database connection timeout errors",
    top_k=50,
    filter_metadata={'level': 'ERROR'}
)

# Results include similarity scores and metadata
for log in similar_logs:
    print(f"Similarity: {log['similarity']:.3f}")
    print(f"Message: {log['document']}")
    print(f"Service: {log['metadata']['service_type']}")
```

### 3. Historical Context Retrieval
```python
# Get historical context for RCA
historical_context = vector_db.get_context_for_issue(
    issue_description="Instance launch failures",
    top_k=10
)
# Returns formatted context string for LLM analysis
```

## 📊 VectorDB Performance

### Performance Metrics

| Operation | Time | Scalability |
|-----------|------|-------------|
| **Document Ingestion** | ~2ms per log | Linear |
| **Similarity Search** | ~50-100ms | Logarithmic |
| **Metadata Filtering** | ~10-50ms | Constant |
| **Context Retrieval** | ~100ms | Linear with top_k |

### Embedding Model Comparison

| Model | Layers | Dimensions | Speed | Accuracy |
|-------|---------|-----------|-------|----------|
| **all-MiniLM-L6-v2** | 6 | 384 | 563 texts/sec | Good |
| **all-MiniLM-L12-v2** | 12 | 384 | 318 texts/sec | **Better** |

*Current: Using L12-v2 for superior semantic understanding*

## 🛠️ VectorDB Configuration

### Configuration Settings (config/config.py)
```python
VECTOR_DB_CONFIG = {
    'type': 'chroma',
    'embedding_model': 'all-MiniLM-L12-v2',
    'collection_name': 'openstack_logs',
    'similarity_threshold': 0.7,
    'top_k_results': 20,
    'persist_directory': 'data/vector_db',
    'embedding_dimensions': 384,
    'distance_metric': 'cosine',
    'max_text_length': 1000,
    'chunk_size': 512,
    'chunk_overlap': 50
}
```

### Environment Variables
```bash
# Override VectorDB directory
export VECTOR_DB_DIR="/custom/vectordb/path"

# Disable VectorDB (use fast mode only)
export DISABLE_VECTOR_DB=true

# Custom embedding model
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

## 🔐 VectorDB Security

### Data Privacy
- **Local storage**: All vectors stored locally
- **No external calls**: Embeddings generated locally
- **Metadata protection**: Sensitive data can be excluded

### Access Control
```python
# Restrict sensitive fields
EXCLUDED_METADATA = ['user_id', 'password', 'token', 'secret']

# Filter out sensitive information before ingestion
filtered_logs = logs_df.drop(columns=EXCLUDED_METADATA, errors='ignore')
vector_db.add_logs(filtered_logs)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. ChromaDB Connection Errors
```bash
# Check VectorDB status
python3 services/vector_db_service.py --action stats

# Rebuild VectorDB if corrupted
python3 main.py --mode vector-db --action reset
python3 main.py --mode vector-db --action ingest --logs logs/
```

#### 2. Embedding Model Issues
```bash
# Download embedding model manually
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L12-v2')
print('Model downloaded successfully')
"
```

#### 3. Memory Issues with Large Collections
```bash
# Use chunked processing for large datasets
python3 -c "
from services.vector_db_service import VectorDBService
vector_db = VectorDBService()
# Process in smaller chunks
vector_db.add_logs(df, enable_chunking=True, chunk_size=100)
"
```

#### 4. Search Quality Issues
```bash
# Adjust similarity threshold
python3 services/vector_db_service.py \
  --action search \
  --query "your query" \
  --similarity-threshold 0.6  # Lower threshold for more results

# Check embedding quality
python3 services/vector_db_service.py --action stats
```

### Performance Issues

#### 1. Slow Search Performance
- **Cause**: Large collection size or complex queries
- **Solution**: Use metadata filtering, adjust top_k

#### 2. High Memory Usage
- **Cause**: Large embedding cache
- **Solution**: Process in chunks, restart service periodically

#### 3. Poor Search Results
- **Cause**: Low-quality embeddings or inappropriate threshold
- **Solution**: Adjust similarity threshold, check query phrasing

## ⚡ Performance Optimization

### Search Optimization
```python
# Use metadata filters to narrow search space
results = vector_db.search_similar_logs(
    query="database error",
    top_k=20,
    filter_metadata={
        'level': 'ERROR',
        'service_type': 'nova-api'
    }
)

# Batch queries for multiple searches
queries = ["error 1", "error 2", "error 3"]
results = vector_db.batch_search(queries, top_k=10)
```

### Memory Optimization
```python
# Process large datasets in chunks
def ingest_large_dataset(df, chunk_size=1000):
    vector_db = VectorDBService()
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        vector_db.add_logs(chunk)
        print(f"Processed {i+len(chunk)}/{len(df)} logs")
```

### Storage Optimization
```bash
# Compress VectorDB (if supported)
# Move to faster storage
sudo mv data/vector_db /fast/ssd/vector_db
ln -s /fast/ssd/vector_db data/vector_db
```

## 🔍 Advanced VectorDB Operations

### 1. Custom Embedding Pipeline
```python
from sentence_transformers import SentenceTransformer

# Load custom embedding model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Generate embeddings manually
texts = ["log message 1", "log message 2"]
embeddings = model.encode(texts)

# Add to VectorDB with custom embeddings
vector_db.add_documents(
    documents=texts,
    embeddings=embeddings.tolist(),
    metadata=[{'custom': 'data'}]
)
```

### 2. Hybrid Search (Vector + Text)
```python
# Combine vector search with text filtering
def hybrid_search(query, text_filter=None):
    # Vector similarity search
    vector_results = vector_db.search_similar_logs(query, top_k=100)
    
    # Additional text filtering
    if text_filter:
        filtered_results = []
        for result in vector_results:
            if text_filter.lower() in result['document'].lower():
                filtered_results.append(result)
        return filtered_results
    
    return vector_results
```

### 3. Collection Management
```python
# Create multiple collections for different log types
collections = {
    'nova_logs': vector_db.create_collection('nova_logs'),
    'keystone_logs': vector_db.create_collection('keystone_logs'),
    'neutron_logs': vector_db.create_collection('neutron_logs')
}

# Query specific collections
nova_results = vector_db.search_similar_logs(
    query="compute error",
    collection_name="nova_logs"
)
```

## 📈 VectorDB Analytics

### Collection Statistics
```python
from services.vector_db_service import VectorDBService

vector_db = VectorDBService()
stats = vector_db.get_collection_stats()

print(f"Total documents: {stats['total_documents']}")
print(f"Collection name: {stats['collection_name']}")
print(f"Embedding dimensions: {stats['embedding_dimensions']}")
```

### Similarity Distribution Analysis
```python
# Analyze similarity score distributions
def analyze_similarities(query, top_k=100):
    results = vector_db.search_similar_logs(query, top_k=top_k)
    similarities = [r['similarity'] for r in results]
    
    print(f"Min similarity: {min(similarities):.3f}")
    print(f"Max similarity: {max(similarities):.3f}")
    print(f"Mean similarity: {sum(similarities)/len(similarities):.3f}")
    
    # Plot distribution
    import matplotlib.pyplot as plt
    plt.hist(similarities, bins=20)
    plt.title(f"Similarity Distribution for: {query}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.show()
```

## 🏷️ VectorDB Best Practices

### 1. Data Ingestion
- **Clean data**: Remove noise and irrelevant information
- **Consistent format**: Maintain uniform log message structure
- **Metadata richness**: Include service, level, timestamp metadata
- **Deduplication**: Use primary keys to prevent duplicates

### 2. Query Optimization
- **Specific queries**: Use precise, descriptive search terms
- **Metadata filters**: Narrow search scope with filters
- **Appropriate top_k**: Balance between coverage and performance
- **Threshold tuning**: Adjust similarity threshold based on use case

### 3. Performance Management
- **Regular maintenance**: Periodic collection cleanup
- **Index optimization**: Rebuild indexes for large collections
- **Memory monitoring**: Track embedding cache usage
- **Storage management**: Monitor disk space usage

### 4. Quality Assurance
- **Embedding validation**: Test embedding quality with known queries
- **Result relevance**: Manually validate search result quality
- **Metadata accuracy**: Ensure metadata consistency
- **Performance benchmarking**: Regular performance testing

The VectorDB system is essential for intelligent log analysis and semantic search! 🚀 