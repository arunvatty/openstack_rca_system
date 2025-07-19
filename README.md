# OpenStack Root Cause Analysis (RCA) System

An intelligent log analysis system that automatically identifies and analyzes issues in OpenStack cloud infrastructure using machine learning, vector databases, and AI-powered analysis with MLflow model management and S3 storage.

## 🚀 Key Features

- **🤖 LSTM-based Log Analysis**: Deep learning model for pattern recognition in OpenStack logs
- **🧠 Claude AI Integration**: Advanced natural language analysis for detailed RCA reports  
- **📊 Interactive Dashboard**: Streamlit-based web interface for easy log analysis
- **⚡ Hybrid RCA Engine**: LSTM importance filtering + Vector DB semantic search + TF-IDF
- **🎯 Dual Analysis Modes**: Full mode (hybrid) and Fast mode (LSTM + TF-IDF only)
- **🔧 MLflow Integration**: Model versioning, experiment tracking, and S3 storage
- **💾 Smart Caching**: Log caching system to avoid repeated file loading
- **🗄️ VectorDB**: ChromaDB with all-MiniLM-L12-v2 for semantic understanding
- **📈 High Performance**: Up to 15x faster than traditional approaches

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenStack     │    │    MLflow       │    │      S3         │
│   Log Sources   │───▶│   Tracking      │───▶│   Model Storage │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Hybrid RCA Analysis Engine                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  LSTM Model     │   VectorDB      │    Claude AI Analysis       │
│  (Importance    │  (Semantic      │   (Natural Language         │
│   Filtering)    │   Search)       │    RCA Reports)             │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            Output Layer (CLI + Streamlit Dashboard)             │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd openstack_rca_system

# Setup environment (see docs/ENVIRONMENT_SETUP.md for details)
source setup_env.sh
source venv/bin/activate
```

### 2. Train LSTM Model
```bash
# Train new model (uploads to MLflow/S3 automatically)
python3 main.py --mode train
```

### 3. Run RCA Analysis
```bash
# CLI Analysis
python3 main.py --mode analyze --issue "Database timeout errors"

# Streamlit Dashboard (uses same S3 model as CLI)
streamlit run streamlit_app/chatbot.py
```

## 📚 Documentation

| Manual | Description | Location |
|--------|-------------|-----------|
| **Environment Setup** | Environment configuration, dependencies, API keys | [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Cache Operations** | Log caching, cache management, performance optimization | [`docs/CACHE_OPERATIONS.md`](docs/CACHE_OPERATIONS.md) |
| **Vector Database** | ChromaDB operations, semantic search, data ingestion | [`docs/VECTOR_DB_OPERATIONS.md`](docs/VECTOR_DB_OPERATIONS.md) |
| **MLflow Integration** | Model versioning, S3 storage, experiment tracking | [`docs/MLFLOW_INTEGRATION.md`](docs/MLFLOW_INTEGRATION.md) |
| **RCA Analysis** | Fast mode vs Hybrid mode, analysis techniques | [`docs/RCA_ANALYSIS.md`](docs/RCA_ANALYSIS.md) |

## 🎯 Analysis Modes

| Mode | Components | Use Case | Performance |
|------|------------|----------|-------------|
| **Fast Mode** | LSTM + TF-IDF | Quick analysis, resource-constrained | ~3-5 seconds |
| **Hybrid Mode** | LSTM + VectorDB + TF-IDF | Comprehensive analysis, best accuracy | ~15-20 seconds |

```bash
# Fast mode
python3 main.py --mode analyze --issue "Your issue" --fast-mode

# Hybrid mode (default)
python3 main.py --mode analyze --issue "Your issue"
```

## 📊 Model Management

- **Training**: Models automatically uploaded to MLflow/S3
- **Storage**: S3 bucket with meaningful folder names (`openstack-rca-system-staging_vXX`)
- **Loading**: CLI and Streamlit both use latest S3 model
- **Versioning**: Automatic version incrementing with experiment tracking
- **Sync**: Both interfaces use identical models from S3

## 🔧 Project Structure

```
openstack_rca_system/
├── main.py                 # CLI entry point
├── streamlit_app/          # Web dashboard
├── lstm/                   # LSTM model and RCA analyzer
├── mlflow_integration/     # MLflow model management
├── services/              # VectorDB and other services
├── data/                  # Log data and cache
├── config/                # Configuration files
├── utils/                 # Utilities and feature engineering
├── docs/                  # Detailed documentation
└── requirements.txt       # Python dependencies
```

## 📈 Performance Metrics

- **Log Processing**: 500-1000 logs/second
- **LSTM Inference**: < 3 seconds for 1000 logs  
- **VectorDB Search**: < 100ms for semantic queries
- **End-to-End RCA**: 15-20 seconds (hybrid), 3-5 seconds (fast)
- **Model Sync**: CLI and Streamlit use identical S3 models

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.14+
- MLflow 2.9+
- ChromaDB 0.4+
- Streamlit 1.29+
- AWS S3 access (for model storage)
- Anthropic API key (for Claude AI)

## 🔍 Common Use Cases

- **Service Outages**: "Nova service not responding"
- **Performance Issues**: "High response times in API calls"  
- **Resource Problems**: "Disk space exhausted on compute nodes"
- **Network Issues**: "Connection timeouts between services"
- **Database Problems**: "Database connection pool exhausted"

## 🚨 Troubleshooting

### Model Loading Issues
```bash
# Check S3 connection
aws s3 ls s3://your-bucket/

# Verify MLflow connection  
python3 -c "import mlflow; print(mlflow.get_tracking_uri())"

# Refresh model in Streamlit
# Use "🔄 Refresh Model" button in sidebar
```

### Performance Issues
```bash
# Clear log cache
python3 utils/cache_manager.py clear --target all

# Use fast mode for quick analysis
python3 main.py --mode analyze --issue "issue" --fast-mode
```

## 📞 Support

For detailed documentation, see the `docs/` directory. Each manual provides comprehensive guidance for specific operations and troubleshooting.

## 🏷️ Version

**Current**: v1.0 with MLflow integration and S3 model storage
**Latest Model**: Automatically detected from S3 (version-based)
**Compatibility**: CLI and Streamlit synchronized via shared S3 models 