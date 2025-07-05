# OpenStack Root Cause Analysis (RCA) System

An intelligent log analysis system that automatically identifies and analyzes issues in OpenStack cloud infrastructure using machine learning, vector databases, and AI-powered analysis.

## Features

- **Multi-Modal Analysis**: Combines LSTM/MLP models, vector similarity search, and Claude AI
- **Smart Architecture**: Automatically switches between LSTM (sequential) and MLP (tabular) based on data type
- **Vector Database**: ChromaDB with semantic similarity search across historical logs
- **Real-time Processing**: CLI and web interfaces for immediate analysis
- **High Accuracy**: 92.98% validation accuracy on OpenStack logs

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file:
```env
ANTHROPIC_API_KEY=your_claude_api_key_here
DATA_DIR=logs
MODELS_DIR=saved_models
```

### Training
```bash
export CUDA_VISIBLE_DEVICES="" && python3 main.py --mode train --clean-vector-db
```

### Analysis
```bash
export CUDA_VISIBLE_DEVICES="" && python3 main.py --mode analyze --issue "Instance launch failures"
```

### Web Interface
```bash
export CUDA_VISIBLE_DEVICES="" && python3 main.py --mode streamlit
```

## Architecture

```
OpenStack Logs → Feature Engineering → LSTM/MLP Classification → Vector DB Storage
                                                      ↓
Issue Query → Multi-Modal Filtering → Claude AI Analysis → RCA Report
```

### Core Components

- **LSTM/MLP Classifier**: Dual architecture for sequential/tabular data
- **Vector DB Service**: ChromaDB with sentence transformers
- **RCA Analyzer**: Multi-modal filtering + Claude AI analysis
- **Feature Engineering**: 91 engineered features from raw logs

## Usage

### CLI Commands

```bash
# Train model
python3 main.py --mode train --clean-vector-db

# Analyze specific issue
python3 main.py --mode analyze --issue "Resource shortage errors"

# Launch web interface
python3 main.py --mode streamlit

# Vector DB management
python3 main.py --mode vector-db --vector-db-action status
```

### Issue Categories

- Resource shortages (CPU, memory, disk)
- Network connectivity issues
- Authentication/authorization problems
- Instance lifecycle failures
- Service availability issues

## Project Structure

```
├── models/                 # ML models (LSTM, RCA analyzer)
├── services/              # Vector DB and external services
├── data/                  # Data processing and ingestion
├── utils/                 # Feature engineering and utilities
├── config/                # Configuration settings
├── streamlit_app/         # Web interface
├── logs/                  # Sample log files
├── saved_models/          # Trained model storage
└── main.py               # Main entry point
```

## Performance

- **Model Accuracy**: 92.98% validation accuracy
- **Vector DB**: 870+ documents with semantic search
- **Processing**: Real-time log analysis
- **Scalability**: Handles large log volumes

## Technical Stack

- Python 3.10+, TensorFlow 2.19.0, ChromaDB
- Sentence Transformers, Anthropic Claude API
- Streamlit, Pandas, NumPy, Scikit-learn

## Use Cases

- Proactive monitoring and issue detection
- Incident response and root cause analysis
- Capacity planning and performance optimization
- Compliance and security event tracking 