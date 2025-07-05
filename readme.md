# OpenStack RCA System 🔍

An intelligent Root Cause Analysis system for OpenStack environments using LSTM neural networks and Claude AI integration.

## 🚀 Features

- **🤖 LSTM-based Log Analysis**: Deep learning model for pattern recognition in OpenStack logs
- **🧠 Claude AI Integration**: Advanced natural language analysis for detailed RCA reports
- **📊 Interactive Dashboard**: Streamlit-based web interface for easy log analysis
- **⚡ Real-time Processing**: Instant analysis of log files and issue identification
- **🎯 Multi-component Support**: Analyzes nova-compute, nova-scheduler, nova-api, and other services
- **📈 Timeline Analysis**: Tracks event sequences and identifies failure patterns
- **🔍 Intelligent Filtering**: Two-stage filtering (LSTM + Cosine Similarity) for relevant logs
- **📋 Pattern Recognition**: Extracts error patterns, service distributions, and resource issues

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Log Files     │───▶│  LSTM Model     │───▶│  Claude AI      │
│                 │    │                 │    │                 │
│ • nova-api.log  │    │ • Pattern       │    │ • Natural       │
│ • nova-compute  │    │   Recognition   │    │   Language      │
│ • nova-scheduler│    │ • Importance    │    │   Analysis      │
└─────────────────┘    │   Scoring       │    │ • RCA Reports   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Feature        │    │  Web Interface  │
                       │  Engineering    │    │                 │
                       │                 │    │ • Streamlit     │
                       │ • Timestamps    │    │ • Interactive   │
                       │ • Service Types │    │ • Visualizations│
                       │ • Error Patterns│    │ • Chat Interface│
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+ (Python 3.12 recommended)
- Anthropic API key for Claude integration

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/openstack-rca-system.git
   cd openstack-rca-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=your_claude_api_key_here" > .env
   ```

5. **Setup the system**
   ```bash
   python main.py --mode setup
   ```

## 🔄 Complete Workflow

### Phase 1: System Setup
```bash
# Initialize project structure and prepare log files
python main.py --mode setup
```

### Phase 2: Model Training
```bash
# Train LSTM model on OpenStack logs
python main.py --mode train --logs logs/
```

**Training Pipeline:**
1. **Log Ingestion** → Reads and parses OpenStack log files
2. **Feature Engineering** → Extracts temporal, text, and sequence features
3. **Data Preprocessing** → Prepares sequences for LSTM training
4. **LSTM Training** → Trains neural network on log patterns
5. **Model Persistence** → Saves trained model to `saved_models/`

### Phase 3: Root Cause Analysis
```bash
# Analyze specific issue
python main.py --mode analyze --issue "Instance launch failures" --logs logs/
```

**Analysis Pipeline:**
1. **Issue Categorization** → Identifies issue type (resource_shortage, network_issues, etc.)
2. **LSTM Filtering** → Identifies important logs using trained model
3. **Cosine Similarity** → Finds semantically relevant logs
4. **Pattern Analysis** → Extracts error patterns and timeline
5. **Claude AI Analysis** → Generates detailed RCA report
6. **Recommendations** → Provides actionable solutions

### Phase 4: Web Interface
```bash
# Launch interactive dashboard
python main.py --mode streamlit
```

## 🧠 How It Works

### Two-Stage Intelligent Filtering

#### Stage 1: LSTM Importance Filtering
```python
# LSTM model predicts importance scores (0.0 to 1.0)
importance_scores = self.lstm_model.predict(X)  # [0.9, 0.1, 0.8, ...]

# Filter top 70% important logs
threshold = np.percentile(importance_scores, 30)
important_logs = logs_df[importance_scores >= threshold]
```

#### Stage 2: Cosine Similarity Filtering
```python
# Vectorize issue description and log messages
all_texts = [issue_description] + messages
tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

# Calculate semantic similarity
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
# [0.8, 0.2, 0.1, 0.7, ...]

# Filter and sort by relevance
final_logs = important_logs[similarities >= 0.1]
final_logs = final_logs.sort_values('similarity_score', ascending=False).head(50)
```

### Context Building for LLM
```python
context = f"""
## Dataset Overview:
- Total relevant log entries: {len(logs_df)}
- Issue category: {issue_category}

## Timeline of Critical Events:
- {timestamp}: {event_type} ({service})
  Message: {message}...

## Critical Error Patterns:
- '{error}': {count} occurrences

## Service Activity Analysis:
- {service}: {count} entries ({percentage}% of total)

## Critical Log Entries (Full Context):
### ERROR Level Entries:
- [{timestamp}] {service}: {message}
- [{timestamp}] {service}: {message}
...
"""
```

### Claude AI Analysis
```python
prompt = f"""
You are an expert OpenStack systems administrator performing root cause analysis.

ISSUE: {issue_description}
CATEGORY: {issue_category}
RELEVANT LOGS: {len(logs_df)} entries analyzed

LOG ANALYSIS CONTEXT:
{context}

TASK: Provide detailed technical root cause analysis based on actual log patterns.

REQUIREMENTS:
1. Identify specific technical root cause from log evidence
2. Cite specific log entries or patterns mentioned in context
3. Provide actionable technical solutions
4. Use OpenStack terminology
5. Be specific about failure sequence shown in timeline
"""
```

## 🎯 Supported Use Cases

### 1. Instance Creation Failures
```
Issue: "Instance launch failures"
Symptoms: "No valid host found" errors
RCA: Resource exhaustion (disk/memory), network issues
Solutions: Resource cleanup, host maintenance
```

### 2. API Service Unavailable
```
Issue: "API service unavailable"
Symptoms: HTTP 500 errors, dashboard unresponsive
RCA: Database connection issues, connection pool exhaustion
Solutions: Database maintenance, connection pool tuning
```

### 3. Network Configuration Failures
```
Issue: "Network connectivity issues"
Symptoms: VM network connectivity problems
RCA: VIF plugging timeouts, neutron service problems
Solutions: Network service restart, configuration fixes
```

### 4. Resource Allocation Failures
```
Issue: "Insufficient resources"
Symptoms: "Insufficient resources" errors
RCA: Memory pressure, overcommitment issues
Solutions: Resource monitoring, capacity planning
```

## 📁 Project Structure

```
openstack_rca_system/
├── data/                          # Data ingestion and processing
│   ├── log_ingestion.py          # Log file ingestion manager
│   ├── preprocessing.py          # Data preprocessing for ML models
│   └── __init__.py
├── models/                        # ML models and analyzers
│   ├── lstm_classifier.py        # LSTM neural network
│   ├── rca_analyzer.py           # Root cause analysis engine
│   └── __init__.py
├── utils/                         # Utility functions
│   ├── feature_engineering.py    # Feature extraction and engineering
│   ├── log_parser.py             # Log parsing utilities
│   └── __init__.py
├── streamlit_app/                 # Web interface
│   ├── components.py             # UI components and visualizations
│   ├── chatbot.py                # Chat-based interaction system
│   └── __init__.py
├── config/                        # Configuration settings
│   ├── config.py                 # System configuration
│   └── __init__.py
├── saved_models/                  # Trained model storage
├── logs/                          # Log files directory
│   └── sample_logs/
├── main.py                       # Main entry point
├── test_rca.py                   # Diagnostic test script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
ANTHROPIC_API_KEY=your_claude_api_key_here
DATA_DIR=logs
MODELS_DIR=saved_models
```

### LSTM Model Configuration

Modify `config/config.py` to adjust model parameters:

```python
LSTM_CONFIG = {
    'max_sequence_length': 100,
    'embedding_dim': 128,
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2
}
```

### TF-IDF Configuration

```python
tfidf_vectorizer = TfidfVectorizer(
    max_features=500,        # Top 500 most frequent terms
    stop_words='english',    # Remove common words
    ngram_range=(1, 2)       # Use 1-gram and 2-gram features
)
```

## 📊 Example RCA Output

### Input:
```
Issue: "Instance launch failures"
Logs: 1000 OpenStack log entries
```

### Processing:
1. **LSTM Filtering**: 300 important logs identified
2. **Similarity Filtering**: 47 most relevant logs selected
3. **Pattern Analysis**: Error patterns and timeline extracted
4. **Claude Analysis**: Detailed technical RCA generated

### Output:
```
==================================================
ROOT CAUSE ANALYSIS RESULTS
==================================================
Issue: Instance launch failures
Category: resource_shortage
Relevant Logs: 47

ROOT CAUSE ANALYSIS:
The instance launch failures are caused by resource exhaustion 
on all available compute hosts. Analysis of the logs reveals:

1. Memory Pressure: All hosts are operating at 95%+ memory usage
2. Disk Space: Available disk space below 5% on compute nodes
3. Resource Claims: 8192MB requested, only 6172MB available

The issue stems from overcommitment ratios exceeding configured limits.

RECOMMENDATIONS:
1. Clean up unused instances to free memory
2. Increase disk space on compute hosts
3. Adjust overcommitment ratios in nova.conf
4. Implement resource monitoring alerts
```

## 🚀 Usage Examples

### Training the Model

Train the LSTM model on your OpenStack logs:

```bash
python main.py --mode train --logs path/to/your/logs
```

### Running the Web Interface

Launch the interactive dashboard:

```bash
python main.py --mode streamlit
```

Then visit `http://localhost:8501` in your browser.

### Command Line Analysis

Analyze specific issues directly:

```bash
python main.py --mode analyze --issue "Instance launch failures" --logs path/to/logs
```

### Testing the System

Run the diagnostic test:

```bash
python test_rca.py
```

## 📊 Supported Log Formats

- **OpenStack Log Format**: Standard OpenStack service logs
- **Nova Services**: nova-api, nova-compute, nova-scheduler, nova-conductor
- **Log Levels**: INFO, WARNING, ERROR, DEBUG
- **File Formats**: .log, .txt

## 🎯 Common RCA Questions

1. **"Why did my instance creation fail?"**
   - Answer: Resource exhaustion (disk/memory) or network issues

2. **"Why is the OpenStack dashboard slow/unresponsive?"**
   - Answer: Database connectivity problems and connection pool exhaustion

3. **"Why can't I connect to my VM?"**
   - Answer: Network VIF plugging failures and neutron service issues

4. **"Why do I get 'No valid host' errors?"**
   - Answer: All compute hosts lack sufficient resources (RAM/disk)

5. **"Why are OpenStack services intermittently failing?"**
   - Answer: Database connection issues causing cascade failures across services

## 🐛 Troubleshooting

### Common Issues

**"No module named 'data'"**
- Ensure you're running from the project root directory

**"ANTHROPIC_API_KEY not found"**
- Check your `.env` file exists and contains the API key
- Install python-dotenv: `pip install python-dotenv`

**"Generic RCA responses"**
- Verify your API key is valid
- Check network connectivity to Anthropic API
- Ensure log files contain relevant OpenStack patterns

**"LSTM model not found"**
- Train the model first: `python main.py --mode train --logs logs/`
- Check if model file exists in `saved_models/` directory

### Getting Help

1. Run the diagnostic test: `python test_rca.py`
2. Check the logs for detailed error messages
3. Open an issue on GitHub with your error details

## 🔑 Getting Anthropic API Key

1. **Visit Anthropic Console**: https://console.anthropic.com/
2. **Create Account**: Sign up and verify your email
3. **Generate API Key**: Navigate to API Keys section
4. **Copy Key**: Copy the key (starts with `sk-ant-api03-...`)
5. **Set Environment**: Add to `.env` file or set environment variable

### Pricing Information
- **Claude 3.5 Sonnet**: $3 per million input tokens, $15 per million output tokens
- **Claude 3.5 Haiku**: $0.25 per million input tokens, $1.25 per million output tokens
- **Typical Analysis Cost**: ~$0.01-0.10 per analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LogHub**: For providing the OpenStack log datasets
- **Anthropic**: For Claude AI API
- **OpenStack Community**: For comprehensive documentation and log format standards
- **Streamlit**: For the excellent web framework

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/openstack-rca-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/openstack-rca-system/discussions)

---

## 🎯 Key Benefits

### 1. Rapid Issue Resolution
- **Traditional**: Hours/days of manual log analysis
- **RCA System**: Minutes of automated analysis

### 2. Intelligent Filtering
- **LSTM**: Identifies important vs. unimportant logs
- **Cosine Similarity**: Finds semantically relevant logs
- **Result**: Focus on most relevant information

### 3. Expert-Level Analysis
- **Claude AI**: Provides detailed technical RCA
- **Context-Aware**: Uses actual log evidence
- **Actionable**: Specific recommendations provided

### 4. User-Friendly Interface
- **Streamlit**: Interactive web dashboard
- **Natural Language**: Chat-based queries
- **Visualizations**: Timeline and pattern displays

### 5. Scalable Architecture
- **Modular Design**: Easy to extend and maintain
- **Fallback Mechanisms**: Continues working if components fail
- **Configurable**: Adaptable to different environments

**⭐ Star this repo if you find it helpful!**