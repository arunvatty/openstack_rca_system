# OpenStack RCA System 🔍

An intelligent Root Cause Analysis system for OpenStack environments using LSTM neural networks and Claude AI integration.

## 🚀 Features

- **🤖 LSTM-based Log Analysis**: Deep learning model for pattern recognition in OpenStack logs
- **🧠 Claude AI Integration**: Advanced natural language analysis for detailed RCA reports
- **📊 Interactive Dashboard**: Streamlit-based web interface for easy log analysis
- **⚡ Real-time Processing**: Instant analysis of log files and issue identification
- **🎯 Multi-component Support**: Analyzes nova-compute, nova-scheduler, nova-api, and other services
- **📈 Timeline Analysis**: Tracks event sequences and identifies failure patterns

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

## 🚀 Usage

### Training the Model

Train the LSTM model on your OpenStack logs:

```bash
python main.py --mode train --logs path/to/your/logs
```

### Running the Web Interface

Launch the interactive dashboard:

```bash
python main.py --mode web
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

## 📁 Project Structure

```
openstack_rca_system/
├── data/                          # Data ingestion and processing
│   ├── log_ingestion.py          # Log file ingestion manager
│   └── __init__.py
├── models/                        # ML models and analyzers
│   ├── lstm_classifier.py        # LSTM neural network
│   ├── rca_analyzer.py           # Root cause analysis engine
│   └── __init__.py
├── utils/                         # Utility functions
│   ├── feature_engineering.py    # Feature extraction and engineering
│   ├── log_parser.py             # Log parsing utilities
│   └── __init__.py
├── saved_models/                  # Trained model storage
├── logs/                          # Log files directory
│   └── sample_logs/
├── web/                          # Streamlit web interface
│   ├── streamlit_app.py          # Main web application
│   └── components.py             # UI components
├── main.py                       # Main entry point
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── test_rca.py                   # Diagnostic test script
└── README.md                     # This file
```

## 🎯 Use Cases

### Instance Launch Failures
```
"My virtual machine launch is failing repeatedly. Can you analyze what's going wrong?"
```

### Resource Issues
```
"I'm getting 'no valid host found' errors when creating instances."
```

### Service Health
```
"Nova-compute seems to be having communication problems. What's the issue?"
```

### Performance Analysis
```
"What OpenStack operations are happening in my environment?"
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

Modify `config.py` to adjust model parameters:

```python
LSTM_CONFIG = {
    'max_sequence_length': 100,
    'embedding_dim': 128,
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'batch_size': 32,
    'epochs': 50
}
```

## 📊 Supported Log Formats

- **OpenStack Log Format**: Standard OpenStack service logs
- **Nova Services**: nova-api, nova-compute, nova-scheduler, nova-conductor
- **Log Levels**: INFO, WARNING, ERROR, DEBUG
- **File Formats**: .log, .txt

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

### Getting Help

1. Run the diagnostic test: `python test_rca.py`
2. Check the logs for detailed error messages
3. Open an issue on GitHub with your error details

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

**⭐ Star this repo if you find it helpful!**