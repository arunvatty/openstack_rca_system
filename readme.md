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

## Issues and RCA samples

# User-Facing OpenStack Errors Requiring RCA

## 1. **Instance Creation Failures**

### What Users See:
```
ERROR: No valid host was found. There are not enough hosts available.
Status: ERROR
Fault: NoValidHost
```

### User Experience:
- Instance creation request accepted (HTTP 202)
- Instance stuck in "BUILD" status
- Eventually transitions to "ERROR" state
- Users cannot access their VM

### RCA from Logs:
- **Disk Space**: Hosts have insufficient disk (cp-1: 2GB available, needs 20GB)
- **Memory**: Insufficient RAM (needs 8192MB, only 6172MB available)
- **Resource Exhaustion**: 95% memory usage, overcommitment ratio exceeded

---

## 2. **API Service Unavailable**

### What Users See:
```
HTTP 500 Internal Server Error
"The server encountered an unexpected condition"
```

### User Experience:
- Dashboard becomes unresponsive
- API calls fail intermittently
- Cannot perform any OpenStack operations

### RCA from Logs:
- **Database Connection Issues**: MySQL server connection timeout
- **Connection Pool Exhausted**: Database connection pool depleted
- **Service Cascade Failure**: nova-api, nova-scheduler, nova-compute all affected

---

## 3. **Network Configuration Failures**

### What Users See:
```
Instance Status: ERROR
"Network setup failed during instance creation"
```

### User Experience:
- VM appears to start but becomes inaccessible
- No network connectivity to instance
- Instance automatically terminated

### RCA from Logs:
- **VIF Plugging Timeout**: Port 4f8a7b6c-5d4e-3f2a-1b0c-9e8d7c6b5a4f failed
- **Neutron Service Down**: Connection refused (111) to neutron
- **Network Agent Failure**: Cannot communicate with neutron agents

---

## 4. **Resource Allocation Failures**

### What Users See:
```
ERROR: Insufficient resources available
"Unable to schedule instance on any compute host"
```

### User Experience:
- Instance creation repeatedly fails
- No clear indication of which resource is constrained
- Users may try different flavors unsuccessfully

### RCA from Logs:
- **Memory Pressure**: System using 95% of physical RAM
- **Overcommitment Issues**: Memory ratio 1.5 exceeded
- **Resource Claim Failures**: 8192MB requested, only 6172MB free

---

## 5. **Service Connectivity Issues**

### What Users See:
```
"Service temporarily unavailable"
HTTP timeouts in dashboard
Intermittent connection failures
```

### User Experience:
- Sporadic access to OpenStack services
- Operations may succeed after multiple retries
- Unpredictable service behavior

### RCA from Logs:
- **RPC Call Failures**: MessagingTimeout to nova-compute
- **Service Heartbeat Lost**: Database connection issues
- **Inter-service Communication**: nova-conductor timeouts

---

## Common RCA Questions Users Ask:

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

## Impact Assessment:

- **Availability**: Multiple service outages affecting user operations
- **Performance**: Degraded response times due to resource constraints  
- **Reliability**: Cascade failures affecting multiple OpenStack components
- **User Experience**: Failed instance deployments and network connectivity issues

**⭐ Star this repo if you find it helpful!**