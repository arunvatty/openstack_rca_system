# Environment Setup Guide

Complete guide for setting up the OpenStack RCA system development and production environment.

## 🚀 Quick Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd openstack_rca_system
```

### 2. Automated Setup
```bash
# Run the setup script
source setup_env.sh
source venv/bin/activate
```

## 📋 Manual Setup Steps

### 1. Python Environment
```bash
# Create virtual environment
python3.8+ -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Verify Python version
python --version  # Should be 3.8+
```

### 2. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

### 3. Environment Configuration
Create a `.envrc` file with your configuration:

```bash
# .envrc
export ANTHROPIC_API_KEY="sk-ant-api..."
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="ap-south-1"
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket/group6-capstone/"
export MLFLOW_S3_ENDPOINT_URL="https://s3.ap-south-1.amazonaws.com"
```

### 4. Enable Environment Loading
```bash
# Install direnv (for automatic environment loading)
# Ubuntu/Debian
sudo apt install direnv

# macOS
brew install direnv

# Add to shell profile
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc

# Allow environment loading
direnv allow .
```

## 🔑 API Keys & Credentials

### 1. Anthropic API Key
- Visit [Anthropic Console](https://console.anthropic.com/)
- Create API key
- Add to `.envrc`: `export ANTHROPIC_API_KEY="sk-ant-api..."`

### 2. AWS Credentials
- AWS S3 bucket for model storage
- IAM user with S3 read/write permissions

**Required S3 Permissions:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket/*",
                "arn:aws:s3:::your-bucket"
            ]
        }
    ]
}
```

### 3. MLflow Server
- Remote MLflow tracking server
- S3 artifact storage backend
- Optional: Authentication if required

## 🛠️ MLflow Setup

### Option 1: Local MLflow Server (Development)
```bash
# Start local MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Set in .envrc
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

### Option 2: Remote MLflow Server (Production)
```bash
# Set in .envrc
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
```

### Option 3: MLflow with S3 Backend
```bash
# Set in .envrc
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket/artifacts/"
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

## ✅ Verification

### 1. Environment Test
```bash
# Test all components
python3 -c "
import tensorflow as tf
import mlflow
import chromadb
import anthropic
import boto3
import streamlit
print('✅ All dependencies loaded successfully')
print(f'TensorFlow: {tf.__version__}')
print(f'MLflow: {mlflow.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```

### 2. MLflow Connection Test
```bash
# Test MLflow connection
python3 -c "
import mlflow
print('MLflow Tracking URI:', mlflow.get_tracking_uri())
# Should show your configured URI
"
```

### 3. AWS Connection Test
```bash
# Test S3 access
python3 -c "
import boto3
s3 = boto3.client('s3')
buckets = s3.list_buckets()
print('✅ AWS S3 connection successful')
print('Accessible buckets:', len(buckets['Buckets']))
"
```

### 4. Anthropic API Test
```bash
# Test Claude API
python3 -c "
import anthropic
client = anthropic.Anthropic()
print('✅ Anthropic API configured successfully')
"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues
```bash
# For CPU-only installation
pip install tensorflow-cpu

# For GPU support (requires CUDA)
pip install tensorflow-gpu

# Check installation
python -c "import tensorflow as tf; print('GPU Available:', tf.test.is_gpu_available())"
```

#### 2. ChromaDB Issues
```bash
# Install system dependencies
sudo apt-get install build-essential

# Reinstall ChromaDB
pip uninstall chromadb -y
pip install chromadb==0.4.22
```

#### 3. MLflow Connection Errors
```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Test connectivity
curl -f $MLFLOW_TRACKING_URI/health

# Reset MLflow
mlflow server --help
```

#### 4. AWS Credentials Issues
```bash
# Check credentials
aws configure list

# Test S3 access
aws s3 ls s3://your-bucket/

# Verify region
echo $AWS_DEFAULT_REGION
```

#### 5. Environment Loading Issues
```bash
# Manual environment loading
source .envrc

# Check loaded variables
env | grep -E "(ANTHROPIC|AWS|MLFLOW)"

# Reload direnv
direnv reload
```

### Performance Optimization

#### 1. GPU Setup (Optional)
```bash
# Check NVIDIA drivers
nvidia-smi

# Install CUDA toolkit (if needed)
sudo apt install nvidia-cuda-toolkit

# Install cuDNN (if needed)
# Follow NVIDIA cuDNN installation guide
```

#### 2. Memory Optimization
```bash
# Set TensorFlow memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Limit TensorFlow memory
export TF_MEMORY_LIMIT=4096  # 4GB limit
```

#### 3. CPU Optimization
```bash
# Set CPU thread limits
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
```

## 🔧 Development Setup

### 1. IDE Configuration
**VS Code Extensions:**
- Python
- Pylance
- autoDocstring
- GitLens
- Jupyter

**PyCharm Setup:**
- Configure interpreter: `venv/bin/python`
- Enable virtual environment
- Set project root

### 2. Git Configuration
```bash
# Set up git hooks (optional)
git config --local core.hooksPath .githooks

# Configure user
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. Jupyter Setup (Optional)
```bash
# Install Jupyter in virtual environment
pip install jupyter

# Create kernel
python -m ipykernel install --user --name=openstack_rca --display-name="OpenStack RCA"

# Start Jupyter
jupyter notebook
```

## 📊 Environment Validation Script

Create `validate_environment.py`:

```python
#!/usr/bin/env python3
"""Environment validation script"""

import sys
import os

def validate_environment():
    """Validate all environment components"""
    checks = []
    
    # Check Python version
    if sys.version_info >= (3, 8):
        checks.append("✅ Python version: " + sys.version.split()[0])
    else:
        checks.append("❌ Python version too old: " + sys.version.split()[0])
    
    # Check required packages
    packages = ['tensorflow', 'mlflow', 'streamlit', 'chromadb', 'anthropic', 'boto3']
    for package in packages:
        try:
            __import__(package)
            checks.append(f"✅ {package}: installed")
        except ImportError:
            checks.append(f"❌ {package}: missing")
    
    # Check environment variables
    env_vars = ['ANTHROPIC_API_KEY', 'MLFLOW_TRACKING_URI', 'AWS_ACCESS_KEY_ID']
    for var in env_vars:
        if os.getenv(var):
            checks.append(f"✅ {var}: configured")
        else:
            checks.append(f"❌ {var}: missing")
    
    # Print results
    for check in checks:
        print(check)
    
    # Summary
    passed = sum(1 for check in checks if check.startswith("✅"))
    total = len(checks)
    print(f"\nValidation: {passed}/{total} checks passed")
    
    return passed == total

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
```

Run validation:
```bash
python validate_environment.py
```

## 🏷️ Version Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Python | 3.8 | 3.10+ | For latest TensorFlow support |
| TensorFlow | 2.14.0 | 2.15+ | GPU support optional |
| MLflow | 2.9.0 | 2.10+ | For S3 artifact storage |
| Streamlit | 1.29.0 | 1.30+ | For dashboard features |
| ChromaDB | 0.4.22 | 0.4.22 | Specific version required |
| Node.js | - | 16+ | Optional for web features |

Your environment is now ready for OpenStack RCA development! 🚀 