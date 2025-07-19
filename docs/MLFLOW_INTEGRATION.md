# MLflow Integration Manual

Complete guide for MLflow model versioning, S3 storage, experiment tracking, and model management in the OpenStack RCA system.

## 🚀 Overview

MLflow provides comprehensive ML lifecycle management with automatic model versioning, S3 artifact storage, experiment tracking, and seamless integration between CLI and Streamlit interfaces.

### Key Features
- **Automatic model upload** to S3 with meaningful folder names
- **Version management** with incremental versioning
- **Experiment tracking** with parameters, metrics, and artifacts
- **S3 integration** for scalable model storage
- **Synchronized access** between CLI and Streamlit
- **Latest model detection** from S3 folders

## 🏗️ MLflow Architecture

```
MLflow Server
├── Experiments
│   └── openstack_rca_system_staging
│       ├── Run 1 (Training)
│       ├── Run 2 (Training) 
│       └── Run 3 (RCA Inference)
└── Model Registry
    └── lstm_model
        ├── Version 1 → S3 Path
        ├── Version 2 → S3 Path
        └── Version N → S3 Path

S3 Bucket Structure:
s3://bucket/group6-capstone/
├── openstack-rca-system-staging_0001/
│   └── models/lstm_model_v1.keras
├── openstack-rca-system-staging_0002/
│   └── models/lstm_model_v2.keras
└── ...
```

## 🔧 MLflow Setup

### 1. Environment Configuration
```bash
# .envrc
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket/group6-capstone/"
export MLFLOW_S3_ENDPOINT_URL="https://s3.ap-south-1.amazonaws.com"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="ap-south-1"
```

### 2. MLflow Manager Configuration (config/config.py)
```python
MLFLOW_CONFIG = {
    'experiment_name': 'openstack_rca_system_staging',
    'artifact_root': 's3://your-bucket/group6-capstone/',
    'enable_auto_versioning': True,
    'model_name': 'lstm_model',
    'enable_s3_upload': True,
    'meaningful_folder_names': True,
    'cleanup_temp_files': True
}
```

### 3. Verify MLflow Connection
```bash
# Test MLflow server connection
python3 -c "
import mlflow
print('MLflow URI:', mlflow.get_tracking_uri())
print('MLflow Version:', mlflow.__version__)
"

# Test S3 connection
python3 -c "
import boto3
s3 = boto3.client('s3')
buckets = s3.list_buckets()
print('S3 connection successful')
"
```

## 🎯 Model Training with MLflow

### 1. Training Pipeline
```bash
# Train model (automatically uploads to MLflow/S3)
python3 main.py --mode train

# Output:
# ✅ MLflow enabled for model training
# 🎯 Using auto-configured experiment: openstack_rca_system_staging
# ✅ Model trained successfully
# 📊 Model v15 logged to MLflow and S3 successfully
# ✅ Uploaded: s3://bucket/group6-capstone/openstack-rca-system-staging_0015/models/lstm_model_v15.keras
```

### 2. Training Process Details
```python
# What happens during training:
1. Initialize MLflow manager
2. Create/use experiment: 'openstack_rca_system_staging'
3. Start MLflow run for training
4. Train LSTM model
5. Save model locally: models/lstm_log_classifier.keras
6. Upload to S3: meaningful folder name (openstack-rca-system-staging_vXX)
7. Register in MLflow model registry
8. Log metrics and parameters
9. End MLflow run
```

### 3. Logged Information
- **Parameters**: epochs, batch_size, lstm_units, dropout_rate
- **Metrics**: accuracy, precision, recall, loss
- **Model**: keras model file
- **Artifacts**: training plots, model summary
- **S3 Path**: meaningful folder structure

## 🔍 Model Loading and Inference

### 1. RCA Analysis with MLflow
```bash
# RCA analysis (loads latest model from S3)
python3 main.py --mode analyze --issue "Database connection timeout"

# Output:
# 🔍 Attempting to load model from MLflow/S3...
# 📦 Found latest model folder: openstack-rca-system-staging_0015 (version 15)
# ⬇️ Downloading model from meaningful folder: ...staging_0015/models/lstm_model_v15.keras
# ✅ Successfully loaded LSTM model from MLflow/S3
```

### 2. Model Loading Process
```python
# What happens during model loading:
1. Initialize MLflow manager
2. Search S3 for latest version folder
3. Find highest version number
4. Download model.keras to temporary file
5. Load model into TensorFlow
6. Clean up temporary file
7. Use model for RCA analysis
```

### 3. Streamlit Integration
```bash
# Streamlit dashboard (uses same S3 model)
streamlit run streamlit_app/chatbot.py

# Streamlit automatically:
# - Loads latest model from S3
# - Shows model source in sidebar
# - Provides refresh button for latest model
# - Synchronizes with CLI application
```

## 📊 MLflow Experiment Tracking

### 1. View Experiments
```bash
# Access MLflow UI
# Visit: https://your-mlflow-server.com
# Navigate to: Experiments → openstack_rca_system_staging
```

### 2. Experiment Information
- **Training Runs**: Each training session creates a new run
- **Parameters**: Model hyperparameters and configuration
- **Metrics**: Training accuracy, loss, validation metrics
- **Artifacts**: Model files, plots, logs
- **Duration**: Training time and resource usage

### 3. Model Registry
- **Model Versions**: Incremental versioning (v1, v2, v3...)
- **S3 Locations**: Direct links to S3 model files
- **Staging**: Development/Production model lifecycle
- **Descriptions**: Model performance and notes

## 🗄️ S3 Model Storage

### 1. Folder Structure
```
s3://your-bucket/group6-capstone/
├── openstack-rca-system-staging_0001/
│   └── models/
│       └── lstm_model_v1.keras         # Clean, single file
├── openstack-rca-system-staging_0002/
│   └── models/
│       └── lstm_model_v2.keras
└── openstack-rca-system-staging_vXXX/
    └── models/
        └── lstm_model_vXXX.keras
```

### 2. Benefits of Meaningful Folder Names
- **Human-readable**: Easy to identify model versions
- **Chronological**: Natural sorting by version number
- **Clean storage**: No random hash folders
- **Direct access**: Can access models directly from S3

### 3. Model File Management
- **Single file upload**: Only `.keras` model file
- **No duplicates**: Eliminated duplicate MLflow artifacts
- **Automatic cleanup**: Temporary files removed after upload
- **Version tracking**: Clear version progression

## 🔄 Model Synchronization

### 1. CLI ↔ Streamlit Sync
Both interfaces use identical models:

```python
# CLI Application
model = mlflow_manager.load_model_with_versioning(
    model_name="lstm_model",
    version="latest"
)

# Streamlit Application  
model = mlflow_manager.load_model_with_versioning(
    model_name="lstm_model", 
    version="latest"
)

# Both load the same S3 model file!
```

### 2. Automatic Version Detection
```python
# System automatically finds latest version
def find_latest_model():
    # Search S3 for folders matching pattern
    # openstack-rca-system-staging_XXXX
    # Extract version numbers
    # Return highest version number
    return latest_version_folder
```

### 3. Model Update Workflow
```bash
1. Train new model → Uploads as version N+1
2. CLI analysis → Automatically uses version N+1
3. Streamlit refresh → Downloads version N+1
4. Both interfaces synchronized
```

## 🛠️ MLflow Manager Operations

### 1. Core Functions

```python
from mlflow_integration.mlflow_manager import MLflowManager

# Initialize
mlflow_manager = MLflowManager(
    tracking_uri="https://mlflow-server.com",
    experiment_name="openstack_rca_system_staging",
    enable_mlflow=True
)

# Train and log model
mlflow_manager.log_model(
    model=lstm_model,
    artifact_path="models",
    model_name="lstm_model",
    metadata={
        'accuracy': 0.95,
        'version': 'v15'
    }
)

# Load latest model
model = mlflow_manager.load_model_with_versioning(
    model_name="lstm_model",
    version="latest"
)
```

### 2. Advanced Operations
```python
# Log parameters and metrics
mlflow_manager.log_params({
    'epochs': 50,
    'batch_size': 32,
    'lstm_units': 64
})

mlflow_manager.log_metrics({
    'accuracy': 0.95,
    'loss': 0.05,
    'precision': 0.92
})

# Start inference run
mlflow_manager.start_run(
    run_name=f"rca_inference_{timestamp}",
    experiment_name="openstack_rca_system_staging"
)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. MLflow Connection Errors
```bash
# Check connection
curl -f $MLFLOW_TRACKING_URI/health

# Test from Python
python3 -c "
import mlflow
print('URI:', mlflow.get_tracking_uri())
client = mlflow.tracking.MlflowClient()
print('Connection successful')
"
```

#### 2. S3 Upload Issues
```bash
# Check S3 credentials
aws s3 ls s3://your-bucket/

# Test S3 from Python
python3 -c "
import boto3
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket='your-bucket', MaxKeys=5)
print('S3 connection successful')
"
```

#### 3. Model Loading Issues
```bash
# Check if models exist in S3
aws s3 ls s3://your-bucket/group6-capstone/ --recursive

# Verify MLflow registry
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()
models = client.list_registered_models()
for model in models:
    print(f'Model: {model.name}')
    versions = client.list_model_versions(model.name)
    for v in versions:
        print(f'  Version {v.version}: {v.source}')
"
```

#### 4. Experiment Issues
```bash
# List experiments
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.list_experiments()
for exp in experiments:
    print(f'ID: {exp.experiment_id}, Name: {exp.name}')
"

# Create experiment manually if needed
python3 -c "
import mlflow
mlflow.create_experiment(
    'openstack_rca_system_staging',
    artifact_location='s3://your-bucket/group6-capstone/'
)
"
```

## ⚡ Performance Optimization

### 1. S3 Upload Optimization
```python
# Use multipart upload for large models
MLFLOW_CONFIG = {
    's3_multipart_threshold': 64 * 1024 * 1024,  # 64MB
    's3_multipart_chunksize': 16 * 1024 * 1024,  # 16MB
    'enable_compression': True
}
```

### 2. Model Loading Optimization
```python
# Cache models locally for repeated use
import tempfile
import os

class ModelCache:
    def __init__(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cached_models = {}
    
    def get_model(self, version):
        if version not in self.cached_models:
            model = mlflow_manager.load_model_with_versioning(
                model_name="lstm_model", 
                version=version
            )
            self.cached_models[version] = model
        return self.cached_models[version]
```

### 3. Experiment Cleanup
```bash
# Clean old experiments (if needed)
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()

# Delete old runs (keep last 10)
experiment = client.get_experiment_by_name('openstack_rca_system_staging')
runs = client.list_run_infos(experiment.experiment_id)
old_runs = sorted(runs, key=lambda r: r.start_time)[:-10]

for run in old_runs:
    client.delete_run(run.run_id)
    print(f'Deleted run: {run.run_id}')
"
```

## 📈 MLflow Monitoring

### 1. Model Performance Tracking
```python
# Log model performance metrics
def log_model_performance(model, test_data):
    metrics = evaluate_model(model, test_data)
    
    mlflow_manager.log_metrics({
        'test_accuracy': metrics['accuracy'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'inference_time': metrics['inference_time']
    })
```

### 2. Resource Usage Monitoring
```python
import psutil
import time

# Monitor training resources
def log_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    
    mlflow_manager.log_metrics({
        'cpu_usage_percent': cpu_percent,
        'memory_usage_percent': memory_usage,
        'disk_usage_percent': disk_usage
    })
```

### 3. Model Drift Detection
```python
# Compare model versions
def compare_models(current_version, previous_version):
    current_metrics = get_model_metrics(current_version)
    previous_metrics = get_model_metrics(previous_version)
    
    accuracy_drift = abs(current_metrics['accuracy'] - previous_metrics['accuracy'])
    
    if accuracy_drift > 0.05:  # 5% threshold
        logger.warning(f"Model drift detected: {accuracy_drift:.3f}")
        
    mlflow_manager.log_metrics({
        'accuracy_drift': accuracy_drift,
        'performance_delta': current_metrics['accuracy'] - previous_metrics['accuracy']
    })
```

## 🏷️ MLflow Best Practices

### 1. Training
- **Consistent naming**: Use standardized experiment names
- **Comprehensive logging**: Log all relevant parameters and metrics
- **Model validation**: Validate models before uploading
- **Version descriptions**: Add meaningful version descriptions

### 2. Model Management
- **Regular cleanup**: Remove old, unused models
- **Performance monitoring**: Track model performance over time
- **A/B testing**: Compare model versions systematically
- **Rollback capability**: Maintain ability to revert to previous versions

### 3. Infrastructure
- **Backup strategy**: Regular MLflow database backups
- **S3 lifecycle**: Configure S3 lifecycle policies
- **Access control**: Implement proper authentication and authorization
- **Monitoring**: Monitor MLflow server health and S3 costs

### 4. Development Workflow
- **Feature branches**: Use separate experiments for feature development
- **CI/CD integration**: Automate model testing and deployment
- **Documentation**: Document model changes and improvements
- **Collaboration**: Use MLflow for team collaboration and model sharing

The MLflow integration provides robust ML lifecycle management for the OpenStack RCA system! 🚀 