# MLflow Model Versioning & S3 Integration Guide

## 🎯 Overview

The OpenStack RCA System now includes comprehensive MLflow integration with automatic model versioning, S3 artifact storage, and model registry management. This guide covers all aspects of the versioning system.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Training      │    │   MLflow         │    │   S3 Storage    │
│   Pipeline      │───▶│   Tracking       │───▶│   (Artifacts)   │
│                 │    │   Server         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Model         │    │   Model          │    │   Versioned     │
│   Registry      │    │   Metadata       │    │   Model Files   │
│                 │    │   & Metrics      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                              │
         └──────────────────┐    ┌─────────────────────┘
                            ▼    ▼
                   ┌─────────────────┐
                   │   RCA Analysis  │
                   │   (Inference)   │
                   └─────────────────┘
```

## 🚀 Features

### ✅ Automatic Model Versioning
- **Semantic Versioning**: Automatic version increments (v1, v2, v3...)
- **Run ID Tracking**: Each model version linked to specific MLflow run
- **Metadata Storage**: Complete model lineage and metadata tracking
- **Stage Management**: Staging → Production → Archived lifecycle

### ✅ S3 Integration
- **Artifact Storage**: Models stored in S3 with versioning
- **Backup & Recovery**: Automatic backup of critical models
- **Regional Endpoints**: Support for region-specific S3 endpoints
- **Access Control**: Secure access with AWS credentials

### ✅ Model Registry
- **Centralized Management**: Single source of truth for all models
- **Version Comparison**: Compare different model versions
- **Stage Transitions**: Promote models through lifecycle stages
- **Audit Trail**: Complete history of model changes

### ✅ RCA Integration
- **Automatic Loading**: RCA system automatically loads latest models
- **Fallback Strategy**: Local → MLflow → Training fallback
- **Performance Tracking**: Track inference performance per model version
- **A/B Testing**: Support for testing different model versions

## 📋 Configuration

### Environment Variables
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=s3://your-bucket/mlflow-artifacts
MLFLOW_S3_ENDPOINT_URL=https://s3.region.amazonaws.com

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region

# Optional
MLFLOW_S3_BUCKET=your-bucket
ENVIRONMENT=production
```

### Config File Settings
```python
MLFLOW_CONFIG = {
    # Model Registry & Versioning
    'auto_register_model': True,
    'default_model_stage': 'Staging',
    'auto_promote_threshold': 0.85,
    
    # Model Lifecycle Management
    'max_model_versions': 10,
    'archive_old_versions': True,
    'production_approval_required': False,
    
    # Versioning Strategy
    'versioning_strategy': 'semantic',
    'version_format': 'v{major}.{minor}.{patch}',
    'auto_increment': True,
    
    # S3 Configuration
    's3_model_prefix': 'models/openstack_rca',
    's3_backup_enabled': True,
}
```

## 🔄 Model Lifecycle

### 1. Training with Versioning
```bash
# Train model with automatic MLflow tracking
python main.py --mode train --enable-mlflow

# Output:
# ✅ Model v3 logged to MLflow and S3 successfully
# 🗃️ Model Registry Status:
#    - lstm_model: 3 versions, latest: v3
```

### 2. Model Stages
- **None**: Initial state for new models
- **Staging**: Models ready for testing (default for new models)
- **Production**: Models approved for production use
- **Archived**: Old models no longer in active use

### 3. Automatic Promotion
Models can be automatically promoted based on performance:
```python
# Auto-promote if accuracy > 85%
if accuracy > 0.85:
    promote_to_production(model_version)
```

## 🛠️ Usage Examples

### Training with Versioning
```python
from mlflow_integration.mlflow_manager import MLflowManager

# Initialize MLflow manager
mlflow_manager = MLflowManager(
    tracking_uri="http://localhost:5000",
    experiment_name="openstack_rca_system_auto",
    enable_mlflow=True
)

# Start training run
run_id = mlflow_manager.start_run(
    run_name="lstm_training_v2",
    tags={'model_type': 'lstm', 'purpose': 'production'}
)

# Train model...
# model = train_your_model()

# Log model with versioning
model_version = mlflow_manager.log_model_with_versioning(
    model=model,
    model_name="lstm_model",
    model_type="tensorflow",
    model_stage="Staging"
)

print(f"Model logged as version: {model_version}")
```

### Loading Versioned Models for RCA
```python
# Load latest model version
model_result = mlflow_manager.load_model_with_versioning(
    model_name="lstm_model",
    version="latest",
    model_type="tensorflow"
)

if model_result:
    model = model_result['model']
    metadata = model_result['metadata']
    
    print(f"Loaded model v{metadata['version']} from {metadata['s3_location']}")
    
    # Use model for RCA analysis
    rca_analyzer = RCAAnalyzer(api_key, model)
    results = rca_analyzer.analyze_issue("Instance launch failure", logs_df)
```

### Model Management CLI
```bash
# List all models and versions
python utils/mlflow_model_manager.py list

# Show detailed model information
python utils/mlflow_model_manager.py details --model lstm_model

# Load specific model version
python utils/mlflow_model_manager.py load --model lstm_model --version 3

# Promote model to production
python utils/mlflow_model_manager.py promote --model lstm_model --version 3 --stage Production

# Compare model versions
python utils/mlflow_model_manager.py compare --model lstm_model

# Export model registry to JSON
python utils/mlflow_model_manager.py export --output models.json

# Cleanup old versions (keep latest 5)
python utils/mlflow_model_manager.py cleanup --model lstm_model --keep 5
```

## 🔍 Monitoring & Tracking

### Training Metrics
Each model version tracks:
- **Hyperparameters**: All training configuration
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Training Time**: Duration and resource usage
- **Data Metrics**: Training data size and characteristics
- **System Info**: Environment and dependency versions

### Inference Tracking
Each RCA analysis logs:
- **Model Version**: Which model version was used
- **Performance**: Inference time and resource usage
- **Results**: Analysis quality and confidence scores
- **Data**: Input characteristics and output metrics

### Model Registry Dashboard
Access the MLflow UI to view:
- **Model Versions**: All versions with metadata
- **Performance Comparison**: Compare metrics across versions
- **Deployment Status**: Current stage and deployment info
- **Audit Trail**: Complete history of changes

## 🚨 Best Practices

### 1. Version Management
- **Semantic Versioning**: Use meaningful version numbers
- **Stage Discipline**: Follow proper stage transitions
- **Documentation**: Document significant changes in model versions
- **Testing**: Thoroughly test models before production promotion

### 2. S3 Storage
- **Bucket Organization**: Use consistent prefixes for organization
- **Backup Strategy**: Regular backups of critical model versions
- **Access Control**: Proper IAM policies for S3 access
- **Cost Management**: Lifecycle policies for old artifacts

### 3. Performance Monitoring
- **Baseline Tracking**: Establish performance baselines
- **Drift Detection**: Monitor for model drift over time
- **A/B Testing**: Compare model versions in production
- **Rollback Strategy**: Plan for quick rollbacks if needed

### 4. Security
- **Credential Management**: Secure storage of AWS credentials
- **Network Security**: Secure connections to MLflow and S3
- **Access Logging**: Log all model access and changes
- **Compliance**: Follow data governance requirements

## 🧪 Testing

### Run Comprehensive Tests
```bash
# Test full versioning workflow
python test_mlflow_versioning.py

# Test model management CLI
python utils/mlflow_model_manager.py list
python utils/mlflow_model_manager.py details --model lstm_model
```

### Expected Output
```
🧪 MLFLOW VERSIONING & S3 INTEGRATION TESTS
==================================================
✅ All required environment variables are set

🧪 Running test: Model Versioning
================================================================================
TESTING MLFLOW MODEL VERSIONING & S3 INTEGRATION
================================================================================
✅ MLflow manager initialized
✅ Run started: abc123def456...
✅ Model logged with version: 1
✅ Model loaded successfully with metadata:
   - Version: 1
   - Stage: Staging
   - Run ID: abc123def456...
   - S3 Location: s3://bucket/mlflow-artifacts/1/abc123def456/artifacts

TEST SUMMARY
================================================================================
Model Versioning                        ✅ PASSED
RCA with Versioned Models               ✅ PASSED

Overall: 2/2 tests passed
🎉 All tests passed! MLflow versioning system is working correctly.
```

## 🔧 Troubleshooting

### Common Issues

#### 1. MLflow Connection Failed
```bash
# Check MLflow server status
curl http://localhost:5000/health

# Verify environment variables
echo $MLFLOW_TRACKING_URI
```

#### 2. S3 Access Denied
```bash
# Test AWS credentials
aws s3 ls s3://your-bucket/

# Check S3 endpoint URL
echo $MLFLOW_S3_ENDPOINT_URL
```

#### 3. Model Loading Failed
```python
# Check model registry
python utils/mlflow_model_manager.py list

# Verify model exists
python utils/mlflow_model_manager.py details --model lstm_model
```

#### 4. Version Conflicts
```bash
# List all versions
python utils/mlflow_model_manager.py compare --model lstm_model

# Archive old versions
python utils/mlflow_model_manager.py cleanup --model lstm_model --keep 3
```

### Debug Mode
```bash
# Enable debug logging
export MLFLOW_TRACKING_INSECURE_TLS=true
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run with verbose logging
python main.py --mode train --enable-mlflow --verbose
```

## 📊 Model Registry Schema

### Model Metadata Structure
```json
{
  "model_name": "lstm_model",
  "registered_name": "openstack_rca_system_auto_lstm_model",
  "version": "3",
  "stage": "Production",
  "run_id": "abc123def456789",
  "creation_timestamp": "2024-01-15T10:30:00Z",
  "last_updated_timestamp": "2024-01-15T11:00:00Z",
  "source": "s3://bucket/mlflow-artifacts/1/abc123def456/artifacts/lstm_model",
  "status": "READY",
  "artifact_uri": "s3://bucket/mlflow-artifacts/1/abc123def456/artifacts",
  "s3_location": "s3://bucket/mlflow-artifacts/1/abc123def456/artifacts",
  "model_uri": "models:/openstack_rca_system_auto_lstm_model/3",
  "metrics": {
    "accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.89,
    "f1_score": 0.87
  },
  "parameters": {
    "epochs": 50,
    "batch_size": 32,
    "lstm_units": 64,
    "dropout_rate": 0.2
  }
}
```

## 🎉 Summary

The MLflow versioning system provides:

✅ **Automatic Model Versioning** with run ID tracking  
✅ **S3 Integration** for scalable artifact storage  
✅ **Model Registry** for centralized management  
✅ **RCA Integration** with automatic model loading  
✅ **CLI Tools** for model management  
✅ **Comprehensive Testing** and monitoring  
✅ **Production-Ready** deployment pipeline  

This system ensures that your OpenStack RCA models are properly versioned, stored securely in S3, and can be reliably loaded for both training and inference workflows.

For more information, see:
- `mlflow_integration/mlflow_manager.py` - Core versioning implementation
- `utils/mlflow_model_manager.py` - CLI management tools  
- `test_mlflow_versioning.py` - Comprehensive test suite
- `config/config.py` - Configuration options 