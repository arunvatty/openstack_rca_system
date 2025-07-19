#!/bin/bash

# OpenStack RCA System - Environment Setup Script
# Use this script if you need to manually load environment variables

echo "🔧 Loading OpenStack RCA System Environment..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found in current directory"
    echo "   Please make sure you're in the project root directory"
    exit 1
fi

# Load environment variables
set -a  # automatically export all variables
source .env
set +a  # stop automatically exporting

echo "✅ Environment variables loaded successfully!"
echo "📊 Configuration:"
echo "   MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "   AWS Region: $AWS_DEFAULT_REGION"
echo "   S3 Endpoint: $MLFLOW_S3_ENDPOINT_URL"
echo ""
echo "🚀 You can now run: python3 main.py --mode train" 