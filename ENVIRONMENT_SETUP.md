# Environment Setup Guide

## 🔋 Automatic Environment Loading (Recommended)

### Using direnv (Already Set Up!)

Your system is now configured to **automatically load environment variables** whenever you enter the project directory.

**✅ What's Configured:**
- `.envrc` file created with automatic loading
- `direnv` hook added to `~/.bashrc`
- Environment loads automatically when you `cd` into the project

**🚀 How it works:**
```bash
# When you enter the project directory:
cd ~/mlops-exp/openstack_rca_system
# Output: 🔋 Environment loaded for OpenStack RCA System
#         ✅ MLflow tracking: https://...
#         ✅ AWS region: ap-south-1
#         ✅ S3 endpoint: https://s3.ap-south-1.amazonaws.com

# Environment is automatically available:
python3 main.py --mode train  # Works immediately!
```

## 🔧 Manual Methods (Backup Options)

### Method 1: Using the Setup Script
```bash
# If direnv isn't working, use this:
source ./setup_env.sh
```

### Method 2: Direct Loading
```bash
# Load environment manually:
export $(cat .env | xargs)
```

### Method 3: Source the .env file
```bash
# Alternative manual method:
set -a && source .env && set +a
```

## 🔄 After System Restart

### With direnv (Automatic):
```bash
# Just navigate to the project directory:
cd ~/mlops-exp/openstack_rca_system
# Environment loads automatically!
```

### Without direnv (Manual):
```bash
cd ~/mlops-exp/openstack_rca_system
source ./setup_env.sh
# or
export $(cat .env | xargs)
```

## 🛠️ Troubleshooting

### If direnv isn't working:
1. **Check if hook is loaded:**
   ```bash
   grep "direnv hook" ~/.bashrc
   ```

2. **Reload bashrc:**
   ```bash
   source ~/.bashrc
   ```

3. **Re-allow the directory:**
   ```bash
   direnv allow
   ```

### If environment variables aren't loading:
1. **Check .env file exists:**
   ```bash
   ls -la .env
   ```

2. **Verify .env content:**
   ```bash
   cat .env
   ```

3. **Use manual loading:**
   ```bash
   source ./setup_env.sh
   ```

## 📋 Environment Variables

Your `.env` file contains:
- `MLFLOW_TRACKING_URI` - MLflow server URL
- `MLFLOW_ARTIFACT_ROOT` - S3 bucket for artifacts
- `MLFLOW_S3_ENDPOINT_URL` - Region-specific S3 endpoint
- `AWS_ACCESS_KEY_ID` - AWS credentials
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AWS_DEFAULT_REGION` - AWS region (ap-south-1)

## ✅ Verification

Test that everything works:
```bash
# Check environment is loaded:
echo $AWS_DEFAULT_REGION  # Should show: ap-south-1
echo $MLFLOW_S3_ENDPOINT_URL  # Should show: https://s3.ap-south-1.amazonaws.com

# Run training:
python3 main.py --mode train  # Should work without errors
```

## 🎯 Summary

**Your system is now configured for automatic environment loading!**
- ✅ No manual setup needed after restart
- ✅ Environment loads when you enter project directory
- ✅ Backup manual methods available
- ✅ Full MLflow + S3 integration working 