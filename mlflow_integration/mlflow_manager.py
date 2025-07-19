"""
MLflow Manager for OpenStack RCA System - Cleaned and Precise

This module provides streamlined MLflow integration with single keras upload and S3 organization.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
    
    # Handle TensorFlow imports separately to avoid keras import issues
    TENSORFLOW_AVAILABLE = False
    try:
        # Don't import mlflow.tensorflow here - do it later when needed
        import tensorflow as tf
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        pass
        
except ImportError:
    MLFLOW_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False

from config.config import Config

logger = logging.getLogger(__name__)

class MLflowManager:
    """
    Streamlined MLflow manager with single keras upload and precise S3 organization
    """
    
    def __init__(self, 
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "openstack_rca_system",
                 enable_mlflow: bool = True):
        """Initialize MLflow manager"""
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.client = None
        self.active_run = None
        
        if not self.enable_mlflow:
            if not MLFLOW_AVAILABLE:
                logger.warning("MLflow not available. Install mlflow to enable experiment tracking.")
            else:
                logger.info("MLflow tracking disabled.")
            return
        
        try:
            # Set tracking URI
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            elif hasattr(Config, 'MLFLOW_TRACKING_URI') and Config.MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
                logger.info(f"MLflow tracking URI set from config: {Config.MLFLOW_TRACKING_URI}")
            
            # Set AWS credentials for S3 artifact storage
            if hasattr(Config, 'MLFLOW_CONFIG'):
                mlflow_config = Config.MLFLOW_CONFIG
                if mlflow_config.get('aws_access_key_id') and mlflow_config.get('aws_secret_access_key'):
                    os.environ['AWS_ACCESS_KEY_ID'] = mlflow_config['aws_access_key_id']
                    os.environ['AWS_SECRET_ACCESS_KEY'] = mlflow_config['aws_secret_access_key']
                    if mlflow_config.get('s3_endpoint_url'):
                        os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_config['s3_endpoint_url']
                    logger.info("AWS credentials configured for S3 artifact storage")
            
            self.client = MlflowClient()
            
            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"Creating new MLflow experiment: {experiment_name}")
                artifact_root = None
                if hasattr(Config, 'MLFLOW_CONFIG') and Config.MLFLOW_CONFIG.get('artifact_root'):
                    artifact_root = Config.MLFLOW_CONFIG['artifact_root']
                
                try:
                    experiment_id = mlflow.create_experiment(
                        experiment_name, 
                        artifact_location=artifact_root
                    )
                    logger.info(f"✅ Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
                    
                    if artifact_root:
                        logger.info(f"🗃️ Artifact location: {artifact_root}")
                        
                    # Set the newly created experiment as active
                    mlflow.set_experiment(experiment_name)
                    
                except Exception as create_error:
                    logger.error(f"❌ Failed to create experiment {experiment_name}: {create_error}")
                    # Try to set experiment by name in case it was created by another process
                    try:
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"✅ Found and set existing experiment: {experiment_name}")
                    except Exception as set_error:
                        logger.error(f"❌ Failed to set experiment: {set_error}")
                        raise create_error
            else:
                mlflow.set_experiment(experiment_name)
                logger.info(f"✅ Using existing MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
                
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.enable_mlflow = False

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start MLflow run with meaningful naming"""
        if not self.enable_mlflow:
            return None
        
        try:
            # Generate meaningful run name if not provided
            if not run_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{self.experiment_name}_{timestamp}"
            

            
            # Start the run
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self.active_run = run
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            logger.info(f"Run name: {run_name}")
            logger.info(f"Artifact URI: {run.info.artifact_uri}")
            
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None

    def end_run(self, status: str = "FINISHED"):
        """End MLflow run with S3 summary"""
        if not self.enable_mlflow or not self.active_run:
            return
        
        try:
            mlflow.end_run(status=status)
            logger.info(f"MLflow run ended with status: {status}")
            self.active_run = None
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        if not self.enable_mlflow or not self.active_run:
            return
        
        try:
            # Convert numpy types to Python types for MLflow compatibility
            mlflow_params = {}
            for key, value in params.items():
                if hasattr(value, 'item'):  # numpy scalars
                    mlflow_params[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    mlflow_params[key] = str(value)
                else:
                    mlflow_params[key] = value
            
            mlflow.log_params(mlflow_params)
            logger.info(f"Logged {len(mlflow_params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to MLflow"""
        if not self.enable_mlflow or not self.active_run:
            return
        
        try:
            # Convert numpy types to Python types
            mlflow_metrics = {}
            for key, value in metrics.items():
                if hasattr(value, 'item'):  # numpy scalars
                    mlflow_metrics[key] = float(value.item())
                else:
                    mlflow_metrics[key] = float(value)
            
            mlflow.log_metrics(mlflow_metrics, step=step)
            logger.info(f"Logged {len(mlflow_metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_model_with_versioning(self, 
                                  model: Any, 
                                  model_name: str = "lstm_model",
                                  model_type: str = "tensorflow",
                                  model_stage: str = "Staging",
                                  artifacts: Optional[Dict[str, str]] = None,
                                  signature=None,
                                  input_example=None) -> Optional[str]:
        """
        Log model ONCE in keras format with S3 organization
        """
        if not self.enable_mlflow:
            return None
        
        try:
            registered_model_name = f"{self.experiment_name}_{model_name}"
            
            # Get next model version for naming
            try:
                model_versions = self.client.search_model_versions(f"name='{registered_model_name}'")
                next_version = len(model_versions) + 1
            except Exception:
                next_version = 1
            
            # Save model as .keras file with proper name and version
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            keras_filename = f"{model_name}_v{next_version}.keras"
            keras_path = os.path.join(temp_dir, keras_filename)
            
            logger.info(f"📦 Saving ONLY {keras_filename} to S3")
            model.save(keras_path, save_format='keras')
            
            # Upload to S3 with meaningful folder name
            import boto3
            from botocore.exceptions import ClientError
            
            # Use meaningful run name instead of random run ID
            run_name = f"openstack-rca-system-staging_{next_version:04d}"
            meaningful_s3_key = f"group6-capstone/{run_name}/models/{keras_filename}"
            
            try:
                s3_client = boto3.client('s3')
                bucket_name = "chandanbam-bucket"
                
                # Upload ONLY to S3 with meaningful path (no MLflow duplicate)
                s3_client.upload_file(keras_path, bucket_name, meaningful_s3_key)
                logger.info(f"✅ Uploaded ONLY to: s3://{bucket_name}/{meaningful_s3_key}")
                
            except Exception as s3_error:
                logger.error(f"❌ S3 upload failed: {s3_error}")
                meaningful_s3_key = None
            
            # Manual model registration (only if S3 upload succeeded)
            if meaningful_s3_key:
                try:
                    # Create model registry entry if it doesn't exist
                    try:
                        self.client.create_registered_model(registered_model_name)
                    except Exception:
                        pass  # Model already exists
                    
                    # Use meaningful S3 path for model URI
                    model_uri = f"s3://chandanbam-bucket/{meaningful_s3_key}"
                    model_version_obj = self.client.create_model_version(
                        name=registered_model_name,
                        source=model_uri,
                        run_id=self.active_run.info.run_id
                    )
                    model_version = model_version_obj.version
                    logger.info(f"✅ Model registered as version: {model_version}")
                    
                    # Set tags for tracking
                    mlflow.set_tag("model_version", model_version)
                    mlflow.set_tag("model_stage", model_stage)
                    mlflow.set_tag("model_format", "keras")
                    mlflow.set_tag("keras_filename", keras_filename)
                    mlflow.set_tag("meaningful_folder", run_name)
                    mlflow.set_tag("s3_path", meaningful_s3_key)
                    
                    # Clean up temp file
                    os.remove(keras_path)
                    os.rmdir(temp_dir)
                    
                    return model_version
                    
                except Exception as reg_error:
                    logger.error(f"❌ Model registration failed: {reg_error}")
                    # Clean up temp file
                    try:
                        os.remove(keras_path)
                        os.rmdir(temp_dir)
                    except:
                        pass
                    return None
            else:
                logger.error("❌ Cannot register model: S3 upload failed")
                # Clean up temp file
                try:
                    os.remove(keras_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return None
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return None

    def load_model_with_versioning(self,
                                   model_name: str = "lstm_model", 
                                   version: Union[str, int] = "latest",
                                   stage: Optional[str] = None) -> Optional[Any]:
        """Load model from meaningful S3 folder names by version"""
        if not self.enable_mlflow:
            return None
        
        try:
            # Load directly from meaningful S3 folder names
            logger.info("🔍 Searching for latest model in meaningful S3 folders...")
            return self._load_latest_model_from_meaningful_folders()
            
        except Exception as e:
            logger.error(f"Failed to load model from meaningful folders: {e}")
            return None

    def _load_latest_model_from_meaningful_folders(self):
        """Load the latest model from meaningful S3 folder names"""
        try:
            import boto3
            import tempfile
            import os
            from botocore.exceptions import ClientError
            
            # Get S3 configuration
            if not hasattr(Config, 'MLFLOW_CONFIG') or not Config.MLFLOW_CONFIG.get('artifact_root'):
                logger.error("❌ No S3 configuration found")
                return None
                
            base_artifact_uri = Config.MLFLOW_CONFIG['artifact_root']
            if not base_artifact_uri or not base_artifact_uri.startswith('s3://'):
                logger.error("❌ Not using S3 artifact storage")
                return None
            
            # Parse S3 URI
            s3_parts = base_artifact_uri.replace('s3://', '').split('/', 1)
            bucket_name = s3_parts[0]
            s3_prefix = s3_parts[1] if len(s3_parts) > 1 else ''
            
            logger.info(f"🔍 Searching for latest model in S3 bucket: {bucket_name}")
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # List all folders matching the meaningful pattern
            prefix_pattern = f"{s3_prefix}/openstack-rca-system-staging_" if s3_prefix else "openstack-rca-system-staging_"
            
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix_pattern.strip('/'),
                Delimiter='/'
            )
            
            if 'CommonPrefixes' not in response:
                logger.error("❌ No model folders found in S3")
                return None
            
            # Find the latest version folder
            latest_folder = None
            latest_version = 0
            
            for prefix_info in response['CommonPrefixes']:
                folder_name = prefix_info['Prefix'].rstrip('/')
                folder_basename = folder_name.split('/')[-1]
                
                # Extract version number from folder name (e.g., openstack-rca-system-staging_0016)
                if folder_basename.startswith('openstack-rca-system-staging_'):
                    try:
                        version_str = folder_basename.split('_')[-1]
                        version = int(version_str)
                        if version > latest_version:
                            latest_version = version
                            latest_folder = folder_name
                    except ValueError:
                        continue
            
            if not latest_folder:
                logger.error("❌ No valid versioned model folders found")
                return None
            
            logger.info(f"📦 Found latest model folder: {latest_folder} (version {latest_version})")
            
            # Find the keras model file in the latest folder
            model_prefix = f"{latest_folder}/models/"
            model_response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=model_prefix
            )
            
            keras_file = None
            if 'Contents' in model_response:
                for obj in model_response['Contents']:
                    if obj['Key'].endswith('.keras'):
                        keras_file = obj['Key']
                        break
            
            if not keras_file:
                logger.error("❌ No .keras model file found in S3")
                return None
            
            logger.info(f"⬇️ Downloading model from meaningful folder: {keras_file}")
            
            # Download the model to a temporary file
            temp_dir = tempfile.mkdtemp()
            local_model_path = os.path.join(temp_dir, 'model.keras')
            
            s3_client.download_file(bucket_name, keras_file, local_model_path)
            logger.info(f"✅ Model downloaded to: {local_model_path}")
            
            # Load the model
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(local_model_path)
                logger.info("🎯 Model loaded successfully from meaningful S3 folder")
                
                # Clean up temp file
                try:
                    os.remove(local_model_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                
                return model
                
            except Exception as load_error:
                logger.error(f"❌ Failed to load downloaded model: {load_error}")
                # Clean up temp file
                try:
                    os.remove(local_model_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return None
                
        except Exception as e:
            logger.error(f"❌ Meaningful folder model download failed: {e}")
            return None

    @property
    def is_enabled(self) -> bool:
        """Check if MLflow is enabled"""
        return self.enable_mlflow

    @property
    def tracking_uri(self) -> Optional[str]:
        """Get current tracking URI"""
        return mlflow.get_tracking_uri() if self.enable_mlflow else None

    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current run"""
        if not self.enable_mlflow or not self.active_run:
            return None
        
        try:
            run_info = {
                "run_id": self.active_run.info.run_id,
                "experiment_id": self.active_run.info.experiment_id,
                "status": self.active_run.info.status,
                "start_time": self.active_run.info.start_time,
                "artifact_uri": self.active_run.info.artifact_uri
            }
            return run_info
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return None 