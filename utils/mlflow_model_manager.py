#!/usr/bin/env python3
"""
MLflow Model Manager Utility

This script provides comprehensive model management capabilities for the OpenStack RCA System,
including versioning, S3 storage, model registry operations, and model lifecycle management.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

try:
    from mlflow_integration.mlflow_manager import MLflowManager
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowModelManagerCLI:
    """Command-line interface for MLflow model management"""
    
    def __init__(self):
        self.mlflow_manager = None
        self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        """Initialize MLflow manager"""
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow not available. Please install mlflow.")
            return
        
        try:
            self.mlflow_manager = MLflowManager(
                tracking_uri=Config.MLFLOW_CONFIG.get('tracking_uri'),
                experiment_name=Config.MLFLOW_CONFIG.get('experiment_name', 'openstack_rca_system_auto'),
                enable_mlflow=True
            )
            
            if not self.mlflow_manager.is_enabled:
                logger.error("MLflow initialization failed")
                self.mlflow_manager = None
            else:
                logger.info("✅ MLflow manager initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.mlflow_manager = None
    
    def list_models(self) -> bool:
        """List all models in the registry"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            registry_info = self.mlflow_manager.get_model_registry_info()
            
            if not registry_info or not registry_info.get('models'):
                logger.info("No models found in registry")
                return True
            
            print("\n" + "="*80)
            print("MLflow MODEL REGISTRY")
            print("="*80)
            print(f"Experiment: {registry_info['experiment_name']}")
            print(f"Total Models: {registry_info['total_models']}")
            print()
            
            for model_name, model_info in registry_info['models'].items():
                print(f"📊 Model: {model_name}")
                print(f"   Registered Name: {model_info['registered_name']}")
                print(f"   Total Versions: {model_info['total_versions']}")
                print(f"   Latest Version: v{model_info['latest_version']}")
                print(f"   Created: {model_info['creation_timestamp']}")
                print(f"   Updated: {model_info['last_updated_timestamp']}")
                print()
                
                # Show version details
                if model_info['versions']:
                    print("   📋 Version History:")
                    for version in model_info['versions'][:5]:  # Show top 5 versions
                        print(f"      v{version['version']} ({version['stage']}) - Run: {version['run_id'][:8]}...")
                    
                    if len(model_info['versions']) > 5:
                        print(f"      ... and {len(model_info['versions']) - 5} more versions")
                print()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return False
    
    def show_model_details(self, model_name: str = "lstm_model") -> bool:
        """Show detailed information about a specific model"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            versions = self.mlflow_manager.list_model_versions(model_name)
            
            if not versions:
                logger.info(f"No versions found for model: {model_name}")
                return True
            
            print("\n" + "="*80)
            print(f"MODEL DETAILS: {model_name}")
            print("="*80)
            
            for version in versions:
                print(f"📦 Version {version['version']}")
                print(f"   Stage: {version['stage']}")
                print(f"   Status: {version['status']}")
                print(f"   Run ID: {version['run_id']}")
                print(f"   Created: {version['creation_timestamp']}")
                print(f"   Updated: {version['last_updated_timestamp']}")
                print(f"   Source: {version['source']}")
                print()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to show model details: {e}")
            return False
    
    def load_model_info(self, model_name: str = "lstm_model", version: str = "latest") -> bool:
        """Load and display model information"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            print(f"\n🔍 Loading model: {model_name} (version: {version})")
            
            model_result = self.mlflow_manager.load_model_with_versioning(
                model_name=model_name,
                version=version,
                model_type="tensorflow"
            )
            
            if not model_result:
                logger.error("Failed to load model")
                return False
            
            metadata = model_result['metadata']
            
            print("\n" + "="*60)
            print("MODEL LOAD INFORMATION")
            print("="*60)
            print(f"✅ Model loaded successfully!")
            print(f"   Model Name: {metadata.get('model_name', 'unknown')}")
            print(f"   Registered Name: {metadata.get('registered_name', 'unknown')}")
            print(f"   Version: {metadata.get('version', 'unknown')}")
            print(f"   Stage: {metadata.get('stage', 'unknown')}")
            print(f"   Run ID: {metadata.get('run_id', 'unknown')}")
            print(f"   Load Method: {metadata.get('load_method', 'unknown')}")
            print(f"   Model URI: {metadata.get('model_uri', 'unknown')}")
            print(f"   S3 Location: {metadata.get('s3_location', 'unknown')}")
            print(f"   Status: {metadata.get('status', 'unknown')}")
            
            if metadata.get('creation_timestamp'):
                print(f"   Created: {metadata.get('creation_timestamp')}")
            if metadata.get('last_updated_timestamp'):
                print(f"   Updated: {metadata.get('last_updated_timestamp')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")
            return False
    
    def promote_model(self, model_name: str = "lstm_model", version: str = "latest", stage: str = "Production") -> bool:
        """Promote a model to a specific stage"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            registered_model_name = f"{self.mlflow_manager.experiment_name}_{model_name}"
            
            # Get actual version if "latest" specified
            if version == "latest":
                versions = self.mlflow_manager.list_model_versions(model_name)
                if not versions:
                    logger.error("No versions found")
                    return False
                version = versions[0]['version']
            
            # Transition model stage
            self.mlflow_manager.client.transition_model_version_stage(
                name=registered_model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"✅ Model {model_name} v{version} promoted to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def archive_model(self, model_name: str = "lstm_model", version: str = "latest") -> bool:
        """Archive a model version"""
        return self.promote_model(model_name, version, "Archived")
    
    def compare_models(self, model_name: str = "lstm_model", versions: List[str] = None) -> bool:
        """Compare different versions of a model"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            all_versions = self.mlflow_manager.list_model_versions(model_name)
            
            if not all_versions:
                logger.info(f"No versions found for model: {model_name}")
                return True
            
            # Use specified versions or compare latest 3
            if not versions:
                versions = [v['version'] for v in all_versions[:3]]
            
            print("\n" + "="*80)
            print(f"MODEL COMPARISON: {model_name}")
            print("="*80)
            
            print(f"{'Version':<10} {'Stage':<12} {'Status':<10} {'Run ID':<20} {'Created'}")
            print("-" * 80)
            
            for version in versions:
                version_info = next((v for v in all_versions if v['version'] == version), None)
                if version_info:
                    created = version_info['creation_timestamp'][:19] if version_info['creation_timestamp'] else 'unknown'
                    print(f"v{version_info['version']:<9} {version_info['stage']:<12} {version_info['status']:<10} "
                          f"{version_info['run_id'][:18]:<20} {created}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return False
    
    def export_model_info(self, output_file: str = "model_registry.json") -> bool:
        """Export model registry information to JSON"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            registry_info = self.mlflow_manager.get_model_registry_info()
            
            # Add timestamp
            registry_info['export_timestamp'] = datetime.now().isoformat()
            registry_info['export_tool'] = 'mlflow_model_manager'
            
            with open(output_file, 'w') as f:
                json.dump(registry_info, f, indent=2, default=str)
            
            logger.info(f"✅ Model registry exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model info: {e}")
            return False
    
    def cleanup_old_versions(self, model_name: str = "lstm_model", keep_versions: int = 5) -> bool:
        """Archive old model versions (keep only the latest N versions)"""
        if not self.mlflow_manager:
            logger.error("MLflow not available")
            return False
        
        try:
            versions = self.mlflow_manager.list_model_versions(model_name)
            
            if len(versions) <= keep_versions:
                logger.info(f"Only {len(versions)} versions exist, no cleanup needed")
                return True
            
            # Archive versions beyond the keep limit
            versions_to_archive = versions[keep_versions:]
            
            logger.info(f"Archiving {len(versions_to_archive)} old versions...")
            
            for version in versions_to_archive:
                if version['stage'] not in ['Archived', 'Production']:  # Don't archive Production models
                    try:
                        self.promote_model(model_name, version['version'], "Archived")
                        logger.info(f"   Archived v{version['version']}")
                    except Exception as e:
                        logger.warning(f"   Failed to archive v{version['version']}: {e}")
            
            logger.info("✅ Cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return False

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="MLflow Model Manager for OpenStack RCA System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/mlflow_model_manager.py list
  python utils/mlflow_model_manager.py details --model lstm_model
  python utils/mlflow_model_manager.py load --model lstm_model --version latest
  python utils/mlflow_model_manager.py promote --model lstm_model --version 3 --stage Production
  python utils/mlflow_model_manager.py compare --model lstm_model
  python utils/mlflow_model_manager.py export --output models.json
  python utils/mlflow_model_manager.py cleanup --model lstm_model --keep 3
        """
    )
    
    parser.add_argument('command', 
                       choices=['list', 'details', 'load', 'promote', 'archive', 'compare', 'export', 'cleanup'],
                       help='Command to execute')
    
    parser.add_argument('--model', '-m', 
                       default='lstm_model',
                       help='Model name (default: lstm_model)')
    
    parser.add_argument('--version', '-v',
                       default='latest',
                       help='Model version (default: latest)')
    
    parser.add_argument('--stage', '-s',
                       choices=['None', 'Staging', 'Production', 'Archived'],
                       default='Production',
                       help='Model stage for promotion (default: Production)')
    
    parser.add_argument('--output', '-o',
                       default='model_registry.json',
                       help='Output file for export (default: model_registry.json)')
    
    parser.add_argument('--keep', '-k',
                       type=int,
                       default=5,
                       help='Number of versions to keep during cleanup (default: 5)')
    
    parser.add_argument('--versions',
                       nargs='+',
                       help='Specific versions to compare')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = MLflowModelManagerCLI()
    
    if not manager.mlflow_manager:
        logger.error("Failed to initialize MLflow manager")
        sys.exit(1)
    
    # Execute command
    success = False
    
    if args.command == 'list':
        success = manager.list_models()
    elif args.command == 'details':
        success = manager.show_model_details(args.model)
    elif args.command == 'load':
        success = manager.load_model_info(args.model, args.version)
    elif args.command == 'promote':
        success = manager.promote_model(args.model, args.version, args.stage)
    elif args.command == 'archive':
        success = manager.archive_model(args.model, args.version)
    elif args.command == 'compare':
        success = manager.compare_models(args.model, args.versions)
    elif args.command == 'export':
        success = manager.export_model_info(args.output)
    elif args.command == 'cleanup':
        success = manager.cleanup_old_versions(args.model, args.keep)
    
    if success:
        logger.info("✅ Command completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 