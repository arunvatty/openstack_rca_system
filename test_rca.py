#!/usr/bin/env python3
"""
Simple RCA test script - place in project root directory
Run with: python test_rca.py
"""

import os
import sys

def test_rca():
    """Test RCA analysis with your actual data"""
    
    print("🔍 OpenStack RCA Analysis Test")
    print("=" * 50)
    
    # 1. Check environment
    print("\n1️⃣ Checking Environment...")
    
    # Load .env file if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        print("✅ Found .env file, loading...")
        try:
            # Try to import python-dotenv
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ .env file loaded successfully")
        except ImportError:
            print("⚠️ python-dotenv not installed, loading .env manually...")
            # Manual .env loading
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        os.environ[key.strip()] = value
            print("✅ .env file loaded manually")
    else:
        print("⚠️ No .env file found")
    
    # Check if we're in the right directory
    required_files = ['main.py', 'data', 'models', 'utils']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files/directories: {missing_files}")
        print("Please run this script from the project root directory")
        print("Expected structure:")
        print("  openstack_rca_system/")
        print("  ├── main.py")
        print("  ├── data/")
        print("  ├── models/")
        print("  ├── utils/")
        print("  ├── .env")
        print("  └── test_rca.py  ← this file")
        return
    
    print("✅ Project structure looks good")
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        print("\nTroubleshooting:")
        print("1. Check your .env file contains:")
        print("   ANTHROPIC_API_KEY=your_key_here")
        print("2. Make sure there are no extra spaces or quotes")
        print("3. Try running: pip install python-dotenv")
        print("4. Or set manually: set ANTHROPIC_API_KEY=your_key_here")
        
        # Show .env file content for debugging
        if os.path.exists('.env'):
            print("\n🔍 Current .env file content:")
            with open('.env', 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    # Mask the actual key value for security
                    if 'ANTHROPIC_API_KEY' in line and '=' in line:
                        key_part = line.split('=')[0]
                        print(f"  Line {i}: {key_part}=***masked***")
                    else:
                        print(f"  Line {i}: {line.strip()}")
        return
    else:
        print(f"✅ API key found: {api_key[:8]}...")
    
    # 2. Import modules
    print("\n2️⃣ Loading modules...")
    try:
        from data.log_ingestion import LogIngestionManager
        from models.rca_analyzer import RCAAnalyzer
        from utils.feature_engineering import FeatureEngineer
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # 3. Load log data
    print("\n3️⃣ Loading OpenStack logs...")
    
    log_paths = ['logs/OpenStack_2k.log', 'OpenStack_2k.log']
    log_file = None
    
    for path in log_paths:
        if os.path.exists(path):
            log_file = path
            break
    
    if not log_file:
        print("❌ OpenStack_2k.log not found")
        print("Expected locations:")
        for path in log_paths:
            print(f"  - {path}")
        return
    
    print(f"✅ Found log file: {log_file}")
    
    # Load data
    try:
        ingestion_manager = LogIngestionManager()
        df = ingestion_manager.ingest_single_file(log_file)
        print(f"✅ Loaded {len(df)} log entries")
        
        if df.empty:
            print("❌ No log entries found in file")
            return
            
    except Exception as e:
        print(f"❌ Error loading logs: {e}")
        return
    
    # 4. Feature engineering
    print("\n4️⃣ Processing features...")
    try:
        feature_engineer = FeatureEngineer()
        df = feature_engineer.engineer_all_features(df)
        print(f"✅ Feature engineering complete: {df.shape}")
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return
    
    # 5. Show data summary
    print("\n5️⃣ Data Summary:")
    print(f"Total logs: {len(df)}")
    
    if 'level' in df.columns:
        level_counts = df['level'].value_counts()
        print("Log levels:")
        for level, count in level_counts.items():
            print(f"  {level}: {count}")
    
    if 'service_type' in df.columns:
        service_counts = df['service_type'].value_counts()
        print("Top services:")
        for service, count in service_counts.head(5).items():
            print(f"  {service}: {count}")
    
    # 6. Test RCA
    print("\n6️⃣ Testing RCA Analysis...")
    
    test_questions = [
        "What went wrong in my OpenStack environment?",
        "Are there any instance launch failures in these logs?",
        "Can you identify error patterns in my OpenStack system?"
    ]
    
    try:
        rca_analyzer = RCAAnalyzer(api_key)
        print("✅ RCA Analyzer initialized")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            print("-" * 60)
            
            try:
                results = rca_analyzer.analyze_issue(question, df)
                
                print(f"Category: {results['issue_category']}")
                print(f"Relevant logs: {results['relevant_logs_count']}")
                
                # Show RCA preview
                rca = results['root_cause_analysis']
                lines = rca.split('\n')[:8]  # First 8 lines
                
                print("RCA Analysis Preview:")
                for line in lines:
                    if line.strip():
                        print(f"  {line}")
                
                # Check if generic
                if "Analysis based on available log patterns" in rca:
                    print("⚠️  WARNING: Generic response detected")
                    
                    # Show debug info
                    patterns = results.get('patterns', {})
                    if patterns:
                        print("Patterns found:")
                        for key, value in patterns.items():
                            if isinstance(value, dict):
                                print(f"  {key}: {len(value)} items")
                            else:
                                print(f"  {key}: {value}")
                else:
                    print("✅ Detailed analysis generated")
                
            except Exception as e:
                print(f"❌ RCA failed: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ RCA Analyzer failed to initialize: {e}")
        return
    
    # 7. Direct log inspection
    print("\n7️⃣ Direct Log Analysis:")
    
    if 'message' in df.columns:
        # Check for error patterns
        error_patterns = ['error', 'failed', 'timeout', 'refused', 'no valid host']
        
        print("Error pattern detection:")
        for pattern in error_patterns:
            count = df['message'].str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                print(f"  '{pattern}': {count} occurrences")
        
        # Show sample error messages
        error_logs = df[df['level'].str.upper() == 'ERROR']
        if len(error_logs) > 0:
            print(f"\nSample error messages ({len(error_logs)} total errors):")
            for _, log in error_logs.head(3).iterrows():
                msg = str(log['message'])[:120]
                service = log.get('service_type', 'Unknown')
                print(f"  [{service}] {msg}...")
        else:
            print("\nNo ERROR level logs found")
    
    print("\n✅ RCA Test Complete!")
    print("\nIf you're still getting generic responses:")
    print("1. Check that your API key is valid")
    print("2. Verify network connectivity to Anthropic API")
    print("3. Try more specific questions about the patterns shown above")

if __name__ == "__main__":
    test_rca()