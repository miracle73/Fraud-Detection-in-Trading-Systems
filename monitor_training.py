import json
import os
import time
from datetime import datetime

def monitor_training():
    """Monitor the training progress"""
    print("🔍 Fraud Detection Training Monitor")
    print("=" * 50)
    
    # Check if training is running
    try:
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'fraud_detection_training.py'], 
                               capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training process is running")
        else:
            print("❌ Training process not found")
    except:
        pass
    
    # Check log file
    if os.path.exists('fraud_detection_training.log'):
        print(f"\n📋 Latest log entries:")
        with open('fraud_detection_training.log', 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.strip()}")
    
    # Check results file
    if os.path.exists('fraud_detection_training_report.json'):
        with open('fraud_detection_training_report.json', 'r') as f:
            report = json.load(f)
        
        print(f"\n📊 Training Progress:")
        summary = report.get('training_summary', {})
        print(f"   Models trained: {summary.get('models_trained', 0)}")
        print(f"   Dataset size: {summary.get('dataset_size', 0):,}")
        print(f"   Fraud cases: {summary.get('fraud_cases', 0):,}")
        
        # Show top models
        rankings = report.get('model_rankings', [])
        if rankings:
            print(f"\n🏆 Top 3 Models:")
            for i, model in enumerate(rankings[:3], 1):
                print(f"   {i}. {model['model_name']}: F1={model['f1_score']:.4f}")
    
    # Check saved models
    if os.path.exists('trained_models'):
        model_files = [f for f in os.listdir('trained_models') if f.endswith('.joblib')]
        print(f"\n💾 Saved models: {len(model_files)}")
        for model_file in model_files[:5]:
            print(f"   - {model_file}")
    
    print(f"\n🕒 Check time: {datetime.now()}")
    print("=" * 50)

if __name__ == "__main__":
    monitor_training()
