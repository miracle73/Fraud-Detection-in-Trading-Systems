#!/bin/bash

# One-Click Fraud Detection Training Setup
echo "🚀 One-Click Fraud Detection Training Setup"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command_exists python; then
    echo "❌ Python not found! Please install Python first."
    exit 1
fi

# Check pip
if ! command_exists pip; then
    echo "❌ pip not found! Please install pip first."
    exit 1
fi

# Check if data file exists
if [ ! -f "comprehensive_market_data.csv" ]; then
    echo "❌ Error: comprehensive_market_data.csv not found!"
    echo "Please ensure your data file is in the current directory."
    echo "Current directory contents:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create requirements file
echo "📦 Creating requirements file..."
cat > requirements_ml.txt << 'EOF'
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
EOF

# Install packages
echo "📦 Installing ML packages..."
pip install -r requirements_ml.txt

# Install screen if not available
echo "🖥️  Installing screen for background execution..."
if ! command_exists screen; then
    sudo apt-get update -qq
    sudo apt-get install -y screen
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p trained_models
mkdir -p training_logs
mkdir -p results

# Create monitor script
echo "🔍 Creating monitoring script..."
cat > monitor_training.py << 'EOF'
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
EOF

# Check if training script exists
if [ ! -f "fraud_detection_training.py" ]; then
    echo "❌ fraud_detection_training.py not found!"
    echo "Please create the training script first using the provided code."
    exit 1
fi

# Kill existing training sessions
echo "🧹 Cleaning up any existing training sessions..."
screen -S fraud_training -X quit 2>/dev/null || true

# Start training in background
echo "🚀 Starting fraud detection training in background..."
screen -dmS fraud_training python fraud_detection_training.py

# Wait a moment for training to start
sleep 5

# Check if training started successfully
if screen -list | grep -q "fraud_training"; then
    echo "✅ Training started successfully in screen session 'fraud_training'"
else
    echo "❌ Failed to start training session"
    exit 1
fi

# Display useful commands
echo ""
echo "🎯 TRAINING STARTED SUCCESSFULLY!"
echo "================================"
echo ""
echo "📋 Useful Commands:"
echo "  python monitor_training.py           - Check training progress"
echo "  screen -r fraud_training             - Reconnect to training session"
echo "  screen -ls                           - List all screen sessions"
echo "  tail -f fraud_detection_training.log - View live training logs"
echo "  screen -S fraud_training -X quit     - Stop training"
echo ""

echo "⏱️  Expected Training Time: 30-60 minutes"
echo "📊 Dataset: $(wc -l < comprehensive_market_data.csv) rows"
echo ""

# Show initial training status
echo "📊 Initial Training Status:"
echo "=========================="
python monitor_training.py

echo ""
echo "🎉 Setup complete! Training is running in background."
echo "💡 You can now close your browser/laptop - training will continue!"
echo ""
echo "📁 Results will be saved to:"
echo "   - fraud_detection_training_report.json (detailed results)"
echo "   - trained_models/ (all trained models)"
echo "   - fraud_detection_training.log (training logs)"