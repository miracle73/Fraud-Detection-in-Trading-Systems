#!/bin/bash
echo "🚀 Starting Fraud Detection Model Training"

# Check if data file exists
if [ ! -f "comprehensive_market_data.csv" ]; then
    echo "❌ Error: comprehensive_market_data.csv not found!"
    echo "Please ensure your data file is in the current directory"
    exit 1
fi

# Start training in screen session
screen -dmS fraud_training python fraud_detection_training.py

echo "✅ Training started in screen session 'fraud_training'"
echo ""
echo "📋 Useful commands:"
echo "  screen -r fraud_training    - Reconnect to training session"
echo "  screen -ls                  - List all sessions"
echo "  tail -f fraud_detection_training.log  - View live logs"
echo ""

# Show initial status
sleep 3
if [ -f "fraud_detection_training.log" ]; then
    echo "📊 Training started! Latest log entries:"
    tail -10 fraud_detection_training.log
fi