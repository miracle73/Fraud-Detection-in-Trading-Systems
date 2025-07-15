#!/bin/bash

echo "🚀 Starting Market Data Collection on GitHub Codespaces"
echo "=" * 60

# Check if requirements are installed
if ! python -c "import yfinance, pandas, pytz" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start keep-alive process (optional, helps prevent timeouts)
if ! pgrep -f "keep_alive" > /dev/null; then
    nohup bash -c 'while true; do echo "🟢 Codespace alive at $(date)"; sleep 300; done' > keep_alive.log 2>&1 &
    echo "✅ Keep-alive process started"
fi

# Check if screen session already exists
if screen -list | grep -q "data_collection"; then
    echo "⚠️  Screen session 'data_collection' already exists"
    echo "📋 Use 'screen -r data_collection' to reconnect"
    echo "📋 Or kill it first: 'screen -S data_collection -X quit'"
    exit 1
fi

# Start data collection in screen
screen -dmS data_collection python market_data_collector.py

echo "✅ Collection started in screen session 'data_collection'"
echo ""
echo "📋 Useful commands:"
echo "  screen -r data_collection  - Reconnect to session"
echo "  python monitor.py          - Check progress"
echo "  screen -ls                 - List all sessions"
echo "  tail -f data_collection.log - View live logs"
echo ""

# Show initial status after a brief delay
sleep 3
if [ -f "monitor.py" ]; then
    python monitor.py
fi