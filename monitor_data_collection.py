import json
import os
import pandas as pd
from datetime import datetime

def check_progress():
    print("🔍 Checking Market Data Collection Progress")
    print("=" * 50)
    
    # Check progress file
    if os.path.exists('collection_progress.json'):
        with open('collection_progress.json', 'r') as f:
            progress = json.load(f)
        print(f"📊 Total collected: {progress.get('total_collected', 0)}")
        print(f"🎯 Target: {progress.get('target_rows', 'Unknown')}")
        print(f"⏱️  Elapsed: {progress.get('elapsed_hours', 0):.1f} hours")
        print(f"📈 Progress: {progress.get('completion_percentage', 0):.1f}%")
        print(f"🕒 Last update: {progress.get('last_update', 'Unknown')}")
    else:
        print("❌ No progress file found")
    
    # Check CSV file
    if os.path.exists('comprehensive_market_data.csv'):
        df = pd.read_csv('comprehensive_market_data.csv')
        print(f"📄 CSV rows: {len(df)}")
        if not df.empty:
            print(f"🕒 Latest timestamp: {df['timestamp'].iloc[-1]}")
            anomalies = len(df[df['anomaly_flag'].notna()])
            print(f"🚨 Anomalies found: {anomalies}")
    else:
        print("❌ No CSV file found yet")
    
    print(f"🕒 Check time: {datetime.now()}")
    print("=" * 50)

if __name__ == "__main__":
    check_progress()
