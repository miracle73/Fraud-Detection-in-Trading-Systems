import yfinance as yf
import pandas as pd
import time
import signal
import sys
from datetime import datetime, timedelta
import pytz
import os
import random
import logging
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

class EnhancedMarketDataCollector:
    def __init__(self, target_rows: int = 10000, max_runtime_hours: int = 12):
        # Expanded symbol list for faster data collection
        self.large_cap = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']
        self.mid_cap = ['AMD', 'CRM', 'UBER', 'ABNB', 'COIN', 'ROKU', 'SQ', 'PYPL']
        self.volatile_stocks = ['GME', 'AMC', 'BB', 'PLTR', 'WISH', 'CLOV', 'SPCE', 'TLRY']
        self.penny_stocks = ['SNDL', 'GNUS', 'BBBY', 'NAKD', 'EXPR', 'KOSS', 'NOK', 'SIRI']
        self.etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'ARKK', 'SOXL', 'TQQQ']

        # Combine all symbols (40 total for faster collection)
        self.all_symbols = (self.large_cap + self.mid_cap + self.volatile_stocks +
                           self.penny_stocks + self.etfs)

        self.target_rows = target_rows
        self.max_runtime_hours = max_runtime_hours
        self.max_runtime_seconds = max_runtime_hours * 3600
        self.start_time = time.time()
        self.session_data = []
        self.total_collected = 0
        self.running = True
        self.progress_file = 'collection_progress.json'

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Load existing progress
        self.load_progress()

        logging.info(f"Initialized collector with {len(self.all_symbols)} symbols")
        logging.info(f"Target: {self.target_rows} rows | Max runtime: {self.max_runtime_hours} hours")
        logging.info(f"Current: {self.total_collected} rows")

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        elapsed_time = time.time() - self.start_time
        logging.info("Received shutdown signal. Saving progress...")
        logging.info(f"Total runtime: {elapsed_time/3600:.1f} hours")
        logging.info(f"Collected {self.total_collected} rows")
        self.running = False
        self.save_progress()
        sys.exit(0)

    def load_progress(self):
        """Load collection progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.total_collected = progress.get('total_collected', 0)
                    logging.info(f"Resumed from {self.total_collected} collected rows")
            except Exception as e:
                logging.error(f"Error loading progress: {e}")

    def save_progress(self):
        """Save collection progress to file"""
        try:
            elapsed_time = time.time() - self.start_time
            progress = {
                'total_collected': self.total_collected,
                'last_update': datetime.now().isoformat(),
                'target_rows': self.target_rows,
                'max_runtime_hours': self.max_runtime_hours,
                'elapsed_hours': elapsed_time / 3600,
                'completion_percentage': (self.total_collected / self.target_rows) * 100
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving progress: {e}")

    def is_market_open(self) -> tuple:
        """Check if US market is currently open"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)

        # Weekend check
        if now.weekday() >= 5:
            return False, "Weekend"

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)

        if market_open <= now <= market_close:
            return True, "Market Hours"
        elif pre_market_start <= now < market_open:
            return True, "Pre-Market"
        elif market_close < now <= after_hours_end:
            return True, "After Hours"
        else:
            return False, "Closed"

    def collect_symbol_data(self, symbol: str, session_type: str = "market_hours") -> Optional[Dict]:
        """Collect enhanced data for fraud detection with error handling"""
        try:
            ticker = yf.Ticker(symbol)

            # Get recent data with retry mechanism
            for attempt in range(3):
                try:
                    hist_1d = ticker.history(period="1d", interval="1m")
                    hist_5d = ticker.history(period="5d", interval="5m")
                    break
                except Exception as e:
                    if attempt == 2:
                        logging.warning(f"Failed to get data for {symbol} after 3 attempts: {e}")
                        return None
                    time.sleep(1)

            if hist_1d.empty:
                return None

            latest = hist_1d.iloc[-1]

            # Calculate enhanced features
            volume_5d = hist_5d['Volume'].replace(0, pd.NA).dropna()
            avg_volume_5d = volume_5d.mean() if not volume_5d.empty else 0

            # Price movement features
            price_changes = hist_1d['Close'].pct_change().dropna()
            recent_volatility = price_changes.tail(10).std() if len(price_changes) >= 10 else 0

            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'high': latest['High'],
                'low': latest['Low'],
                'open': latest['Open'],
                'session_type': session_type,

                # Enhanced features for fraud detection
                'volume_5d_avg': avg_volume_5d,
                'volume_ratio': latest['Volume'] / (avg_volume_5d + 1),
                'price_volatility': recent_volatility,
                'price_spread': (latest['High'] - latest['Low']) / latest['Close'] if latest['Close'] > 0 else 0,
                'price_position': (latest['Close'] - latest['Low']) / (latest['High'] - latest['Low'] + 0.001),

                # Market timing features
                'hour': datetime.now().hour,
                'minute': datetime.now().minute,
                'is_opening_hour': 1 if 9 <= datetime.now().hour <= 10 else 0,
                'is_closing_hour': 1 if 15 <= datetime.now().hour <= 16 else 0,
                'is_pre_market': 1 if 4 <= datetime.now().hour < 9 else 0,

                # Initialize flags
                'anomaly_flag': None,
                'fraud_probability': 0.0
            }

        except Exception as e:
            logging.error(f"Error collecting {symbol}: {e}")
            return None

    def enhanced_anomaly_detection(self, data: Dict) -> Dict:
        """Enhanced anomaly detection for different fraud types"""
        if not data:
            return data

        anomalies = []
        details = []
        fraud_score = 0.0

        # 1. PUMP-AND-DUMP INDICATORS
        if data['volume_ratio'] > 10:
            anomalies.append('MASSIVE_VOLUME_SPIKE')
            details.append(f"Volume {data['volume_ratio']:.1f}x normal")
            fraud_score += 0.4

        if data['volume_ratio'] > 5 and data.get('price_volatility', 0) > 0.05:
            anomalies.append('PUMP_PATTERN')
            details.append("High volume + high volatility")
            fraud_score += 0.3

        # 2. SPOOFING INDICATORS
        if data['price_spread'] > 0.05:
            anomalies.append('UNUSUAL_SPREAD')
            details.append(f"Spread {data['price_spread']:.2%}")
            fraud_score += 0.2

        # 3. TIMING-BASED ANOMALIES
        if data['session_type'] != 'market_hours' and data['volume'] > data['volume_5d_avg']:
            anomalies.append('OFF_HOURS_ACTIVITY')
            details.append("Unusual activity outside market hours")
            fraud_score += 0.3

        # 4. PENNY STOCK SPECIFIC
        if data['symbol'] in self.penny_stocks:
            if data['volume_ratio'] > 3:
                anomalies.append('PENNY_STOCK_SPIKE')
                details.append("Penny stock unusual activity")
                fraud_score += 0.2

        # 5. OPENING/CLOSING MANIPULATION
        if (data['is_opening_hour'] or data['is_closing_hour']) and data['volume_ratio'] > 8:
            anomalies.append('MARKET_TIMING_MANIPULATION')
            details.append("Suspicious activity during market open/close")
            fraud_score += 0.3

        # 6. PRE-MARKET MANIPULATION
        if data.get('is_pre_market', 0) == 1 and data['volume_ratio'] > 5:
            anomalies.append('PRE_MARKET_MANIPULATION')
            details.append("Unusual pre-market activity")
            fraud_score += 0.25

        # Set flags
        if anomalies:
            data['anomaly_flag'] = ';'.join(anomalies)
            data['anomaly_details'] = ';'.join(details)
            data['fraud_probability'] = min(fraud_score, 1.0)

        return data

    def collect_continuous_data(self, filename: str = 'comprehensive_market_data.csv'):
        """Collect data continuously until target is reached OR time limit exceeded"""
        logging.info(f"Starting continuous data collection with time limit")
        logging.info(f"Target: {self.target_rows} rows | Current: {self.total_collected} rows")
        logging.info(f"Max runtime: {self.max_runtime_hours} hours")

        cycle_count = 0
        last_status_update = time.time()

        while self.running and self.total_collected < self.target_rows:
            # Check time limit
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.max_runtime_seconds:
                logging.info(f"⏰ TIME LIMIT REACHED! Ran for {elapsed_time/3600:.1f} hours")
                logging.info(f"📊 Collected {self.total_collected} rows out of {self.target_rows} target")
                self.running = False
                break
                
            cycle_count += 1
            cycle_start = time.time()

            # Check market status
            is_open, status = self.is_market_open()

            # Adjust collection interval based on market status
            if status == "Market Hours":
                interval = 30  # 30 seconds during market hours
            elif status in ["Pre-Market", "After Hours"]:
                interval = 60  # 1 minute during extended hours
            else:  # Closed/Weekend
                interval = 300  # 5 minutes when closed

            remaining_hours = (self.max_runtime_seconds - elapsed_time) / 3600
            logging.info(f"Cycle {cycle_count} | Status: {status} | Elapsed: {elapsed_time/3600:.1f}h | Remaining: {remaining_hours:.1f}h")

            batch_collected = 0
            batch_anomalies = 0

            # Collect data from all symbols
            for symbol in self.all_symbols:
                if not self.running:
                    break

                # Check time limit during symbol collection too
                if time.time() - self.start_time >= self.max_runtime_seconds:
                    logging.info("⏰ Time limit reached during symbol collection")
                    self.running = False
                    break

                data = self.collect_symbol_data(symbol, status.lower().replace(' ', '_'))

                if data:
                    data = self.enhanced_anomaly_detection(data)

                    # Save to CSV
                    self.save_to_csv(data, filename)

                    # Update counters
                    batch_collected += 1
                    self.total_collected += 1

                    if data.get('anomaly_flag'):
                        batch_anomalies += 1

                    # Check if target reached
                    if self.total_collected >= self.target_rows:
                        logging.info(f"🎯 TARGET REACHED! Collected {self.total_collected} rows in {elapsed_time/3600:.1f} hours")
                        self.running = False
                        break

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            # Progress update
            cycle_time = time.time() - cycle_start

            if time.time() - last_status_update > 300:  # Every 5 minutes
                progress_pct = (self.total_collected / self.target_rows) * 100
                remaining = self.target_rows - self.total_collected
                elapsed_hours = elapsed_time / 3600
                remaining_hours = (self.max_runtime_seconds - elapsed_time) / 3600

                logging.info(f"📈 Progress: {self.total_collected}/{self.target_rows} ({progress_pct:.1f}%)")
                logging.info(f"⏱️  Time: {elapsed_hours:.1f}h / {self.max_runtime_hours}h ({remaining_hours:.1f}h left)")
                logging.info(f"📊 Remaining: {remaining} rows | Batch: {batch_collected} | Anomalies: {batch_anomalies}")
                logging.info(f"⚡ Rate: {batch_collected/cycle_time:.1f} rows/sec")

                self.save_progress()
                last_status_update = time.time()

            # Wait for next cycle
            if self.running and self.total_collected < self.target_rows:
                time.sleep(max(0, interval - cycle_time))

        final_elapsed = time.time() - self.start_time
        logging.info(f"🏁 Collection completed! Total rows: {self.total_collected}")
        logging.info(f"⏱️  Total runtime: {final_elapsed/3600:.1f} hours")
        self.save_progress()

        return self.total_collected

    def save_to_csv(self, data: Dict, filename: str):
        """Save data to CSV with headers"""
        try:
            file_exists = os.path.exists(filename)
            df = pd.DataFrame([data])
            df.to_csv(filename, mode='a', header=not file_exists, index=False)
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")

    def get_collection_stats(self, filename: str = 'comprehensive_market_data.csv'):
        """Get statistics about collected data"""
        if not os.path.exists(filename):
            return None

        try:
            df = pd.read_csv(filename)
            stats = {
                'total_rows': len(df),
                'unique_symbols': df['symbol'].nunique(),
                'anomalies': len(df[df['anomaly_flag'].notna()]),
                'session_types': df['session_type'].value_counts().to_dict(),
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                }
            }
            return stats
        except Exception as e:
            logging.error(f"Error getting stats: {e}")
            return None

def main():
    """Main execution function"""
    target_rows = 10000  # 🎯 CHANGED FROM 50000 to 10000
    max_runtime_hours = 12  # ⏰ Maximum runtime in hours

    print(f"🚀 Starting Market Data Collection")
    print(f"🎯 Target: {target_rows:,} rows")
    print(f"⏰ Max runtime: {max_runtime_hours} hours")
    print(f"📈 Symbols: 40 (Large Cap, Mid Cap, Volatile, Penny Stocks, ETFs)")
    print(f"🔍 Features: Enhanced fraud detection with anomaly flagging")
    print("=" * 60)

    collector = EnhancedMarketDataCollector(
        target_rows=target_rows, 
        max_runtime_hours=max_runtime_hours
    )

    try:
        # Start continuous collection
        rows_collected = collector.collect_continuous_data()

        # Show final stats
        stats = collector.get_collection_stats()
        if stats:
            print("\n" + "=" * 60)
            print("🏁 COLLECTION COMPLETED!")
            print(f"📊 Total rows: {stats['total_rows']:,}")
            print(f"🏢 Unique symbols: {stats['unique_symbols']}")
            print(f"🚨 Anomalies detected: {stats['anomalies']}")
            print(f"🕒 Session types: {stats['session_types']}")
            print(f"📅 Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted by user")
        collector.save_progress()
    except Exception as e:
        logging.error(f"❌ Error in main execution: {e}")
        collector.save_progress()

if __name__ == "__main__":
    main()