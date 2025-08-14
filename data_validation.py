"""
Data Validation Script to Detect and Fix Data Leakage Issues
Run this BEFORE training your models
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_data_leakage(data_path='comprehensive_market_data.csv'):
    """Comprehensive analysis to detect data leakage"""
    print("🔍 Analyzing Data for Potential Leakage Issues...")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create target variable
    df['is_fraud'] = (df['anomaly_flag'].notna() & 
                     (df['anomaly_flag'] != '') & 
                     (df['anomaly_flag'] != 'nan')).astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print()
    
    # 1. CHECK FOR OBVIOUS LEAKAGE
    print("🚨 CHECKING FOR DATA LEAKAGE:")
    print("-" * 30)
    
    # Check correlation with target
    suspect_features = []
    
    # Features that could cause leakage
    leakage_candidates = [
        'fraud_probability', 'anomaly_flag', 'anomaly_details'
    ]
    
    for feature in leakage_candidates:
        if feature in df.columns:
            if feature == 'fraud_probability':
                correlation = df[feature].corr(df['is_fraud'])
                print(f"❌ CRITICAL: {feature} correlation with target: {correlation:.4f}")
                if correlation > 0.8:
                    suspect_features.append(feature)
            else:
                # For categorical features, check overlap
                fraud_mask = df['is_fraud'] == 1
                non_fraud_mask = df['is_fraud'] == 0
                
                fraud_values = set(df[fraud_mask][feature].dropna().unique())
                non_fraud_values = set(df[non_fraud_mask][feature].dropna().unique())
                
                if len(fraud_values & non_fraud_values) == 0:
                    print(f"❌ CRITICAL: {feature} perfectly separates fraud/non-fraud")
                    suspect_features.append(feature)
    
    # 2. CHECK MUTUAL INFORMATION
    print("\n📊 MUTUAL INFORMATION ANALYSIS:")
    print("-" * 30)
    
    numeric_features = [
        'price', 'volume', 'high', 'low', 'open',
        'volume_5d_avg', 'volume_ratio', 'price_volatility', 
        'price_spread', 'price_position', 'hour', 'minute',
        'is_opening_hour', 'is_closing_hour', 'is_pre_market',
        'fraud_probability'  # Check this specifically
    ]
    
    mi_scores = {}
    for feature in numeric_features:
        if feature in df.columns:
            # Handle missing values
            clean_data = df[[feature, 'is_fraud']].dropna()
            if len(clean_data) > 0:
                mi_score = mutual_info_score(clean_data[feature], clean_data['is_fraud'])
                mi_scores[feature] = mi_score
                
                status = "❌ SUSPICIOUS" if mi_score > 0.5 else "✅ OK"
                print(f"{status} {feature}: {mi_score:.4f}")
    
    # 3. TEMPORAL ANALYSIS
    print("\n⏰ TEMPORAL ANALYSIS:")
    print("-" * 30)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Check if fraud is clustered in time
        df['time_group'] = pd.cut(range(len(df)), bins=10, labels=False)
        fraud_by_time = df.groupby('time_group')['is_fraud'].mean()
        
        print("Fraud rate by time period:")
        for i, rate in enumerate(fraud_by_time):
            status = "⚠️ HIGH" if rate > 0.6 else "✅ OK"
            print(f"  Period {i}: {rate:.3f} {status}")
    
    # 4. FEATURE ENGINEERING VALIDATION
    print("\n🔧 FEATURE ENGINEERING VALIDATION:")
    print("-" * 30)
    
    # Check if engineered features are too perfect
    engineered_features = ['volume_ratio', 'price_volatility', 'price_spread', 'price_position']
    
    for feature in engineered_features:
        if feature in df.columns:
            fraud_mean = df[df['is_fraud'] == 1][feature].mean()
            normal_mean = df[df['is_fraud'] == 0][feature].mean()
            
            if abs(fraud_mean - normal_mean) / (fraud_mean + normal_mean + 1e-8) > 0.5:
                print(f"⚠️ {feature}: Large difference between fraud/normal")
                print(f"   Fraud mean: {fraud_mean:.4f}, Normal mean: {normal_mean:.4f}")
            else:
                print(f"✅ {feature}: Reasonable separation")
    
    # 5. RECOMMENDATIONS
    print("\n🔧 RECOMMENDATIONS:")
    print("-" * 30)
    
    if suspect_features:
        print("❌ REMOVE these features (data leakage):")
        for feature in suspect_features:
            print(f"   - {feature}")
        print()
    
    print("✅ SAFE features to use:")
    safe_features = [
        'price', 'volume', 'high', 'low', 'open',
        'volume_5d_avg', 'volume_ratio', 'price_volatility', 
        'price_spread', 'price_position', 'hour', 'minute',
        'is_opening_hour', 'is_closing_hour', 'is_pre_market'
    ]
    
    for feature in safe_features:
        if feature in df.columns and feature not in suspect_features:
            print(f"   ✓ {feature}")
    
    print("\n🎯 EXPECTED REALISTIC PERFORMANCE:")
    print("   - Accuracy: 75-85% (not 100%!)")
    print("   - F1-Score: 70-80%")
    print("   - Precision: 65-85%")
    print("   - Recall: 70-85%")
    
    return suspect_features

def create_clean_dataset(input_path='comprehensive_market_data.csv', 
                        output_path='clean_market_data.csv'):
    """Create a clean dataset without data leakage"""
    print("\n🧹 CREATING CLEAN DATASET...")
    print("-" * 30)
    
    df = pd.read_csv(input_path)
    
    # Create target
    df['is_fraud'] = (df['anomaly_flag'].notna() & 
                     (df['anomaly_flag'] != '') & 
                     (df['anomaly_flag'] != 'nan')).astype(int)
    
    # Remove leakage features
    leakage_features = ['fraud_probability', 'anomaly_flag', 'anomaly_details']
    features_to_remove = [f for f in leakage_features if f in df.columns]
    
    if features_to_remove:
        print(f"Removing features: {features_to_remove}")
        df = df.drop(columns=features_to_remove)
    
    # Keep only safe features + target
    safe_features = [
        'timestamp', 'symbol', 'price', 'volume', 'high', 'low', 'open',
        'session_type', 'volume_5d_avg', 'volume_ratio', 'price_volatility', 
        'price_spread', 'price_position', 'hour', 'minute',
        'is_opening_hour', 'is_closing_hour', 'is_pre_market', 'is_fraud'
    ]
    
    # Keep only existing columns
    existing_safe_features = [f for f in safe_features if f in df.columns]
    clean_df = df[existing_safe_features].copy()
    
    # Save clean dataset
    clean_df.to_csv(output_path, index=False)
    
    print(f"✅ Clean dataset saved to: {output_path}")
    print(f"   Original features: {len(df.columns)}")
    print(f"   Clean features: {len(clean_df.columns)}")
    print(f"   Rows: {len(clean_df)}")
    
    return clean_df

if __name__ == "__main__":
    # Run analysis
    suspect_features = analyze_data_leakage()
    
    # Create clean dataset
    clean_df = create_clean_dataset()
    
    print("\n" + "="*50)
    print("🎯 NEXT STEPS:")
    print("1. Use 'clean_market_data.csv' for training")
    print("2. Update fraud_detection_training.py with fixed code")
    print("3. Expect realistic performance (75-85% accuracy)")
    print("4. Use temporal validation splits")
    print("5. Add proper regularization to models")