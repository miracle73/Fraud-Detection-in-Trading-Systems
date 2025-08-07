import pandas as pd
import numpy as np
import pickle
import joblib
import json
import time
import logging
import warnings
from datetime import datetime
import os
import signal
import sys

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection_training.log'),
        logging.StreamHandler()
    ]
)

class ComprehensiveFraudDetectionSystem:
    def __init__(self, data_path='comprehensive_market_data.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.start_time = time.time()
        self.running = True
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logging.info("🚀 Initializing Comprehensive Fraud Detection Training System")
        
    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        elapsed_time = time.time() - self.start_time
        logging.info("Received shutdown signal. Saving progress...")
        logging.info(f"Total training time: {elapsed_time/3600:.2f} hours")
        self.running = False
        self.save_all_progress()
        sys.exit(0)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the trading data"""
        logging.info("📊 Loading and preprocessing data...")
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            logging.info(f"✅ Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
            
            # Basic data info
            logging.info(f"Data shape: {self.df.shape}")
            logging.info(f"Missing values: {self.df.isnull().sum().sum()}")
            
            # Create binary target variable from anomaly_flag
            self.df['is_fraud'] = (self.df['anomaly_flag'].notna() & 
                                  (self.df['anomaly_flag'] != '') & 
                                  (self.df['anomaly_flag'] != 'nan')).astype(int)
            
            # Feature engineering
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            self.df['month'] = self.df['timestamp'].dt.month
            
            # Encode categorical variables
            categorical_columns = ['symbol', 'session_type']
            for col in categorical_columns:
                if col in self.df.columns:
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            
            # Select features for training
            feature_columns = [
                'price', 'volume', 'high', 'low', 'open',
                'volume_5d_avg', 'volume_ratio', 'price_volatility', 
                'price_spread', 'price_position', 'hour', 'minute',
                'is_opening_hour', 'is_closing_hour', 'is_pre_market',
                'fraud_probability', 'day_of_week', 'month',
                'symbol_encoded', 'session_type_encoded'
            ]
            
            # Remove columns that don't exist
            available_features = [col for col in feature_columns if col in self.df.columns]
            logging.info(f"Available features: {len(available_features)}")
            
            # Prepare X and y
            self.X = self.df[available_features].fillna(0)
            self.y = self.df['is_fraud']
            
            # Log class distribution
            fraud_count = self.y.sum()
            fraud_percentage = (fraud_count / len(self.y)) * 100
            logging.info(f"🚨 Fraud cases: {fraud_count} ({fraud_percentage:.2f}%)")
            logging.info(f"✅ Normal cases: {len(self.y) - fraud_count} ({100-fraud_percentage:.2f}%)")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            return False
    
    def split_and_scale_data(self):
        """Split data and apply scaling"""
        logging.info("🔄 Splitting and scaling data...")
        
        try:
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            logging.info(f"Training set: {self.X_train.shape[0]} samples")
            logging.info(f"Test set: {self.X_test.shape[0]} samples")
            logging.info(f"Features: {self.X_train.shape[1]}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error splitting data: {e}")
            return False
    
    def initialize_models(self):
        """Initialize all models for training"""
        logging.info("🤖 Initializing ML models...")
        
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'use_scaling': False
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'use_scaling': False
            },
            'svm': {
                'model': SVC(
                    kernel='rbf',
                    C=1.0,
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ),
                'use_scaling': True
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'use_scaling': True
            },
            'neural_network': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                ),
                'use_scaling': True
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                ),
                'use_scaling': False
            },
            'adaboost': {
                'model': AdaBoostClassifier(
                    n_estimators=100,
                    random_state=42
                ),
                'use_scaling': False
            }
        }
        
        logging.info(f"✅ Initialized {len(self.model_configs)} models")
        
    def train_individual_models(self):
        """Train all individual models"""
        logging.info("🎯 Starting individual model training...")
        
        for model_name, config in self.model_configs.items():
            if not self.running:
                break
                
            logging.info(f"🔄 Training {model_name.upper()}...")
            start_time = time.time()
            
            try:
                model = config['model']
                use_scaling = config['use_scaling']
                
                # Select appropriate data
                if use_scaling:
                    X_train_data = self.X_train_scaled
                    X_test_data = self.X_test_scaled
                else:
                    X_train_data = self.X_train
                    X_test_data = self.X_test
                
                # Handle imbalanced data with SMOTE for some models
                if model_name in ['svm', 'logistic_regression', 'neural_network']:
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_data, self.y_train)
                    model.fit(X_train_balanced, y_train_balanced)
                else:
                    model.fit(X_train_data, self.y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_data)
                y_pred_proba = model.predict_proba(X_test_data)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
                
                # Store results
                self.models[model_name] = model
                self.results[model_name] = {
                    'metrics': metrics,
                    'training_time': time.time() - start_time,
                    'model_params': model.get_params()
                }
                
                training_time = time.time() - start_time
                logging.info(f"✅ {model_name.upper()} completed - Accuracy: {metrics['accuracy']:.4f}, "
                           f"F1-Score: {metrics['f1_score']:.4f}, Time: {training_time:.2f}s")
                
            except Exception as e:
                logging.error(f"❌ Error training {model_name}: {e}")
                continue
        
        logging.info("✅ Individual model training completed")
    
    def train_ensemble_models(self):
        """Train ensemble models"""
        logging.info("🎪 Training ensemble models...")
        
        try:
            # Get best performing models for ensemble
            best_models = []
            for name, results in self.results.items():
                if results['metrics']['f1_score'] > 0.7:  # Only include good models
                    model_config = self.model_configs[name]
                    if model_config['use_scaling']:
                        model_data = (name, self.models[name], True)
                    else:
                        model_data = (name, self.models[name], False)
                    best_models.append(model_data)
            
            if len(best_models) >= 2:
                # Create voting classifier
                estimators = [(name, model) for name, model, _ in best_models[:5]]  # Top 5 models
                
                voting_clf = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
                
                # Train ensemble (using original data, scaling handled internally)
                start_time = time.time()
                voting_clf.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_ensemble = voting_clf.predict(self.X_test)
                y_pred_proba_ensemble = voting_clf.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                ensemble_metrics = self.calculate_metrics(self.y_test, y_pred_ensemble, y_pred_proba_ensemble)
                
                # Store ensemble results
                self.models['ensemble_voting'] = voting_clf
                self.results['ensemble_voting'] = {
                    'metrics': ensemble_metrics,
                    'training_time': time.time() - start_time,
                    'base_models': [name for name, _, _ in best_models]
                }
                
                logging.info(f"✅ Ensemble model completed - Accuracy: {ensemble_metrics['accuracy']:.4f}, "
                           f"F1-Score: {ensemble_metrics['f1_score']:.4f}")
            else:
                logging.warning("⚠️ Not enough good models for ensemble (need at least 2)")
                
        except Exception as e:
            logging.error(f"❌ Error training ensemble: {e}")
    
    def train_unsupervised_models(self):
        """Train unsupervised anomaly detection models"""
        logging.info("🔍 Training unsupervised anomaly detection models...")
        
        unsupervised_models = {
            'isolation_forest': IsolationForest(
                contamination=0.375,  # Based on your fraud rate
                random_state=42,
                n_jobs=-1
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=0.375,
                novelty=True,
                n_jobs=-1
            )
        }
        
        for model_name, model in unsupervised_models.items():
            if not self.running:
                break
                
            logging.info(f"🔄 Training {model_name.upper()}...")
            start_time = time.time()
            
            try:
                # Train on scaled data
                model.fit(self.X_train_scaled)
                
                # Make predictions (1 = normal, -1 = anomaly, convert to 0/1)
                y_pred_unsupervised = model.predict(self.X_test_scaled)
                y_pred_binary = (y_pred_unsupervised == -1).astype(int)
                
                # Calculate metrics
                metrics = self.calculate_metrics(self.y_test, y_pred_binary, None)
                
                # Store results
                self.models[model_name] = model
                self.results[model_name] = {
                    'metrics': metrics,
                    'training_time': time.time() - start_time,
                    'model_type': 'unsupervised'
                }
                
                logging.info(f"✅ {model_name.upper()} completed - Accuracy: {metrics['accuracy']:.4f}, "
                           f"F1-Score: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logging.error(f"❌ Error training {model_name}: {e}")
                continue
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            
        return metrics
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on best models"""
        logging.info("⚙️ Starting hyperparameter tuning...")
        
        # Define parameter grids for top models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        # Get top 2 models for tuning
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['metrics']['f1_score'], 
                               reverse=True)
        
        for model_name, _ in sorted_results[:2]:
            if model_name not in param_grids:
                continue
                
            if not self.running:
                break
                
            logging.info(f"🎯 Tuning {model_name.upper()}...")
            
            try:
                base_model = self.model_configs[model_name]['model']
                use_scaling = self.model_configs[model_name]['use_scaling']
                
                X_train_data = self.X_train_scaled if use_scaling else self.X_train
                
                grid_search = GridSearchCV(
                    base_model,
                    param_grids[model_name],
                    cv=3,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train_data, self.y_train)
                
                # Store tuned model
                tuned_model_name = f"{model_name}_tuned"
                self.models[tuned_model_name] = grid_search.best_estimator_
                
                # Test tuned model
                X_test_data = self.X_test_scaled if use_scaling else self.X_test
                y_pred_tuned = grid_search.predict(X_test_data)
                y_pred_proba_tuned = grid_search.predict_proba(X_test_data)[:, 1]
                
                tuned_metrics = self.calculate_metrics(self.y_test, y_pred_tuned, y_pred_proba_tuned)
                
                self.results[tuned_model_name] = {
                    'metrics': tuned_metrics,
                    'best_params': grid_search.best_params_,
                    'model_type': 'tuned'
                }
                
                logging.info(f"✅ {tuned_model_name} - Best F1: {tuned_metrics['f1_score']:.4f}")
                
            except Exception as e:
                logging.error(f"❌ Error tuning {model_name}: {e}")
                continue
    
    def generate_comprehensive_report(self):
        """Generate comprehensive training report"""
        logging.info("📊 Generating comprehensive report...")
        
        # Sort results by F1 score
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['metrics']['f1_score'], 
                               reverse=True)
        
        report = {
            'training_summary': {
                'total_training_time': time.time() - self.start_time,
                'models_trained': len(self.results),
                'dataset_size': len(self.df),
                'fraud_cases': self.y.sum(),
                'normal_cases': len(self.y) - self.y.sum(),
                'features_used': list(self.X.columns),
                'timestamp': datetime.now().isoformat()
            },
            'model_rankings': [],
            'detailed_results': self.results,
            'best_model_info': {}
        }
        
        # Create model rankings
        for i, (model_name, results) in enumerate(sorted_results, 1):
            ranking = {
                'rank': i,
                'model_name': model_name,
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'f1_score': results['metrics']['f1_score'],
                'roc_auc': results['metrics']['roc_auc'],
                'training_time': results.get('training_time', 0)
            }
            report['model_rankings'].append(ranking)
        
        # Best model information
        if sorted_results:
            best_model_name, best_results = sorted_results[0]
            report['best_model_info'] = {
                'name': best_model_name,
                'metrics': best_results['metrics'],
                'training_time': best_results.get('training_time', 0)
            }
        
        # Save report
        with open('fraud_detection_training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logging.info("🏆 TRAINING RESULTS SUMMARY")
        logging.info("=" * 60)
        
        for ranking in report['model_rankings'][:5]:  # Top 5
            logging.info(f"{ranking['rank']}. {ranking['model_name'].upper()}")
            logging.info(f"   Accuracy: {ranking['accuracy']:.4f} | F1-Score: {ranking['f1_score']:.4f}")
            logging.info(f"   Precision: {ranking['precision']:.4f} | Recall: {ranking['recall']:.4f}")
            logging.info(f"   ROC-AUC: {ranking['roc_auc']:.4f} | Time: {ranking['training_time']:.2f}s")
            logging.info("-" * 40)
        
        total_time = time.time() - self.start_time
        logging.info(f"🎯 BEST MODEL: {report['best_model_info']['name'].upper()}")
        logging.info(f"⏱️  TOTAL TRAINING TIME: {total_time/3600:.2f} hours")
        logging.info(f"📊 MODELS TRAINED: {len(self.results)}")
        
        return report
    
    def save_all_progress(self):
        """Save all models and progress"""
        logging.info("💾 Saving all models and progress...")
        
        try:
            # Create models directory
            os.makedirs('trained_models', exist_ok=True)
            
            # Save individual models
            for model_name, model in self.models.items():
                model_path = f'trained_models/{model_name}_model.joblib'
                joblib.dump(model, model_path)
                logging.info(f"✅ Saved {model_name} to {model_path}")
            
            # Save scaler
            joblib.dump(self.scaler, 'trained_models/scaler.joblib')
            
            # Save feature columns
            with open('trained_models/feature_columns.json', 'w') as f:
                json.dump(list(self.X.columns), f)
            
            # Save results
            with open('trained_models/training_results.json', 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logging.info("✅ All models and progress saved successfully")
            
        except Exception as e:
            logging.error(f"❌ Error saving models: {e}")
    
    def run_complete_training_pipeline(self):
        """Run the complete training pipeline"""
        logging.info("🚀 Starting Complete Fraud Detection Training Pipeline")
        logging.info("=" * 80)
        
        pipeline_steps = [
            ("Data Loading & Preprocessing", self.load_and_preprocess_data),
            ("Data Splitting & Scaling", self.split_and_scale_data),
            ("Model Initialization", lambda: self.initialize_models() or True),
            ("Individual Model Training", lambda: self.train_individual_models() or True),
            ("Ensemble Model Training", lambda: self.train_ensemble_models() or True),
            ("Unsupervised Model Training", lambda: self.train_unsupervised_models() or True),
            ("Hyperparameter Tuning", lambda: self.hyperparameter_tuning() or True),
            ("Report Generation", lambda: self.generate_comprehensive_report() or True),
            ("Save Progress", lambda: self.save_all_progress() or True)
        ]
        
        for step_name, step_function in pipeline_steps:
            if not self.running:
                break
                
            logging.info(f"🔄 {step_name}...")
            
            try:
                success = step_function()
                if success is False:
                    logging.error(f"❌ {step_name} failed")
                    break
                logging.info(f"✅ {step_name} completed")
                
            except Exception as e:
                logging.error(f"❌ Error in {step_name}: {e}")
                break
        
        total_time = time.time() - self.start_time
        logging.info("🎉 TRAINING PIPELINE COMPLETED!")
        logging.info(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        logging.info(f"📊 Check 'fraud_detection_training_report.json' for detailed results")
        logging.info(f"💾 Models saved in 'trained_models/' directory")

def main():
    """Main execution function"""
    logging.info("🎯 Initializing Fraud Detection Training System")
    
    # Check if data file exists
    if not os.path.exists('comprehensive_market_data.csv'):
        logging.error("❌ Data file 'comprehensive_market_data.csv' not found!")
        logging.info("Please ensure the CSV file is in the same directory")
        return
    
    # Initialize and run training system
    fraud_detector = ComprehensiveFraudDetectionSystem()
    
    try:
        fraud_detector.run_complete_training_pipeline()
        logging.info("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("⚠️ Training interrupted by user")
        fraud_detector.save_all_progress()
        
    except Exception as e:
        logging.error(f"❌ Training failed with error: {e}")
        fraud_detector.save_all_progress()

if __name__ == "__main__":
    main()