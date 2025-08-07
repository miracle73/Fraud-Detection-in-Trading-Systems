#!/usr/bin/env python3
"""
Comprehensive Training Metrics Analyzer
Extracts and visualizes all training metrics from your fraud detection models
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class TrainingMetricsAnalyzer:
    def __init__(self, report_path='fraud_detection_training_report.json'):
        self.report_path = report_path
        self.load_report()
    
    def load_report(self):
        """Load training report"""
        try:
            with open(self.report_path, 'r') as f:
                self.report = json.load(f)
            print("✅ Training report loaded successfully")
        except FileNotFoundError:
            print(f"❌ Report file not found: {self.report_path}")
            return
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in report file")
            return
    
    def print_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("🎯 FRAUD DETECTION TRAINING METRICS SUMMARY")
        print("="*60)
        
        # Training overview
        summary = self.report.get('training_summary', {})
        print(f"\n📊 Training Overview:")
        print(f"   Total Training Time: {summary.get('total_training_time', 0):.2f} seconds")
        print(f"   Models Trained: {summary.get('models_trained', 0)}")
        print(f"   Dataset Size: {summary.get('dataset_size', 0):,} rows")
        
        # Handle fraud_cases and normal_cases as strings or numbers
        fraud_cases = summary.get('fraud_cases', 0)
        normal_cases = summary.get('normal_cases', 0)
        
        # Convert to int if string
        if isinstance(fraud_cases, str):
            fraud_cases = int(fraud_cases)
        if isinstance(normal_cases, str):
            normal_cases = int(normal_cases)
            
        print(f"   Fraud Cases: {fraud_cases:,}")
        print(f"   Normal Cases: {normal_cases:,}")
        print(f"   Features Used: {len(summary.get('features_used', []))}")
        
        # Best model info
        best_model = self.report.get('best_model_info', {})
        print(f"\n🏆 Best Model: {best_model.get('name', 'N/A').upper()}")
        if 'metrics' in best_model:
            metrics = best_model['metrics']
            print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"   Precision: {metrics.get('precision', 0):.4f}")
            print(f"   Recall: {metrics.get('recall', 0):.4f}")
            print(f"   F1-Score: {metrics.get('f1_score', 0):.4f}")
            print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    
    def print_detailed_rankings(self):
        """Print detailed model rankings"""
        print(f"\n📈 Detailed Model Rankings:")
        print("-" * 80)
        
        rankings = self.report.get('model_rankings', [])
        for rank_info in rankings:
            print(f"Rank {rank_info['rank']:2d}: {rank_info['model_name'].upper():20}")
            print(f"          Accuracy: {rank_info['accuracy']:.4f} | F1-Score: {rank_info['f1_score']:.4f}")
            print(f"          Precision: {rank_info['precision']:.4f} | Recall: {rank_info['recall']:.4f}")
            print(f"          ROC-AUC: {rank_info['roc_auc']:.4f} | Time: {rank_info['training_time']:.2f}s")
            print("-" * 80)
    
    def create_metrics_dataframe(self):
        """Create pandas DataFrame with all metrics"""
        rankings = self.report.get('model_rankings', [])
        
        df_data = []
        for rank_info in rankings:
            df_data.append({
                'rank': rank_info['rank'],
                'model_name': rank_info['model_name'],
                'accuracy': rank_info['accuracy'],
                'precision': rank_info['precision'],
                'recall': rank_info['recall'],
                'f1_score': rank_info['f1_score'],
                'roc_auc': rank_info['roc_auc'],
                'training_time': rank_info['training_time']
            })
        
        df = pd.DataFrame(df_data)
        return df
    
    def generate_metrics_csv(self, filename='training_metrics.csv'):
        """Export metrics to CSV"""
        df = self.create_metrics_dataframe()
        df.to_csv(filename, index=False)
        print(f"✅ Metrics exported to {filename}")
        return df
    
    def create_performance_visualization(self):
        """Create performance visualization"""
        try:
            df = self.create_metrics_dataframe()
            
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Fraud Detection Model Performance Metrics', fontsize=16, fontweight='bold')
            
            # 1. F1-Score comparison
            axes[0, 0].barh(df['model_name'], df['f1_score'], color='skyblue')
            axes[0, 0].set_xlabel('F1-Score')
            axes[0, 0].set_title('F1-Score by Model')
            axes[0, 0].set_xlim(0, 1.1)
            
            # 2. Accuracy vs Training Time
            axes[0, 1].scatter(df['training_time'], df['accuracy'], 
                              s=100, alpha=0.7, color='orange')
            axes[0, 1].set_xlabel('Training Time (seconds)')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy vs Training Time')
            
            # Add model names as labels
            for i, txt in enumerate(df['model_name']):
                axes[0, 1].annotate(txt, (df['training_time'].iloc[i], df['accuracy'].iloc[i]), 
                                   fontsize=8, rotation=45)
            
            # 3. Precision vs Recall
            axes[1, 0].scatter(df['recall'], df['precision'], 
                              s=100, alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision vs Recall')
            axes[1, 0].set_xlim(0, 1.1)
            axes[1, 0].set_ylim(0, 1.1)
            
            # 4. Overall Performance Heatmap
            metrics_for_heatmap = df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].T
            metrics_for_heatmap.columns = df['model_name']
            
            sns.heatmap(metrics_for_heatmap, annot=True, fmt='.3f', 
                       cmap='RdYlGn', ax=axes[1, 1], cbar_kws={'shrink': 0.8})
            axes[1, 1].set_title('Performance Metrics Heatmap')
            axes[1, 1].set_xlabel('Models')
            
            plt.tight_layout()
            plt.savefig('training_metrics_visualization.png', dpi=300, bbox_inches='tight')
            print("✅ Visualization saved as 'training_metrics_visualization.png'")
            plt.show()
            
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
    
    def compare_top_models(self, top_n=5):
        """Compare top N models"""
        print(f"\n🥇 Top {top_n} Models Comparison:")
        print("="*60)
        
        rankings = self.report.get('model_rankings', [])[:top_n]
        
        for i, model in enumerate(rankings, 1):
            print(f"{i}. {model['model_name'].upper()}")
            print(f"   🎯 F1-Score: {model['f1_score']:.4f}")
            print(f"   📊 Accuracy: {model['accuracy']:.4f}")
            print(f"   ⚡ Speed: {model['training_time']:.2f}s")
            print(f"   📈 ROC-AUC: {model['roc_auc']:.4f}")
            print()
    
    def get_model_details(self, model_name):
        """Get detailed information about a specific model"""
        detailed_results = self.report.get('detailed_results', {})
        
        if model_name in detailed_results:
            model_info = detailed_results[model_name]
            print(f"\n🔍 Detailed Analysis: {model_name.upper()}")
            print("="*50)
            
            # Metrics
            metrics = model_info.get('metrics', {})
            print("📊 Performance Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric.capitalize()}: {value:.4f}")
            
            # Training time
            training_time = model_info.get('training_time', 0)
            print(f"\n⏱️ Training Time: {training_time:.2f} seconds")
            
            # Model parameters (if available)
            if 'model_params' in model_info:
                print(f"\n⚙️ Model Parameters:")
                params = model_info['model_params']
                for param, value in list(params.items())[:10]:  # Show first 10 params
                    print(f"   {param}: {value}")
                if len(params) > 10:
                    print(f"   ... and {len(params) - 10} more parameters")
        else:
            print(f"❌ Model '{model_name}' not found in results")
    
    def run_full_analysis(self):
        """Run complete metrics analysis"""
        self.print_summary()
        self.print_detailed_rankings()
        self.compare_top_models()
        
        # Export metrics
        df = self.generate_metrics_csv()
        
        # Create visualization
        self.create_performance_visualization()
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📁 Files generated:")
        print(f"   - training_metrics.csv")
        print(f"   - training_metrics_visualization.png")

def main():
    """Main function"""
    print("🔍 Starting Training Metrics Analysis...")
    
    analyzer = TrainingMetricsAnalyzer()
    
    # Interactive menu
    while True:
        print(f"\n{'='*50}")
        print("📊 TRAINING METRICS ANALYZER")
        print("="*50)
        print("1. Full Analysis Report")
        print("2. Summary Only")
        print("3. Detailed Rankings")
        print("4. Top Models Comparison")
        print("5. Export to CSV")
        print("6. Create Visualizations")
        print("7. Analyze Specific Model")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            analyzer.run_full_analysis()
        elif choice == '2':
            analyzer.print_summary()
        elif choice == '3':
            analyzer.print_detailed_rankings()
        elif choice == '4':
            analyzer.compare_top_models()
        elif choice == '5':
            analyzer.generate_metrics_csv()
        elif choice == '6':
            analyzer.create_performance_visualization()
        elif choice == '7':
            model_name = input("Enter model name: ").strip()
            analyzer.get_model_details(model_name)
        elif choice == '8':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-8.")

if __name__ == "__main__":
    main()