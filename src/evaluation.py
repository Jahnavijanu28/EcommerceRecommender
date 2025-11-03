"""
Evaluation Module
Comprehensive model evaluation with multiple metrics and visualizations
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from pathlib import Path

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model to evaluate
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary of metrics
        """
        print("\n" + "=" * 70)
        print("📊 MODEL EVALUATION")
        print("=" * 70)
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        # Get predictions
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                users = batch['user'].to(self.device)
                items = batch['item'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(users, items)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        binary_preds = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'AUC': roc_auc_score(labels, predictions),
            'Accuracy': accuracy_score(labels, binary_preds),
            'Precision': precision_score(labels, binary_preds, zero_division=0),
            'Recall': recall_score(labels, binary_preds, zero_division=0),
            'F1 Score': f1_score(labels, binary_preds, zero_division=0)
        }
        
        # Calculate Precision@K and Recall@K
        k_values = [5, 10, 20]
        for k in k_values:
            prec_k, rec_k = self._calculate_precision_recall_at_k(predictions, labels, k)
            metrics[f'Precision@{k}'] = prec_k
            metrics[f'Recall@{k}'] = rec_k
        
        # Calculate MAP (Mean Average Precision)
        metrics['MAP'] = self._calculate_map(predictions, labels)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels,
            'binary_predictions': binary_preds
        }
        
        # Print results
        print("\n📈 Test Set Results:")
        print("-" * 70)
        for metric, value in metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        print("-" * 70)
        
        return metrics
    
    def _calculate_precision_recall_at_k(self, predictions, labels, k):
        """Calculate Precision@K and Recall@K"""
        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-k:]
        
        # Check how many are actually positive
        relevant = labels[top_k_indices].sum()
        total_relevant = labels.sum()
        
        precision_k = relevant / k if k > 0 else 0
        recall_k = relevant / total_relevant if total_relevant > 0 else 0
        
        return precision_k, recall_k
    
    def _calculate_map(self, predictions, labels):
        """Calculate Mean Average Precision"""
        # Sort by prediction scores
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Calculate average precision
        precisions = []
        num_relevant = 0
        
        for i, label in enumerate(sorted_labels):
            if label == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precisions.append(precision_at_i)
        
        if len(precisions) == 0:
            return 0.0
        
        return np.mean(precisions)
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        if 'binary_predictions' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        cm = confusion_matrix(self.results['labels'], self.results['binary_predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        return plt.gcf()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        if 'predictions' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        fpr, tpr, thresholds = roc_curve(self.results['labels'], self.results['predictions'])
        auc = self.results['metrics']['AUC']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curve saved to: {save_path}")
        
        plt.show()
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot Precision-Recall curve"""
        if 'predictions' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        precision, recall, thresholds = precision_recall_curve(
            self.results['labels'], 
            self.results['predictions']
        )
        ap = average_precision_score(self.results['labels'], self.results['predictions'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ PR curve saved to: {save_path}")
        
        plt.show()
        
        return plt.gcf()
    
    def plot_prediction_distribution(self, save_path=None):
        """Plot distribution of predictions"""
        if 'predictions' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.results['predictions'], bins=50, color='skyblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0.5, color='r', linestyle='--', linewidth=2, 
                       label='Threshold (0.5)')
        axes[0].set_title('Prediction Score Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Prediction Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot by true label
        pred_df = pd.DataFrame({
            'score': self.results['predictions'],
            'label': ['Positive' if l == 1 else 'Negative' for l in self.results['labels']]
        })
        
        sns.boxplot(data=pred_df, x='label', y='score', ax=axes[1])
        axes[1].set_title('Prediction Scores by True Label', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('True Label')
        axes[1].set_ylabel('Prediction Score')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Prediction distribution saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        if 'binary_predictions' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        print("\n" + "=" * 70)
        print("📋 CLASSIFICATION REPORT")
        print("=" * 70)
        
        report = classification_report(
            self.results['labels'],
            self.results['binary_predictions'],
            target_names=['Negative', 'Positive'],
            digits=4
        )
        
        print(report)
        
        return report
    
    def plot_all_metrics(self, save_path=None):
        """Plot all evaluation metrics in one figure"""
        if 'metrics' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        # Prepare data
        main_metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        main_values = [self.results['metrics'][m] for m in main_metrics]
        
        prec_k_metrics = ['Precision@5', 'Precision@10', 'Precision@20']
        prec_k_values = [self.results['metrics'][m] for m in prec_k_metrics]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Main metrics bar chart
        bars1 = axes[0].bar(main_metrics, main_values, color='skyblue', edgecolor='black')
        axes[0].set_title('Main Evaluation Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, main_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Precision@K line plot
        k_values = [5, 10, 20]
        axes[1].plot(k_values, prec_k_values, 'o-', linewidth=2, 
                    markersize=8, color='coral')
        axes[1].set_title('Precision@K', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('K (Top-K Recommendations)')
        axes[1].set_ylabel('Precision')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(k_values)
        
        # Add value labels
        for k, p in zip(k_values, prec_k_values):
            axes[1].text(k, p + 0.03, f'{p:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ All metrics plot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_models(self, model_results_dict, save_path=None):
        """
        Compare multiple models
        
        Args:
            model_results_dict: Dict of {model_name: metrics_dict}
            save_path: Path to save comparison plot
        """
        print("\n" + "=" * 70)
        print("📊 MODEL COMPARISON")
        print("=" * 70)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(model_results_dict).T
        
        print("\n", comparison_df.to_string())
        
        # Plot comparison
        metrics_to_plot = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            ax = comparison_df[available_metrics].plot(
                kind='bar', 
                figsize=(12, 6),
                rot=0,
                width=0.8
            )
            plt.title('Model Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Model', fontsize=12)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim([0, 1])
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Comparison plot saved to: {save_path}")
            
            plt.show()
        
        return comparison_df
    
    def save_results(self, filepath):
        """Save evaluation results to CSV file"""
        if 'metrics' not in self.results:
            print("⚠️ Run evaluate() first!")
            return
        
        results_df = pd.DataFrame([self.results['metrics']])
        results_df.to_csv(filepath, index=False)
        print(f"✅ Results saved to: {filepath}")
        
    def get_metrics_summary(self):
        """Get summary of all metrics"""
        if 'metrics' not in self.results:
            print("⚠️ Run evaluate() first!")
            return None
        
        return self.results['metrics']


# ========================================
# UTILITY FUNCTIONS
# ========================================

def evaluate_multiple_models(models_dict, test_loader, device='cpu', output_dir=None):
    """
    Evaluate multiple models and compare them
    
    Args:
        models_dict: Dictionary of {model_name: model}
        test_loader: DataLoader for test data
        device: 'cpu' or 'cuda'
        output_dir: Directory to save results
    
    Returns:
        DataFrame with comparison results
    """
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*70}")
        
        evaluator = ModelEvaluator(model, device=device)
        metrics = evaluator.evaluate(test_loader)
        results[model_name] = metrics
        
        # Save individual results
        if output_dir:
            output_dir = Path(output_dir)
            evaluator.plot_confusion_matrix(output_dir / f'{model_name}_confusion_matrix.png')
            evaluator.plot_roc_curve(output_dir / f'{model_name}_roc_curve.png')
            evaluator.save_results(output_dir / f'{model_name}_results.csv')
    
    # Compare all models
    evaluator = ModelEvaluator(list(models_dict.values())[0], device=device)
    comparison_df = evaluator.compare_models(results, 
                                            output_dir / 'model_comparison.png' if output_dir else None)
    
    return comparison_df


# ========================================
# TEST EVALUATOR
# ========================================

if __name__ == "__main__":
    from src.models import MatrixFactorization, ReviewDataset
    from src.config import Config
    from torch.utils.data import DataLoader
    import pandas as pd
    
    print("Testing Model Evaluator...")
    
    # Create dummy test data
    print("\nCreating dummy test data...")
    num_samples = 500
    test_data = pd.DataFrame({
        'user_id': np.random.randint(0, 100, num_samples),
        'item_id': np.random.randint(0, 50, num_samples),
        'rating_normalized': np.random.rand(num_samples),
        'is_positive': np.random.randint(0, 2, num_samples)
    })
    
    # Create dataset and loader
    test_dataset = ReviewDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create and load model
    print("\nCreating model...")
    model = MatrixFactorization(num_users=100, num_items=50, embedding_dim=32)
    
    # Evaluate
    device = Config.get_device()
    evaluator = ModelEvaluator(model, device=device)
    
    print("\nEvaluating model...")
    metrics = evaluator.evaluate(test_loader)
    
    # Generate report
    evaluator.generate_classification_report()
    
    # Plot results
    print("\nGenerating plots...")
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    evaluator.plot_precision_recall_curve()
    evaluator.plot_prediction_distribution()
    evaluator.plot_all_metrics()
    
    # Get summary
    summary = evaluator.get_metrics_summary()
    print("\nMetrics Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Evaluator test complete!")