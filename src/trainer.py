"""
Trainer Module
Handles model training, validation, and real-time updates
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import time

class Trainer:
    """Universal trainer for all recommendation models with real-time update capabilities"""
    
    def __init__(self, model, device='cpu', model_save_path=None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
            model_save_path: Path to save best model
        """
        self.model = model.to(device)
        self.device = device
        self.model_save_path = model_save_path
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        
        self.best_auc = 0
        self.best_epoch = 0
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Move data to device
            users = batch['user'].to(self.device)
            items = batch['item'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            predictions = self.model(users, items)
            loss = criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                users = batch['user'].to(self.device)
                items = batch['item'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                predictions = self.model(users, items)
                loss = criterion(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / num_batches
        auc = roc_auc_score(all_labels, all_predictions)
        
        # Binary predictions
        binary_preds = (all_predictions > 0.5).astype(int)
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        
        metrics = {
            'loss': avg_loss,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def fit(self, train_loader, val_loader, epochs=20, lr=0.001, weight_decay=1e-5, patience=5):
        """
        Train the model with early stopping
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            
        Returns:
            best_auc: Best validation AUC achieved
        """
        print("\n" + "=" * 70)
        print("🚀 TRAINING STARTED")
        print("=" * 70)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        criterion = nn.BCELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
        
        # Early stopping
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader, criterion)
            self.val_losses.append(val_metrics['loss'])
            self.val_aucs.append(val_metrics['auc'])
            self.val_precisions.append(val_metrics['precision'])
            self.val_recalls.append(val_metrics['recall'])
            self.val_f1s.append(val_metrics['f1'])
            
            # Update learning rate
            scheduler.step(val_metrics['auc'])
            
            # Print metrics
            epoch_time = time.time() - epoch_start_time
            print(f"\nResults (Time: {epoch_time:.1f}s):")
            print(f"  Train Loss:     {train_loss:.4f}")
            print(f"  Val Loss:       {val_metrics['loss']:.4f}")
            print(f"  Val AUC:        {val_metrics['auc']:.4f}")
            print(f"  Val Precision:  {val_metrics['precision']:.4f}")
            print(f"  Val Recall:     {val_metrics['recall']:.4f}")
            print(f"  Val F1:         {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['auc'] > self.best_auc:
                self.best_auc = val_metrics['auc']
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                if self.model_save_path:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"  ✅ Best model saved! (AUC: {self.best_auc:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                    break
        
        print("\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE!")
        print(f"   Best Validation AUC: {self.best_auc:.4f} at epoch {self.best_epoch}")
        print("=" * 70)
        
        return self.best_auc
    
    def update_with_new_interaction(self, user_id, item_id, label, lr=0.0001):
        """
        Real-time update: Fine-tune model with new user interaction
        
        Args:
            user_id: User ID
            item_id: Item ID
            label: Interaction label (0 or 1)
            lr: Learning rate for update
            
        Returns:
            loss: Loss value after update
        """
        self.model.train()
        
        # Convert to tensors
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        item_tensor = torch.LongTensor([item_id]).to(self.device)
        label_tensor = torch.FloatTensor([label]).to(self.device)
        
        # Single gradient update
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        optimizer.zero_grad()
        prediction = self.model(user_tensor, item_tensor)
        loss = criterion(prediction, label_tensor)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def recommend_for_user(self, user_id, all_item_ids, top_k=10):
        """
        Generate top-K recommendations for a user
        
        Args:
            user_id: User ID
            all_item_ids: List of all item IDs to rank
            top_k: Number of recommendations to return
        
        Returns:
            List of (item_id, score) tuples
        """
        self.model.eval()
        
        # Create user-item pairs
        user_ids = torch.LongTensor([user_id] * len(all_item_ids)).to(self.device)
        item_ids = torch.LongTensor(all_item_ids).to(self.device)
        
        # Predict scores
        with torch.no_grad():
            scores = self.model(user_ids, item_ids).cpu().numpy()
        
        # Get top-K items
        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommendations = [(all_item_ids[i], float(scores[i])) for i in top_indices]
        
        return recommendations
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 1].plot(epochs, self.val_aucs, 'g-', linewidth=2)
        axes[0, 1].set_title('Validation AUC', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=max(self.val_aucs), color='r', linestyle='--', alpha=0.5, 
                          label=f'Best: {max(self.val_aucs):.4f}')
        axes[0, 1].legend()
        
        # Precision and Recall
        axes[1, 0].plot(epochs, self.val_precisions, 'b-', label='Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.val_recalls, 'r-', label='Recall', linewidth=2)
        axes[1, 0].set_title('Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 1].plot(epochs, self.val_f1s, 'purple', linewidth=2)
        axes[1, 1].set_title('F1 Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def load_best_model(self):
        """Load the best saved model"""
        if self.model_save_path and Path(self.model_save_path).exists():
            self.model.load_state_dict(
                torch.load(self.model_save_path, map_location=self.device)
            )
            print(f"✅ Loaded best model from: {self.model_save_path}")
        else:
            print("⚠️ No saved model found")
    
    def get_training_summary(self):
        """Get summary of training"""
        summary = {
            'total_epochs': len(self.train_losses),
            'best_val_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_precision': self.val_precisions[-1] if self.val_precisions else None,
            'final_recall': self.val_recalls[-1] if self.val_recalls else None,
            'final_f1': self.val_f1s[-1] if self.val_f1s else None
        }
        return summary


# ========================================
# TEST TRAINER
# ========================================

if __name__ == "__main__":
    from src.models import MatrixFactorization, ReviewDataset
    from src.config import Config
    import pandas as pd
    
    print("Testing Trainer...")
    
    # Create dummy data
    print("\nCreating dummy data...")
    num_samples = 1000
    dummy_data = pd.DataFrame({
        'user_id': np.random.randint(0, 100, num_samples),
        'item_id': np.random.randint(0, 50, num_samples),
        'rating_normalized': np.random.rand(num_samples),
        'is_positive': np.random.randint(0, 2, num_samples)
    })
    
    # Create datasets and loaders
    train_data = dummy_data.iloc[:800]
    val_data = dummy_data.iloc[800:]
    
    train_dataset = ReviewDataset(train_data)
    val_dataset = ReviewDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    print("\nCreating model...")
    model = MatrixFactorization(num_users=100, num_items=50, embedding_dim=32)
    
    # Create trainer
    device = Config.get_device()
    trainer = Trainer(model, device=device, model_save_path='test_model.pt')
    
    # Train
    print("\nTraining model (3 epochs for testing)...")
    trainer.fit(train_loader, val_loader, epochs=3, lr=0.01, patience=10)
    
    # Plot
    print("\nGenerating training plots...")
    trainer.plot_training_history()
    
    # Test real-time update
    print("\nTesting real-time update...")
    loss = trainer.update_with_new_interaction(user_id=5, item_id=10, label=1)
    print(f"  Update loss: {loss:.4f}")
    
    # Get recommendations
    print("\nGenerating recommendations...")
    recommendations = trainer.recommend_for_user(user_id=0, all_item_ids=list(range(50)), top_k=5)
    print(f"Top 5 recommendations for user 0:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: {score:.4f}")
    
    # Summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Trainer test complete!")