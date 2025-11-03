"""
PyTorch Models Module
Contains all recommendation models:
- Matrix Factorization
- Neural Collaborative Filtering (NCF)
- Deep Factorization Machine (DeepFM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# ========================================
# PYTORCH DATASET
# ========================================

class ReviewDataset(Dataset):
    """PyTorch Dataset for user-item interactions"""
    
    def __init__(self, df):
        """
        Args:
            df: DataFrame with columns: user_id, item_id, rating_normalized, is_positive
        """
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['item_id'].values)
        self.ratings = torch.FloatTensor(df['rating_normalized'].values)
        self.labels = torch.FloatTensor(df['is_positive'].values)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'rating': self.ratings[idx],
            'label': self.labels[idx]
        }


# ========================================
# MODEL 1: MATRIX FACTORIZATION
# ========================================

class MatrixFactorization(nn.Module):
    """
    Simple Matrix Factorization with Bias
    Fast and effective baseline model
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(MatrixFactorization, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product of embeddings
        dot_product = (user_emb * item_emb).sum(dim=1, keepdim=True)
        
        # Add bias terms
        prediction = (dot_product + 
                     self.user_bias(user_ids) + 
                     self.item_bias(item_ids) + 
                     self.global_bias)
        
        return torch.sigmoid(prediction.squeeze())
    
    def predict(self, user_ids, item_ids):
        """Predict ratings for user-item pairs"""
        self.eval()
        with torch.no_grad():
            if isinstance(user_ids, (list, np.ndarray)):
                user_ids = torch.LongTensor(user_ids)
                item_ids = torch.LongTensor(item_ids)
            return self.forward(user_ids, item_ids).cpu().numpy()


# ========================================
# MODEL 2: NEURAL COLLABORATIVE FILTERING (NCF)
# ========================================

class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering
    Combines Generalized Matrix Factorization (GMF) with Multi-Layer Perceptron (MLP)
    State-of-the-art collaborative filtering model
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NeuralCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # GMF (Generalized Matrix Factorization) embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.output_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
    def forward(self, user_ids, item_ids):
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.output_layer(concat)
        
        return torch.sigmoid(prediction.squeeze())
    
    def predict(self, user_ids, item_ids):
        """Predict ratings for user-item pairs"""
        self.eval()
        with torch.no_grad():
            if isinstance(user_ids, (list, np.ndarray)):
                user_ids = torch.LongTensor(user_ids)
                item_ids = torch.LongTensor(item_ids)
            return self.forward(user_ids, item_ids).cpu().numpy()


# ========================================
# MODEL 3: DEEP FACTORIZATION MACHINE (DeepFM)
# ========================================

class DeepFM(nn.Module):
    """
    Deep Factorization Machine
    Combines low-order feature interactions (FM) with high-order (Deep)
    Captures both explicit and implicit feature interactions
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64]):
        super(DeepFM, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embeddings for FM and Deep parts
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Linear weights (first-order)
        self.user_linear = nn.Embedding(num_users, 1)
        self.item_linear = nn.Embedding(num_items, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Deep layers
        deep_layers = []
        input_size = embedding_dim * 2
        
        for hidden_size in hidden_layers:
            deep_layers.append(nn.Linear(input_size, hidden_size))
            deep_layers.append(nn.BatchNorm1d(hidden_size))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(0.3))
            input_size = hidden_size
        
        deep_layers.append(nn.Linear(hidden_layers[-1], 1))
        self.deep = nn.Sequential(*deep_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.item_linear.weight)
        
    def forward(self, user_ids, item_ids):
        # First-order (linear)
        linear_part = (self.user_linear(user_ids) + 
                      self.item_linear(item_ids) + 
                      self.bias)
        
        # Second-order (FM)
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        fm_part = (user_emb * item_emb).sum(dim=1, keepdim=True)
        
        # Deep part
        deep_input = torch.cat([user_emb, item_emb], dim=-1)
        deep_part = self.deep(deep_input)
        
        # Combine all parts
        prediction = linear_part + fm_part + deep_part
        
        return torch.sigmoid(prediction.squeeze())
    
    def predict(self, user_ids, item_ids):
        """Predict ratings for user-item pairs"""
        self.eval()
        with torch.no_grad():
            if isinstance(user_ids, (list, np.ndarray)):
                user_ids = torch.LongTensor(user_ids)
                item_ids = torch.LongTensor(item_ids)
            return self.forward(user_ids, item_ids).cpu().numpy()


# ========================================
# MODEL 4: ENGAGEMENT PREDICTION MODEL
# ========================================

class EngagementPredictor(nn.Module):
    """
    Deep Neural Network for engagement prediction
    Predicts probability of user clicking/purchasing
    Uses additional features beyond user-item interactions
    """
    
    def __init__(self, input_dim, hidden_layers=[256, 128, 64, 32]):
        super(EngagementPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        logits = self.network(x)
        return torch.sigmoid(logits.squeeze())


# ========================================
# MODEL FACTORY
# ========================================

def create_model(model_type, num_users, num_items, embedding_dim=64, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'mf', 'ncf', or 'deepfm'
        num_users: number of unique users
        num_items: number of unique items
        embedding_dim: dimension of embeddings
        **kwargs: additional model-specific parameters
    
    Returns:
        PyTorch model
    """
    if model_type == 'mf':
        return MatrixFactorization(num_users, num_items, embedding_dim)
    elif model_type == 'ncf':
        return NeuralCF(num_users, num_items, embedding_dim, 
                       kwargs.get('hidden_layers', [128, 64, 32]))
    elif model_type == 'deepfm':
        return DeepFM(num_users, num_items, embedding_dim,
                     kwargs.get('hidden_layers', [128, 64]))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, model_name="Model"):
    """Print model summary"""
    print(f"\n{'='*70}")
    print(f"{model_name} Summary")
    print(f"{'='*70}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model architecture:")
    print(model)
    print(f"{'='*70}\n")


# ========================================
# TEST MODELS
# ========================================

if __name__ == "__main__":
    print("Testing PyTorch Models...")
    
    # Test parameters
    num_users = 1000
    num_items = 500
    batch_size = 32
    
    # Create dummy data
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    
    # Test Matrix Factorization
    print("\n1️⃣ Testing Matrix Factorization...")
    mf_model = MatrixFactorization(num_users, num_items, embedding_dim=64)
    mf_output = mf_model(user_ids, item_ids)
    print(f"   Output shape: {mf_output.shape}")
    print(f"   Output range: [{mf_output.min():.4f}, {mf_output.max():.4f}]")
    print(f"   Parameters: {count_parameters(mf_model):,}")
    
    # Test Neural CF
    print("\n2️⃣ Testing Neural Collaborative Filtering...")
    ncf_model = NeuralCF(num_users, num_items, embedding_dim=64)
    ncf_output = ncf_model(user_ids, item_ids)
    print(f"   Output shape: {ncf_output.shape}")
    print(f"   Output range: [{ncf_output.min():.4f}, {ncf_output.max():.4f}]")
    print(f"   Parameters: {count_parameters(ncf_model):,}")
    
    # Test DeepFM
    print("\n3️⃣ Testing Deep Factorization Machine...")
    deepfm_model = DeepFM(num_users, num_items, embedding_dim=64)
    deepfm_output = deepfm_model(user_ids, item_ids)
    print(f"   Output shape: {deepfm_output.shape}")
    print(f"   Output range: [{deepfm_output.min():.4f}, {deepfm_output.max():.4f}]")
    print(f"   Parameters: {count_parameters(deepfm_model):,}")
    
    # Test Engagement Predictor
    print("\n4️⃣ Testing Engagement Predictor...")
    engagement_model = EngagementPredictor(input_dim=20)
    dummy_features = torch.randn(batch_size, 20)
    engagement_output = engagement_model(dummy_features)
    print(f"   Output shape: {engagement_output.shape}")
    print(f"   Output range: [{engagement_output.min():.4f}, {engagement_output.max():.4f}]")
    print(f"   Parameters: {count_parameters(engagement_model):,}")
    
    print("\n✅ All models tested successfully!")