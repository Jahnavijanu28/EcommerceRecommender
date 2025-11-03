"""
E-Commerce Recommendation System
Source Package Initialization

This file makes the 'src' directory a Python package,
allowing imports like: from src.config import Config
"""

# Package metadata
__version__ = '1.0.0'
__author__ = 'Your Name'
__description__ = 'E-Commerce Recommendation System with PyTorch'

# Import main components for easy access
from .config import Config
from .data_preprocessing import DataPreprocessor
from .models import (
    MatrixFactorization,
    NeuralCF,
    DeepFM,
    ReviewDataset,
    create_model,
    count_parameters
)
from .trainer import Trainer
from .evaluation import ModelEvaluator

# Define what gets imported with "from src import *"
__all__ = [
    'Config',
    'DataPreprocessor',
    'MatrixFactorization',
    'NeuralCF',
    'DeepFM',
    'ReviewDataset',
    'create_model',
    'count_parameters',
    'Trainer',
    'ModelEvaluator'
]

# Package initialization message (optional - can be removed)
import warnings
warnings.filterwarnings('ignore')

# Verify PyTorch is installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found! Please install: pip install torch")

# Print package info when imported (optional - can comment out)
def _print_package_info():
    """Print package information"""
    print(f"📦 E-Commerce Recommender Package v{__version__}")
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch {torch.__version__} available")
        print(f"✅ Device: {Config.get_device()}")
    else:
        print("⚠️  PyTorch not available")

# Uncomment the line below to see package info on import
# _print_package_info()