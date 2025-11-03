"""
E-Commerce Recommendation System - Main Execution Script
Complete ML Project with PyTorch

Usage:
    python main.py --preprocess --train --evaluate --model ncf
    python main.py --all  (runs everything)
"""

import sys
import os
from pathlib import Path

# CRITICAL FIX: Add project root and src to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import modules with fallback
try:
    from src.config import Config
    from src.data_preprocessing import DataPreprocessor
    from src.models import create_model, ReviewDataset, count_parameters
    from src.trainer import Trainer
    from src.evaluation import ModelEvaluator
except ModuleNotFoundError:
    # Fallback: try direct import from src folder
    try:
        from config import Config
        from data_preprocessing import DataPreprocessor
        from models import create_model, ReviewDataset, count_parameters
        from trainer import Trainer
        from evaluation import ModelEvaluator
    except ModuleNotFoundError as e:
        print(f"❌ ERROR: Could not import modules: {e}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Project root: {project_root}")
        print(f"   Python path: {sys.path}")
        sys.exit(1)

# ========================================
# MAIN PIPELINE FUNCTIONS
# ========================================

def run_preprocessing(data_path):
    """Step 1: Data Preprocessing"""
    print("\n" + "=" * 70)
    print("🔄 PIPELINE STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Load and clean data
    preprocessor.load_data()
    preprocessor.clean_data()
    
    # Create features
    preprocessor.create_features()
    
    # Split data
    preprocessor.split_data(
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED
    )
    
    # Save processed data
    preprocessor.save_processed_data(Config.PROCESSED_DATA_DIR)
    
    # Visualize
    preprocessor.visualize_data(Config.OUTPUTS_DIR / 'data_exploration.png')
    
    # Print stats
    stats = preprocessor.get_stats()
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return preprocessor


def run_training(model_type='ncf', epochs=None, batch_size=None, lr=None):
    """Step 2: Model Training"""
    print("\n" + "=" * 70)
    print(f"🚀 PIPELINE STEP 2: TRAINING {model_type.upper()} MODEL")
    print("=" * 70)
    
    # Load processed data
    print("\n📂 Loading processed data...")
    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'train_data.csv')
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'val_data.csv')
    
    print(f"  Train set: {len(train_df):,} samples")
    print(f"  Val set:   {len(val_df):,} samples")
    
    # Get dimensions
    num_users = max(train_df['user_id'].max(), val_df['user_id'].max()) + 1
    num_items = max(train_df['item_id'].max(), val_df['item_id'].max()) + 1
    
    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = ReviewDataset(train_df)
    val_dataset = ReviewDataset(val_df)
    
    # Use config values or provided values
    batch_size = batch_size or Config.BATCH_SIZE
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\n🧠 Creating {model_type.upper()} model...")
    model = create_model(
        model_type=model_type,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=Config.EMBEDDING_DIM
    )
    
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Create trainer
    device = Config.get_device()
    print(f"  Device: {device}")
    
    model_save_path = Config.MODELS_DIR / f'best_{model_type}_model.pt'
    trainer = Trainer(model, device=device, model_save_path=model_save_path)
    
    # Train
    epochs = epochs or Config.EPOCHS
    lr = lr or Config.LEARNING_RATE
    
    best_auc = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        patience=5
    )
    
    # Plot training history
    trainer.plot_training_history(Config.OUTPUTS_DIR / f'{model_type}_training_history.png')
    
    # Save summary
    summary = trainer.get_training_summary()
    print("\n📊 Training Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return trainer, model


def run_evaluation(model, model_name='model'):
    """Step 3: Model Evaluation"""
    print("\n" + "=" * 70)
    print(f"📊 PIPELINE STEP 3: EVALUATING {model_name.upper()}")
    print("=" * 70)
    
    # Load test data
    print("\n📂 Loading test data...")
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'test_data.csv')
    print(f"  Test set: {len(test_df):,} samples")
    
    # Create test loader
    test_dataset = ReviewDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Evaluate
    device = Config.get_device()
    evaluator = ModelEvaluator(model, device=device)
    
    metrics = evaluator.evaluate(test_loader)
    
    # Generate detailed report
    evaluator.generate_classification_report()
    
    # Create visualizations
    print("\n📈 Creating evaluation plots...")
    evaluator.plot_confusion_matrix(Config.OUTPUTS_DIR / f'{model_name}_confusion_matrix.png')
    evaluator.plot_roc_curve(Config.OUTPUTS_DIR / f'{model_name}_roc_curve.png')
    evaluator.plot_precision_recall_curve(Config.OUTPUTS_DIR / f'{model_name}_pr_curve.png')
    evaluator.plot_prediction_distribution(Config.OUTPUTS_DIR / f'{model_name}_predictions.png')
    evaluator.plot_all_metrics(Config.OUTPUTS_DIR / f'{model_name}_all_metrics.png')
    
    # Save results
    evaluator.save_results(Config.OUTPUTS_DIR / f'{model_name}_results.csv')
    
    return evaluator, metrics


def generate_recommendations(trainer, num_users=5, top_k=10):
    """Generate sample recommendations"""
    print("\n" + "=" * 70)
    print("🎯 GENERATING SAMPLE RECOMMENDATIONS")
    print("=" * 70)
    
    # Load data to get item IDs
    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'train_data.csv')
    all_item_ids = train_df['item_id'].unique().tolist()
    
    print(f"\n📊 Generating top-{top_k} recommendations for {num_users} sample users:\n")
    
    for user_id in range(min(num_users, 5)):  # Limit to 5 users
        recommendations = trainer.recommend_for_user(user_id, all_item_ids, top_k=top_k)
        
        print(f"👤 User {user_id}:")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            print(f"  {rank:2d}. Item {item_id:5d} - Score: {score:.4f}")
        print()


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='E-Commerce Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all --model ncf          # Run complete pipeline
  python main.py --preprocess               # Only preprocess data
  python main.py --train --model mf         # Only train Matrix Factorization
  python main.py --evaluate --model ncf     # Only evaluate NCF model
        """
    )
    
    parser.add_argument('--preprocess', action='store_true', 
                       help='Run data preprocessing')
    parser.add_argument('--train', action='store_true', 
                       help='Train model')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate model')
    parser.add_argument('--recommend', action='store_true', 
                       help='Generate sample recommendations')
    parser.add_argument('--all', action='store_true', 
                       help='Run complete pipeline (preprocess + train + evaluate + recommend)')
    parser.add_argument('--model', type=str, default='ncf', 
                       choices=['mf', 'ncf', 'deepfm'],
                       help='Model type: mf (Matrix Factorization), ncf (Neural CF), deepfm (Deep FM)')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=None, 
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # If --all or no arguments, run everything
    if args.all or not any([args.preprocess, args.train, args.evaluate, args.recommend]):
        args.preprocess = True
        args.train = True
        args.evaluate = True
        args.recommend = True
    
    # Print header
    print("\n" + "=" * 70)
    print("🛍️  E-COMMERCE RECOMMENDATION SYSTEM")
    print("=" * 70)
    print(f"Model: {args.model.upper()}")
    print(f"Device: {Config.get_device()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Project root: {project_root}")
    print("=" * 70)
    
    # Setup directories
    Config.setup_directories()
    
    # Step 1: Preprocessing
    if args.preprocess:
        data_path = Config.RAW_DATA_DIR / 'amazon_review.csv'
        
        if not data_path.exists():
            print(f"\n❌ ERROR: Dataset not found at {data_path}")
            print(f"   Please place amazon_reviews.csv in data/raw/ folder")
            print(f"\n   Expected location: {data_path}")
            print(f"   Current directory: {os.getcwd()}")
            return
        
        try:
            preprocessor = run_preprocessing(data_path)
        except Exception as e:
            print(f"\n❌ ERROR during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 2: Training
    trainer = None
    model = None
    
    if args.train:
        try:
            trainer, model = run_training(
                model_type=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
        except Exception as e:
            print(f"\n❌ ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 3: Evaluation
    if args.evaluate:
        if model is None:
            # Load trained model
            print("\n📂 Loading trained model...")
            try:
                test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'test_data.csv')
                num_users = test_df['user_id'].max() + 1
                num_items = test_df['item_id'].max() + 1
                
                model = create_model(
                    model_type=args.model,
                    num_users=num_users,
                    num_items=num_items,
                    embedding_dim=Config.EMBEDDING_DIM
                )
                
                model_path = Config.MODELS_DIR / f'best_{args.model}_model.pt'
                if model_path.exists():
                    model.load_state_dict(torch.load(model_path, map_location=Config.get_device()))
                    print(f"  ✅ Loaded model from {model_path}")
                else:
                    print(f"  ⚠️  No saved model found at {model_path}")
                    print(f"     Using untrained model for demonstration.")
            except Exception as e:
                print(f"\n❌ ERROR loading model: {e}")
                import traceback
                traceback.print_exc()
                return
        
        try:
            evaluator, metrics = run_evaluation(model, model_name=args.model)
        except Exception as e:
            print(f"\n❌ ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 4: Generate recommendations
    if args.recommend:
        if trainer is None and model is not None:
            # Create trainer with loaded model
            device = Config.get_device()
            trainer = Trainer(model, device=device)
        
        if trainer is not None:
            try:
                generate_recommendations(trainer, num_users=5, top_k=10)
            except Exception as e:
                print(f"\n❌ ERROR generating recommendations: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print("\n⚠️  No trained model available for recommendations.")
            print("   Please run with --train first.")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n📁 Outputs saved in:")
    print(f"  - Processed data: {Config.PROCESSED_DATA_DIR}")
    print(f"  - Trained models: {Config.MODELS_DIR}")
    print(f"  - Visualizations: {Config.OUTPUTS_DIR}")
    print("\n🎉 Project complete! Check the outputs folder for results.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)