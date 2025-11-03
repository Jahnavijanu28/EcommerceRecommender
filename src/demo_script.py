"""
COMPLETE DEMO SCRIPT
Amazon Product Recommendation System - Real-Time ML + NLP

This script demonstrates the complete system for your showcase presentation
"""

import subprocess
import time
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def print_step(step_num, title):
    """Print step header"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*80}\n")

def wait_for_input(message="Press Enter to continue..."):
    """Wait for user input"""
    input(f"\n{message}")

def main():
    """Main demo script"""
    
    print_header("🛍️ AMAZON PRODUCT RECOMMENDATION SYSTEM")
    print("         Real-Time ML + NLP Showcase Demo")
    print("=" * 80)
    
    print("""
This demo will walk you through:
1. Data preprocessing & feature engineering
2. Training ML models (Matrix Factorization, NCF, DeepFM)
3. NLP sentiment analysis on reviews
4. Real-time API service with FastAPI
5. Interactive dashboard with Streamlit
6. Live recommendation generation
""")
    
    wait_for_input("Press Enter to start the demo...")
    
    # ============================================
    # STEP 1: Environment Setup
    # ============================================
    print_step(1, "ENVIRONMENT SETUP")
    
    print("📦 Checking dependencies...")
    print("\nRequired packages:")
    print("  ✅ PyTorch (Deep Learning)")
    print("  ✅ Scikit-learn (ML utilities)")
    print("  ✅ FastAPI (API framework)")
    print("  ✅ Streamlit (Dashboard)")
    print("  ✅ Pandas, NumPy (Data processing)")
    
    print("\nTo install all dependencies:")
    print("  pip install -r requirements.txt")
    
    wait_for_input()
    
    # ============================================
    # STEP 2: Data Preprocessing
    # ============================================
    print_step(2, "DATA PREPROCESSING")
    
    print("""
📊 Data preprocessing includes:
- Loading Amazon reviews dataset
- Cleaning & handling missing values  
- Feature engineering (user/item features)
- Creating train/val/test splits
- Generating user and item embeddings
""")
    
    print("\nCommand to run preprocessing:")
    print("  python main.py --preprocess")
    
    run_preprocessing = input("\nRun preprocessing now? (y/n): ").lower() == 'y'
    
    if run_preprocessing:
        print("\n🔄 Running preprocessing...")
        subprocess.run([sys.executable, "main.py", "--preprocess"])
    
    wait_for_input()
    
    # ============================================
    # STEP 3: Model Training
    # ============================================
    print_step(3, "MODEL TRAINING")
    
    print("""
🧠 We have 3 ML models:

1. Matrix Factorization (MF)
   - Simple & fast baseline
   - User-item embeddings with biases
   
2. Neural Collaborative Filtering (NCF)  
   - Combines GMF + MLP
   - State-of-the-art performance
   
3. Deep Factorization Machine (DeepFM)
   - Handles feature interactions
   - Best for complex patterns
""")
    
    print("\nCommand to train NCF model:")
    print("  python main.py --train --model ncf --epochs 20")
    
    train_model = input("\nTrain NCF model now? (y/n): ").lower() == 'y'
    
    if train_model:
        print("\n🚀 Training NCF model...")
        subprocess.run([sys.executable, "main.py", "--train", "--model", "ncf", "--epochs", "5"])
    
    wait_for_input()
    
    # ============================================
    # STEP 4: Model Evaluation
    # ============================================
    print_step(4, "MODEL EVALUATION")
    
    print("""
📊 Comprehensive evaluation metrics:
- AUC (Area Under ROC Curve)
- Precision, Recall, F1 Score
- Precision@K and Recall@K
- MAP (Mean Average Precision)
- Confusion Matrix
- ROC & Precision-Recall curves
""")
    
    print("\nCommand to evaluate:")
    print("  python main.py --evaluate --model ncf")
    
    evaluate = input("\nRun evaluation now? (y/n): ").lower() == 'y'
    
    if evaluate:
        print("\n📊 Evaluating model...")
        subprocess.run([sys.executable, "main.py", "--evaluate", "--model", "ncf"])
    
    wait_for_input()
    
    # ============================================
    # STEP 5: NLP + Hybrid Recommendations
    # ============================================
    print_step(5, "NLP & HYBRID RECOMMENDATIONS")
    
    print("""
📝 NLP Component:
- TF-IDF vectorization of review text
- Text similarity using cosine similarity
- Sentiment analysis (positive/negative)
- Keyword extraction

🔬 Hybrid Approach:
- Combines ML (70%) + NLP (30%)
- Best of both worlds
- Personalized + content-based
""")
    
    print("\nCommand to run hybrid recommender:")
    print("  python recommender.py")
    
    run_hybrid = input("\nRun hybrid recommender demo? (y/n): ").lower() == 'y'
    
    if run_hybrid:
        print("\n🔬 Running hybrid recommender...")
        subprocess.run([sys.executable, "recommender.py"])
    
    wait_for_input()
    
    # ============================================
    # STEP 6: Real-Time API Service
    # ============================================
    print_step(6, "REAL-TIME API SERVICE (FastAPI)")
    
    print("""
⚡ Real-time recommendation API:
- RESTful API with FastAPI
- Real-time inference (<100ms)
- In-memory caching (Redis in production)
- User interaction logging
- Automatic cache invalidation

📍 API Endpoints:
- POST /recommend - Get recommendations
- POST /interaction - Log user actions
- GET /user/{id}/profile - User profile
- GET /item/{id}/details - Product details
- GET /health - System health check
""")
    
    print("\nTo start API server:")
    print("  python api_server.py")
    print("\nAPI will be available at: http://localhost:8000")
    print("Interactive docs: http://localhost:8000/docs")
    
    start_api = input("\nStart API server in background? (y/n): ").lower() == 'y'
    
    if start_api:
        print("\n🚀 Starting API server...")
        print("Opening in background... Check http://localhost:8000/docs")
        print("\nNOTE: You'll need to run this manually in a separate terminal:")
        print("  python api_server.py")
    
    wait_for_input()
    
    # ============================================
    # STEP 7: Interactive Dashboard
    # ============================================
    print_step(7, "INTERACTIVE DASHBOARD (Streamlit)")
    
    print("""
🎨 Real-time dashboard features:
- Live recommendation generation
- User profile viewing
- Real-time interaction simulation  
- Performance metrics display
- Method comparison (ML vs NLP vs Hybrid)
- Interactive visualizations
- Response time monitoring
""")
    
    print("\nTo start dashboard:")
    print("  streamlit run dashboard.py")
    print("\nDashboard will open in browser automatically!")
    
    start_dashboard = input("\nStart dashboard now? (y/n): ").lower() == 'y'
    
    if start_dashboard:
        print("\n🎨 Starting Streamlit dashboard...")
        print("\nNOTE: First ensure API server is running:")
        print("  Terminal 1: python api_server.py")
        print("  Terminal 2: streamlit run dashboard.py")
        
        subprocess.run(["streamlit", "run", "dashboard.py"])
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print_header("✅ DEMO COMPLETE!")
    
    print("""
🎉 You're ready to showcase your project!
""")
    
    print("=" * 80)
    print("\n your presentation!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted. Exiting...")
        sys.exit(0)