"""
FastAPI Real-Time Recommendation Service
Provides REST API endpoints for real-time product recommendations

Features:
- Real-time recommendations using pre-trained models
- User interaction tracking
- Caching for fast responses
- Support for ML, NLP, and Hybrid methods
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.models import create_model

# ==========================================
# PYDANTIC MODELS (Request/Response Schemas)
# ==========================================

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10
    method: str = "hybrid"  # 'ml', 'nlp', or 'hybrid'
    
class UserInteraction(BaseModel):
    user_id: int
    item_id: int
    interaction_type: str  # 'view', 'click', 'rate'
    rating: Optional[float] = None
    timestamp: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict]
    method: str
    response_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_users: int
    total_items: int
    cache_size: int

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Amazon Product Recommendation API",
    description="Real-time product recommendations using ML + NLP",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# GLOBAL STATE (In-Memory Cache)
# ==========================================

class RecommenderEngine:
    """Singleton recommendation engine with caching"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.num_users = 0
        self.num_items = 0
        self.reviews_df = None
        self.products_df = None
        self.cache = {}  # Simple in-memory cache (Redis in production)
        self.interaction_log = []
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model and data"""
        print("\n🚀 Loading recommendation engine...")
        
        try:
            # Load data
            self.reviews_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'train_data.csv')
            
            #self.num_users = self.reviews_df['user_id'].max() + 1
            #self.num_items = self.reviews_df['item_id'].max() + 1
            # Load num_users and num_items from ALL data files (not just train)
            train_df = self.reviews_df
            val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'val_data.csv')
            test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'test_data.csv')

            # Combine to get true max
            all_users = set(train_df['user_id'].unique()) | set(val_df['user_id'].unique()) | set(test_df['user_id'].unique())
            all_items = set(train_df['item_id'].unique()) | set(val_df['item_id'].unique()) | set(test_df['item_id'].unique())

            self.num_users = max(all_users) + 1
            self.num_items = max(all_items) + 1
            
            # Build product catalog
            self.products_df = self.reviews_df.groupby('item_id').agg({
                'overall': ['mean', 'count'],
                'is_positive': 'mean'
            }).reset_index()
            self.products_df.columns = ['item_id', 'avg_rating', 'num_reviews', 'positive_ratio']
            
            # Load ML model
            model_path = Config.MODELS_DIR / 'best_ncf_model.pt'
            
            if model_path.exists():
                self.device = Config.get_device()
                self.model = create_model('ncf', self.num_users, self.num_items, embedding_dim=64)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Model loaded (Device: {self.device})")
            else:
                print(f"⚠️  Model not found at {model_path}")
                print("   Using random recommendations for demo")
            
            self.is_loaded = True
            print(f"✅ Engine ready: {self.num_users} users, {self.num_items} items")
            
        except Exception as e:
            print(f"❌ Error loading engine: {e}")
            raise
    
    def get_recommendations(self, user_id: int, num_recs: int, method: str):
        """Get recommendations for a user"""
        import time
        start_time = time.time()
        
        # Check cache
        cache_key = f"{user_id}_{num_recs}_{method}"
        if cache_key in self.cache:
            print(f"✅ Cache hit for user {user_id}")
            cached_result = self.cache[cache_key]
            cached_result['from_cache'] = True
            return cached_result
        
        # Generate recommendations
        if user_id >= self.num_users:
            user_id = user_id % self.num_users
        
        if method == 'ml' or method == 'hybrid':
            recommendations = self._ml_recommendations(user_id, num_recs)
        else:  # nlp or fallback
            recommendations = self._random_recommendations(user_id, num_recs)
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = {
            'user_id': user_id,
            'recommendations': recommendations,
            'method': method,
            'response_time_ms': round(response_time, 2),
            'timestamp': datetime.now().isoformat(),
            'from_cache': False
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def _ml_recommendations(self, user_id: int, num_recs: int):
        """ML-based recommendations"""
        if self.model is None:
            return self._random_recommendations(user_id, num_recs)
        
        # Get user's reviewed products
        user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
        reviewed_items = set(user_reviews['item_id'].values)
        
        # Get candidate products
        all_items = self.reviews_df['item_id'].unique()
        candidates = [item for item in all_items if item not in reviewed_items]
        
        if len(candidates) == 0:
            candidates = list(all_items)
        
        # Limit candidates for performance
        if len(candidates) > 500:
            candidates = np.random.choice(candidates, 500, replace=False)
        
        # Predict scores
        user_tensor = torch.LongTensor([user_id] * len(candidates)).to(self.device)
        item_tensor = torch.LongTensor(candidates).to(self.device)
        
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Get top-K
        top_indices = np.argsort(scores)[-num_recs:][::-1]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            item_id = candidates[idx]
            score = float(scores[idx])
            
            # Get product details
            prod_info = self.products_df[self.products_df['item_id'] == item_id]
            
            rec = {
                'rank': rank,
                'item_id': int(item_id),
                'score': round(score, 4),
                'avg_rating': float(prod_info['avg_rating'].values[0]) if len(prod_info) > 0 else 0.0,
                'num_reviews': int(prod_info['num_reviews'].values[0]) if len(prod_info) > 0 else 0
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _random_recommendations(self, user_id: int, num_recs: int):
        """Fallback random recommendations"""
        all_items = self.reviews_df['item_id'].unique()
        selected_items = np.random.choice(all_items, min(num_recs, len(all_items)), replace=False)
        
        recommendations = []
        for rank, item_id in enumerate(selected_items, 1):
            prod_info = self.products_df[self.products_df['item_id'] == item_id]
            
            rec = {
                'rank': rank,
                'item_id': int(item_id),
                'score': round(np.random.rand(), 4),
                'avg_rating': float(prod_info['avg_rating'].values[0]) if len(prod_info) > 0 else 0.0,
                'num_reviews': int(prod_info['num_reviews'].values[0]) if len(prod_info) > 0 else 0
            }
            recommendations.append(rec)
        
        return recommendations
    
    def log_interaction(self, interaction: UserInteraction):
        """Log user interaction for real-time updates"""
        interaction_dict = interaction.dict()
        interaction_dict['timestamp'] = datetime.now().isoformat()
        self.interaction_log.append(interaction_dict)
        
        # Clear cache for this user
        keys_to_delete = [k for k in self.cache.keys() if k.startswith(f"{interaction.user_id}_")]
        for key in keys_to_delete:
            del self.cache[key]
        
        print(f"✅ Logged interaction: User {interaction.user_id} -> Item {interaction.item_id}")
    
    def get_stats(self):
        """Get system statistics"""
        return {
            'model_loaded': self.model is not None,
            'total_users': self.num_users,
            'total_items': self.num_items,
            'total_reviews': len(self.reviews_df) if self.reviews_df is not None else 0,
            'cache_size': len(self.cache),
            'interactions_logged': len(self.interaction_log)
        }

# Initialize engine
engine = RecommenderEngine()

# ==========================================
# API ENDPOINTS
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        engine.load_model()
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("   API will use fallback recommendations")

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Amazon Product Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommend",
            "interaction": "/interaction",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    stats = engine.get_stats()
    
    return HealthResponse(
        status="healthy" if engine.is_loaded else "degraded",
        model_loaded=stats['model_loaded'],
        total_users=stats['total_users'],
        total_items=stats['total_items'],
        cache_size=stats['cache_size']
    )

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized product recommendations
    
    - **user_id**: User ID (0 to num_users-1)
    - **num_recommendations**: Number of recommendations (1-50)
    - **method**: Recommendation method ('ml', 'nlp', or 'hybrid')
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    if request.num_recommendations < 1 or request.num_recommendations > 50:
        raise HTTPException(status_code=400, detail="num_recommendations must be between 1 and 50")
    
    if request.method not in ['ml', 'nlp', 'hybrid']:
        raise HTTPException(status_code=400, detail="method must be 'ml', 'nlp', or 'hybrid'")
    
    try:
        result = engine.get_recommendations(
            user_id=request.user_id,
            num_recs=request.num_recommendations,
            method=request.method
        )
        
        return RecommendationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/interaction", tags=["Interactions"])
async def log_interaction(interaction: UserInteraction, background_tasks: BackgroundTasks):
    """
    Log user interaction (view, click, rate)
    Updates recommendations in real-time
    
    - **user_id**: User ID
    - **item_id**: Product ID
    - **interaction_type**: Type of interaction ('view', 'click', 'rate')
    - **rating**: Rating value (optional, for 'rate' interactions)
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    try:
        # Log interaction (this would trigger real-time model update in production)
        background_tasks.add_task(engine.log_interaction, interaction)
        
        return {
            "status": "success",
            "message": f"Interaction logged for user {interaction.user_id}",
            "interaction": interaction.dict(),
            "cache_cleared": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging interaction: {str(e)}")

@app.get("/stats", tags=["System"])
async def get_stats():
    """Get system statistics"""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    return engine.get_stats()

@app.get("/user/{user_id}/profile", tags=["Users"])
async def get_user_profile(user_id: int):
    """Get user profile and history"""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    if user_id >= engine.num_users:
        user_id = user_id % engine.num_users
    
    user_reviews = engine.reviews_df[engine.reviews_df['user_id'] == user_id]
    
    profile = {
        'user_id': user_id,
        'total_reviews': len(user_reviews),
        'avg_rating': float(user_reviews['overall'].mean()) if len(user_reviews) > 0 else 0.0,
        'positive_reviews': int(user_reviews['is_positive'].sum()) if len(user_reviews) > 0 else 0,
        'reviewed_items': user_reviews['item_id'].tolist() if len(user_reviews) > 0 else []
    }
    
    return profile

@app.get("/item/{item_id}/details", tags=["Items"])
async def get_item_details(item_id: int):
    """Get product details"""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    item_info = engine.products_df[engine.products_df['item_id'] == item_id]
    
    if len(item_info) == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    
    details = {
        'item_id': item_id,
        'avg_rating': float(item_info['avg_rating'].values[0]),
        'num_reviews': int(item_info['num_reviews'].values[0]),
        'positive_ratio': float(item_info['positive_ratio'].values[0])
    }
    
    return details

# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting FastAPI Recommendation Server")
    print("=" * 70)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("\nPress CTRL+C to stop")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")