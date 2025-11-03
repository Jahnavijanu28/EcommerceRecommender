"""
Streamlit Dashboard - Real-Time Product Recommendation System
Interactive interface for showcasing ML + NLP recommendations

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Amazon Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9900;
    }
    .recommendation-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# API CLIENT
# ==========================================

API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_recommendations(user_id, num_recs, method):
    """Get recommendations from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={
                "user_id": user_id,
                "num_recommendations": num_recs,
                "method": method
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def log_interaction(user_id, item_id, interaction_type, rating=None):
    """Log user interaction"""
    try:
        payload = {
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction_type
        }
        if rating is not None:
            payload["rating"] = rating
        
        response = requests.post(
            f"{API_BASE_URL}/interaction",
            json=payload,
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def get_user_profile(user_id):
    """Get user profile"""
    try:
        response = requests.get(f"{API_BASE_URL}/user/{user_id}/profile", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_stats():
    """Get system stats"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# ==========================================
# MAIN APP
# ==========================================

def main():
    # Header
    st.markdown('<div class="main-header">🛍️ Amazon Product Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Real-Time ML + NLP Recommendations</p>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ API Server is not running! Please start it first:")
        st.code("python api_server.py", language="bash")
        st.info("Once the server is running, refresh this page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # User selection
        user_id = st.number_input(
            "👤 Customer ID",
            min_value=0,
            max_value=5000,
            value=0,
            step=1,
            help="Select a customer to get recommendations for"
        )
        
        # Number of recommendations
        num_recs = st.slider(
            "🔢 Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        # Method selection
        method = st.selectbox(
            "🎯 Recommendation Method",
            options=["hybrid", "ml", "nlp"],
            format_func=lambda x: {
                "hybrid": "🔬 Hybrid (ML + NLP)",
                "ml": "🧠 ML (Neural Network)",
                "nlp": "📝 NLP (Text Similarity)"
            }[x],
            help="Choose recommendation algorithm"
        )
        
        st.divider()
        
        # System stats
        st.header("📊 System Stats")
        stats = get_stats()
        if stats:
            st.metric("Total Users", f"{stats['total_users']:,}")
            st.metric("Total Products", f"{stats['total_items']:,}")
            st.metric("Total Reviews", f"{stats['total_reviews']:,}")
            st.metric("Cache Size", stats['cache_size'])
        
        st.divider()
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Real-Time Recommendation System**
            
            - ✅ ML: Neural Collaborative Filtering
            - ✅ NLP: TF-IDF Text Analysis
            - ✅ Hybrid: Combined approach
            - ✅ Real-time updates
            - ✅ Caching for performance
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🎯 Recommendations for Customer #{user_id}")
        
        # Get user profile
        profile = get_user_profile(user_id)
        
        if profile:
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Total Reviews", profile['total_reviews'])
            with metrics_cols[1]:
                st.metric("Avg Rating", f"{profile['avg_rating']:.2f}/5.0")
            with metrics_cols[2]:
                st.metric("Positive Reviews", profile['positive_reviews'])
            with metrics_cols[3]:
                engagement = (profile['positive_reviews'] / max(profile['total_reviews'], 1)) * 100
                st.metric("Engagement", f"{engagement:.0f}%")
        
        st.divider()
        
        # Get recommendations button
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                start_time = time.time()
                result = get_recommendations(user_id, num_recs, method)
                elapsed_time = (time.time() - start_time) * 1000
                
                if result:
                    st.success(f"✅ Generated {len(result['recommendations'])} recommendations in {elapsed_time:.0f}ms")
                    
                    # Store in session state
                    st.session_state['last_recommendations'] = result
                    st.session_state['show_recommendations'] = True
    
    with col2:
        st.header("⚡ Real-Time Interactions")
        
        st.markdown("Simulate user actions:")
        
        # Interaction simulator
        interaction_item = st.number_input(
            "Product ID",
            min_value=0,
            max_value=5000,
            value=0,
            key="interact_item"
        )
        
        interaction_type = st.radio(
            "Interaction Type",
            options=["view", "click", "rate"],
            horizontal=True
        )
        
        rating_value = None
        if interaction_type == "rate":
            rating_value = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
        
        if st.button("📝 Log Interaction", use_container_width=True):
            success = log_interaction(user_id, interaction_item, interaction_type, rating_value)
            if success:
                st.success(f"✅ Logged {interaction_type} for product #{interaction_item}")
                st.info("💡 Recommendations updated! Cache cleared for this user.")
            else:
                st.error("❌ Failed to log interaction")
    
    # Display recommendations
    if st.session_state.get('show_recommendations', False):
        st.divider()
        result = st.session_state['last_recommendations']
        
        # Method info
        method_name = {
            "ml": "🧠 Machine Learning (Neural Network)",
            "nlp": "📝 Natural Language Processing (Text Similarity)",
            "hybrid": "🔬 Hybrid (ML + NLP Combined)"
        }.get(result['method'], result['method'])
        
        st.subheader(f"📋 Results using {method_name}")
        
        # Performance metrics
        perf_cols = st.columns(3)
        with perf_cols[0]:
            st.metric("Response Time", f"{result['response_time_ms']:.1f}ms")
        with perf_cols[1]:
            st.metric("Total Recommendations", len(result['recommendations']))
        with perf_cols[2]:
            cached = "✅ Yes" if result.get('from_cache', False) else "❌ No"
            st.metric("From Cache", cached)
        
        # Recommendations table
        recs_df = pd.DataFrame(result['recommendations'])
        
        # Display in a nice format
        for idx, rec in enumerate(result['recommendations']):
            with st.container():
                cols = st.columns([1, 3, 2, 2, 2])
                
                with cols[0]:
                    st.markdown(f"### #{rec['rank']}")
                
                with cols[1]:
                    st.markdown(f"**Product #{rec['item_id']}**")
                    stars = "⭐" * int(rec['avg_rating'])
                    st.caption(f"{stars} ({rec['avg_rating']:.2f}/5.0)")
                
                with cols[2]:
                    st.metric("Confidence", f"{rec['score']:.1%}")
                
                with cols[3]:
                    st.metric("Reviews", f"{rec['num_reviews']}")
                
                with cols[4]:
                    if st.button("👁️ View", key=f"view_{rec['item_id']}"):
                        log_interaction(user_id, rec['item_id'], "view")
                        st.toast(f"Viewed product #{rec['item_id']}")
            
            if idx < len(result['recommendations']) - 1:
                st.divider()
        
        # Visualization
        st.divider()
        st.subheader("📊 Recommendation Scores Visualization")
        
        fig = px.bar(
            recs_df,
            x='item_id',
            y='score',
            color='avg_rating',
            hover_data=['num_reviews'],
            labels={'item_id': 'Product ID', 'score': 'Recommendation Score', 'avg_rating': 'Avg Rating'},
            title='Recommendation Scores by Product',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution
        fig2 = px.scatter(
            recs_df,
            x='avg_rating',
            y='score',
            size='num_reviews',
            hover_data=['item_id'],
            labels={'avg_rating': 'Average Rating', 'score': 'Recommendation Score', 'num_reviews': 'Number of Reviews'},
            title='Recommendation Score vs Product Rating',
            color='score',
            color_continuous_scale='RdYlGn'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    # Initialize session state
    if 'show_recommendations' not in st.session_state:
        st.session_state['show_recommendations'] = False
    
    main()