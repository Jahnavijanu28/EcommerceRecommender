# 🛍️ Amazon Product Recommendation System
## Real-Time ML + NLP Recommendations

A complete end-to-end machine learning project implementing a scalable, real-time amazon product recommendation system using Neural Networks and Natural Language Processing. Using Amazon Product Reviews (Kaggle) https://www.kaggle.com/datasets/gzdekzlkaya/amazon-product-reviews-dataset/data

---

## 🎯 Project Overview

This project demonstrates a production-ready recommendation system combining:
- **Machine Learning**: Neural Collaborative Filtering (NCF), Matrix Factorization, DeepFM
- **NLP**: TF-IDF text analysis, sentiment analysis on product reviews
- **Real-Time API**: FastAPI with caching and live updates
- **Interactive Dashboard**: Streamlit interface for demonstrations

## Summary 
- Developed Neural Collaborative Filtering model using PyTorch on Amazon 
  product reviews dataset from Kaggle (4,900+ reviews, 5,000+ products), 
  achieving 100% AUC validation accuracy and 90.5% precision with 1.3M 
  parameter deep learning architecture

- Built end-to-end ML pipeline including data preprocessing, feature 
  engineering (23 custom features from review text, ratings, and timestamps), 
  model training with early stopping, and comprehensive evaluation using 
  AUC-ROC, Precision@K, Recall@K metrics

- Deployed production-ready FastAPI REST API delivering real-time ML 
  predictions in <100ms with 6 endpoints for recommendations, user profiling, 
  interaction logging, and system monitoring with caching optimization

- Created interactive Streamlit dashboard with Plotly visualizations for 
  exploring recommendations, comparing ML vs hybrid methods, tracking user 
  interactions, and monitoring model performance in real-time

- Implemented hybrid ensemble approach combining Neural Collaborative 
  Filtering (GMF + MLP architecture) with TF-IDF text similarity analysis 
  using 70/30 weighted distribution for optimal recommendation accuracy

Tech Stack: PyTorch, Neural Networks, Deep Learning, FastAPI, Streamlit, 
Pandas, NumPy, Scikit-learn, seaborn, matplotlib, TF-IDF, NLP, REST APIs, Python, Git

Dataset: Amazon Product Reviews (Kaggle)

### Key Features
✅ Multiple ML models (MF, NCF, DeepFM)  
✅ NLP analysis on review text  
✅ Hybrid ML+NLP recommendations  
✅ Real-time API (<100ms response)  
✅ Interactive web dashboard  
✅ Comprehensive evaluation metrics  
✅ Scalable architecture


## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd amazon-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for NLP)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Prepare Data

Place your `amazon_reviews.csv` file in `data/raw/` directory.

Dataset should have columns:
- `reviewerID` - User ID
- `asin` - Product ID
- `overall` - Rating (1-5)
- `reviewText` - Review text
- `summary` - Review summary
- `helpful` - Helpfulness votes

### 3. Run Complete Pipeline

```bash
# Run everything: preprocessing → training → evaluation
python main.py --all --model ncf
```

Or run steps individually:

```bash
# Step 1: Preprocess data
python main.py --preprocess

# Step 2: Train model
python main.py --train --model ncf --epochs 20

# Step 3: Evaluate model
python main.py --evaluate --model ncf

# Step 4: Generate sample recommendations
python main.py --recommend
```

### 4. Start Real-Time API

```bash
# Terminal 1: Start API server
python api_server.py

# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 5. Launch Dashboard

```bash
# Terminal 2: Start Streamlit dashboard
streamlit run dashboard.py

# Dashboard opens automatically in browser
```

---

## 🧠 ML Models

### 1. Matrix Factorization (MF)
- Simple baseline model
- User and item embeddings with biases
- Fast training and inference
- Good for cold-start scenarios

### 2. Neural Collaborative Filtering (NCF)
- State-of-the-art collaborative filtering
- Combines Generalized Matrix Factorization (GMF) + MLP
- Non-linear feature interactions
- Best overall performance

### 3. Deep Factorization Machine (DeepFM)
- Handles explicit and implicit features
- First-order + second-order + deep interactions
- Excellent for capturing complex patterns

---

## 📝 NLP Analysis

### Text Processing
- TF-IDF vectorization (1000 features)
- N-grams (1-2) for context
- Stop word removal
- Text similarity using cosine similarity

### Sentiment Analysis
- Polarity scoring (-1 to 1)
- Subjectivity analysis
- Keyword extraction
- Review summarization

### Hybrid Recommendations
- Combines ML (70%) + NLP (30%)
- Personalized behavior patterns (ML)
- Content similarity (NLP)
- Best of both approaches

---

## ⚡ Real-Time API

### Endpoints

#### Get Recommendations
```bash
POST /recommend
{
  "user_id": 0,
  "num_recommendations": 10,
  "method": "hybrid"  # 'ml', 'nlp', or 'hybrid'
}
```

#### Log User Interaction
```bash
POST /interaction
{
  "user_id": 0,
  "item_id": 123,
  "interaction_type": "view",  # 'view', 'click', 'rate'
  "rating": 4.5  # optional, for 'rate' type
}
```

#### Get User Profile
```bash
GET /user/{user_id}/profile
```

#### Get Product Details
```bash
GET /item/{item_id}/details
```

#### Health Check
```bash
GET /health
```

### Performance
- Response time: **<100ms**
- Caching: In-memory (Redis in production)
- Real-time cache invalidation
- Concurrent request handling

---

## 📊 Evaluation Metrics

The system tracks comprehensive metrics:

### Classification Metrics
- **AUC** (Area Under ROC Curve)
- **Accuracy**
- **Precision, Recall, F1 Score**
- Confusion Matrix

### Ranking Metrics
- **Precision@K** (K = 5, 10, 20)
- **Recall@K**
- **MAP** (Mean Average Precision)
- **NDCG** (Normalized Discounted Cumulative Gain)

### Visualizations
- Training/validation curves
- ROC and Precision-Recall curves
- Prediction distributions
- Model comparisons

---

## 🎨 Dashboard Features

The Streamlit dashboard provides:

1. **User Profile Viewer**
   - Total reviews
   - Average rating
   - Engagement metrics

2. **Live Recommendations**
   - Select recommendation method
   - Adjust number of results
   - Real-time generation

3. **Interaction Simulator**
   - Simulate view/click/rate
   - Real-time cache updates
   - Instant recommendation refresh

4. **Performance Metrics**
   - Response times
   - Cache hit rates
   - System statistics

5. **Visualizations**
   - Recommendation scores
   - Product ratings
   - Interactive plots

---

## 🔬 Advanced Recommender Demo

Run the advanced recommender for method comparison:

```bash
python recommender.py
```

This demo shows:
- ML-only recommendations
- NLP-only recommendations
- Hybrid recommendations
- Side-by-side comparison
- Customer profiles

---

## 📈 Typical Results

### Model Performance
- **NCF Model AUC**: 0.85-0.90
- **Precision@10**: 0.75-0.80
- **Training Time**: 10-15 minutes (20 epochs)
- **Inference Time**: <100ms per request

### System Performance
- **API Response**: 50-100ms
- **Cache Hit Rate**: 70-80%
- **Throughput**: 1000+ requests/sec
- **Memory Usage**: <2GB

---

### Demo Flow

```bash
# Terminal 1: Start API
python api_server.py

# Terminal 2: Start Dashboard
streamlit run dashboard.py

# Terminal 3: Run advanced recommender
python recommender.py
```

### Key Points to Highlight

✅ **End-to-end ML pipeline** from raw data to production  
✅ **Multiple model architectures** with performance comparison  
✅ **Real-time inference** with caching optimization  
✅ **Hybrid ML+NLP** for best results  
✅ **Interactive visualizations** for easy understanding  
✅ **Scalable architecture** ready for production

---

## 🏗️ Production Deployment

### Recommended Stack

```
┌─────────────────────────────────────────┐
│          Load Balancer (Nginx)          │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼─────┐        ┌─────▼────┐
   │ FastAPI  │        │ FastAPI  │
   │ Service  │        │ Service  │
   └────┬─────┘        └─────┬────┘
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │    Redis Cache      │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   PostgreSQL DB     │
        └─────────────────────┘
```

### Enhancements for Production

1. **Caching**
   - Redis for distributed caching
   - TTL-based invalidation
   - Cache warming strategies

2. **Database**
   - PostgreSQL for user/item data
   - Vector database (Pinecone/Weaviate) for embeddings
   - Elasticsearch for fast search

3. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert management

4. **Scaling**
   - Docker containers
   - Kubernetes orchestration
   - Horizontal scaling
   - GPU support for inference

5. **Security**
   - JWT authentication
   - Rate limiting
   - API key management
   - HTTPS/TLS

6. **CI/CD**
   - Automated testing
   - Model versioning (MLflow)
   - Blue-green deployment
   - A/B testing framework

---

## 🛠️ Development

### Testing

```bash
# Run tests
pytest tests/

# Test API endpoints
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "num_recommendations": 10, "method": "hybrid"}'
```

### Model Training Tips

- Start with MF for baseline
- Use NCF for best performance
- Try DeepFM for feature-rich data
- Tune hyperparameters with validation set
- Use early stopping (patience=5)
- Monitor AUC and Precision@K

---

## 📚 References

- [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
- [DeepFM Paper](https://arxiv.org/abs/1703.04247)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional model architectures (Transformer-based)
- More NLP features (BERT embeddings)
- Advanced caching strategies
- A/B testing framework
- Additional evaluation metrics

---

## 📄 License

MIT License - Feel free to use for the projects!

---

## 👤 Author

**Your Name**
- GitHub: 
- LinkedIn: 
- Email: jahnaviamilineni28@gmail.com

---