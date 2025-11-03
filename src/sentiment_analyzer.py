"""
NLP Sentiment Analysis Module
Analyzes Amazon review text to extract sentiment and insights
Supports TextBlob and VADER for sentiment analysis
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try importing NLP libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not installed. Install with: pip install textblob")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  NLTK not installed. Install with: pip install nltk")


class SentimentAnalyzer:
    """
    Sentiment analysis for product reviews
    Uses TextBlob and/or VADER for sentiment scoring and feature extraction
    """
    
    def __init__(self, method='textblob'):
        """
        Initialize sentiment analyzer
        
        Args:
            method: 'textblob', 'vader', or 'both'
        """
        self.method = method
        
        # Predefined word lists
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect',
            'awesome', 'fantastic', 'wonderful', 'outstanding', 'superb', 'brilliant',
            'impressive', 'incredible', 'spectacular', 'delightful', 'fabulous',
            'exceptional', 'remarkable', 'quality', 'recommend', 'satisfied', 'happy',
            'beautiful', 'nice', 'pleasant', 'enjoyable', 'comfortable', 'reliable',
            'sturdy', 'durable', 'efficient', 'effective', 'helpful', 'useful',
            'powerful', 'fast', 'quick', 'easy', 'simple', 'convenient', 'affordable'
        ])
        
        self.negative_words = set([
            'bad', 'poor', 'worst', 'terrible', 'hate', 'awful', 'disappointing',
            'useless', 'broken', 'defective', 'horrible', 'pathetic', 'waste',
            'garbage', 'junk', 'cheap', 'failed', 'returned', 'refund', 'complaint',
            'avoid', 'never', 'disgusting', 'unacceptable', 'frustrating', 'annoying',
            'difficult', 'hard', 'complicated', 'confusing', 'slow', 'expensive',
            'overpriced', 'uncomfortable', 'unreliable', 'flimsy', 'weak', 'fragile'
        ])
        
        # Initialize VADER if available and requested
        self.vader_analyzer = None
        if method in ['vader', 'both'] and VADER_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                print("Downloading VADER lexicon...")
                nltk.download('vader_lexicon', quiet=True)
                self.vader_analyzer = SentimentIntensityAnalyzer()
        
        print(f"✅ Sentiment Analyzer initialized (Method: {method})")
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment_textblob(self, text):
        """
        Get sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            dict with polarity, subjectivity, and sentiment label
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
        
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
        
        try:
            blob = TextBlob(cleaned)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'polarity': round(polarity, 4),
                'subjectivity': round(subjectivity, 4),
                'sentiment': sentiment
            }
        except Exception as e:
            print(f"Error in TextBlob analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
    
    def get_sentiment_vader(self, text):
        """
        Get sentiment using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            dict with compound score and sentiment label
        """
        if not self.vader_analyzer:
            return {'compound': 0.0, 'sentiment': 'neutral'}
        
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return {'compound': 0.0, 'sentiment': 'neutral'}
        
        try:
            scores = self.vader_analyzer.polarity_scores(cleaned)
            compound = scores['compound']
            
            # Classify sentiment
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'compound': round(compound, 4),
                'positive': round(scores['pos'], 4),
                'negative': round(scores['neg'], 4),
                'neutral': round(scores['neu'], 4),
                'sentiment': sentiment
            }
        except Exception as e:
            print(f"Error in VADER analysis: {e}")
            return {'compound': 0.0, 'sentiment': 'neutral'}
    
    def get_sentiment_score(self, text):
        """
        Get sentiment score using selected method
        
        Args:
            text: Text to analyze
            
        Returns:
            dict with sentiment scores and label
        """
        if self.method == 'textblob':
            return self.get_sentiment_textblob(text)
        elif self.method == 'vader':
            return self.get_sentiment_vader(text)
        elif self.method == 'both':
            textblob_result = self.get_sentiment_textblob(text)
            vader_result = self.get_sentiment_vader(text)
            
            # Combine results
            combined = {
                'textblob_polarity': textblob_result['polarity'],
                'textblob_subjectivity': textblob_result['subjectivity'],
                'vader_compound': vader_result.get('compound', 0.0),
                'vader_positive': vader_result.get('positive', 0.0),
                'vader_negative': vader_result.get('negative', 0.0),
                'vader_neutral': vader_result.get('neutral', 0.0),
                'sentiment': textblob_result['sentiment']  # Use TextBlob classification
            }
            return combined
        else:
            return self.get_sentiment_textblob(text)
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text
        
        Args:
            text: Text to analyze
            top_n: Number of keywords to extract
            
        Returns:
            List of (word, frequency) tuples
        """
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return []
        
        # Simple word frequency
        words = cleaned.split()
        
        # Remove common stop words
        stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'don', 'now'
        ])
        
        # Filter words
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        return word_freq.most_common(top_n)
    
    def classify_review_sentiment(self, text):
        """
        Classify review as positive, negative, or neutral with confidence
        
        Args:
            text: Review text
            
        Returns:
            dict with sentiment, confidence, and reasoning
        """
        # Get sentiment scores
        scores = self.get_sentiment_score(text)
        
        # Count positive and negative words
        cleaned = self.clean_text(text)
        words = set(cleaned.split())
        
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        
        # Calculate confidence
        if self.method == 'textblob' or self.method == 'both':
            polarity = scores.get('polarity', scores.get('textblob_polarity', 0))
            confidence = abs(polarity)
        else:
            compound = scores.get('compound', scores.get('vader_compound', 0))
            confidence = abs(compound)
        
        return {
            'sentiment': scores['sentiment'],
            'confidence': round(confidence, 4),
            'positive_words_count': positive_count,
            'negative_words_count': negative_count,
            'scores': scores
        }
    
    def analyze_reviews_batch(self, reviews_df, text_column='reviewText'):
        """
        Analyze sentiment for a batch of reviews
        
        Args:
            reviews_df: DataFrame with reviews
            text_column: Name of column containing review text
            
        Returns:
            DataFrame with added sentiment columns
        """
        print(f"\n📊 Analyzing sentiment for {len(reviews_df)} reviews...")
        
        results = []
        
        for idx, text in enumerate(reviews_df[text_column]):
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(reviews_df)} reviews...")
            
            sentiment_result = self.classify_review_sentiment(str(text))
            results.append(sentiment_result)
        
        # Add results to dataframe
        result_df = reviews_df.copy()
        result_df['sentiment'] = [r['sentiment'] for r in results]
        result_df['sentiment_confidence'] = [r['confidence'] for r in results]
        result_df['positive_words_count'] = [r['positive_words_count'] for r in results]
        result_df['negative_words_count'] = [r['negative_words_count'] for r in results]
        
        print(f"✅ Sentiment analysis complete!")
        print(f"\n📈 Distribution:")
        print(result_df['sentiment'].value_counts())
        
        return result_df
    
    def get_sentiment_summary(self, reviews_df):
        """
        Get summary statistics of sentiment analysis
        
        Args:
            reviews_df: DataFrame with sentiment columns
            
        Returns:
            dict with summary statistics
        """
        if 'sentiment' not in reviews_df.columns:
            print("⚠️  No sentiment column found. Run analyze_reviews_batch first.")
            return {}
        
        summary = {
            'total_reviews': len(reviews_df),
            'positive_count': (reviews_df['sentiment'] == 'positive').sum(),
            'negative_count': (reviews_df['sentiment'] == 'negative').sum(),
            'neutral_count': (reviews_df['sentiment'] == 'neutral').sum(),
            'positive_ratio': (reviews_df['sentiment'] == 'positive').mean(),
            'negative_ratio': (reviews_df['sentiment'] == 'negative').mean(),
            'neutral_ratio': (reviews_df['sentiment'] == 'neutral').mean(),
            'avg_confidence': reviews_df['sentiment_confidence'].mean() if 'sentiment_confidence' in reviews_df.columns else 0
        }
        
        return summary
    
    def get_top_positive_reviews(self, reviews_df, top_n=5):
        """Get top N most positive reviews"""
        if 'sentiment_confidence' not in reviews_df.columns:
            return []
        
        positive_reviews = reviews_df[reviews_df['sentiment'] == 'positive']
        top_positive = positive_reviews.nlargest(top_n, 'sentiment_confidence')
        
        return top_positive
    
    def get_top_negative_reviews(self, reviews_df, top_n=5):
        """Get top N most negative reviews"""
        if 'sentiment_confidence' not in reviews_df.columns:
            return []
        
        negative_reviews = reviews_df[reviews_df['sentiment'] == 'negative']
        top_negative = negative_reviews.nlargest(top_n, 'sentiment_confidence')
        
        return top_negative


# ========================================
# UTILITY FUNCTIONS
# ========================================

def analyze_product_sentiment(reviews_df, product_id, text_column='reviewText', product_id_column='asin'):
    """
    Analyze sentiment for a specific product
    
    Args:
        reviews_df: DataFrame with reviews
        product_id: Product ID to analyze
        text_column: Column with review text
        product_id_column: Column with product IDs
        
    Returns:
        dict with product sentiment summary
    """
    # Filter reviews for this product
    product_reviews = reviews_df[reviews_df[product_id_column] == product_id]
    
    if len(product_reviews) == 0:
        return {'error': 'No reviews found for this product'}
    
    # Analyze sentiment
    analyzer = SentimentAnalyzer(method='textblob')
    analyzed = analyzer.analyze_reviews_batch(product_reviews, text_column=text_column)
    
    # Get summary
    summary = analyzer.get_sentiment_summary(analyzed)
    summary['product_id'] = product_id
    
    return summary


def compare_products_sentiment(reviews_df, product_ids, text_column='reviewText', product_id_column='asin'):
    """
    Compare sentiment across multiple products
    
    Args:
        reviews_df: DataFrame with reviews
        product_ids: List of product IDs to compare
        text_column: Column with review text
        product_id_column: Column with product IDs
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for product_id in product_ids:
        print(f"\nAnalyzing product: {product_id}")
        summary = analyze_product_sentiment(
            reviews_df, product_id, text_column, product_id_column
        )
        if 'error' not in summary:
            results.append(summary)
    
    comparison_df = pd.DataFrame(results)
    
    return comparison_df


def sentiment_over_time(reviews_df, date_column='unixReviewTime', text_column='reviewText'):
    """
    Analyze sentiment trends over time
    
    Args:
        reviews_df: DataFrame with reviews
        date_column: Column with timestamps
        text_column: Column with review text
        
    Returns:
        DataFrame with sentiment over time
    """
    # Analyze sentiment
    analyzer = SentimentAnalyzer(method='textblob')
    analyzed = analyzer.analyze_reviews_batch(reviews_df, text_column=text_column)
    
    # Convert timestamp to datetime if needed
    if date_column in analyzed.columns:
        if analyzed[date_column].dtype != 'datetime64[ns]':
            analyzed['date'] = pd.to_datetime(analyzed[date_column], unit='s')
        else:
            analyzed['date'] = analyzed[date_column]
        
        # Group by date and calculate sentiment ratio
        time_sentiment = analyzed.groupby('date').agg({
            'sentiment': lambda x: (x == 'positive').mean()
        }).reset_index()
        
        time_sentiment.columns = ['date', 'positive_ratio']
        
        return time_sentiment
    
    return analyzed


# ========================================
# TEST SENTIMENT ANALYZER
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING SENTIMENT ANALYZER")
    print("=" * 70)
    
    # Test reviews
    test_reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality. Complete waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "Love it! Highly recommend to everyone. Five stars!",
        "Broke after one day. Worst product I've ever bought.",
        "Good value for money. Works as expected.",
        "Not what I expected but still acceptable.",
        "Outstanding quality! Exceeded all my expectations!",
        "Avoid this product at all costs. Total garbage.",
        "Pretty decent. No complaints so far."
    ]
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'reviewText': test_reviews,
        'asin': ['PROD001'] * len(test_reviews)
    })
    
    # Test TextBlob
    print("\n" + "=" * 70)
    print("TEST 1: TextBlob Sentiment Analysis")
    print("=" * 70)
    
    analyzer_tb = SentimentAnalyzer(method='textblob')
    
    for i, review in enumerate(test_reviews[:3]):
        print(f"\nReview {i+1}: \"{review}\"")
        result = analyzer_tb.classify_review_sentiment(review)
        print(f"  Sentiment: {result['sentiment'].upper()}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Positive words: {result['positive_words_count']}")
        print(f"  Negative words: {result['negative_words_count']}")
    
    # Test batch analysis
    print("\n" + "=" * 70)
    print("TEST 2: Batch Analysis")
    print("=" * 70)
    
    analyzed_df = analyzer_tb.analyze_reviews_batch(test_df)
    summary = analyzer_tb.get_sentiment_summary(analyzed_df)
    
    print("\n📊 Summary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test keyword extraction
    print("\n" + "=" * 70)
    print("TEST 3: Keyword Extraction")
    print("=" * 70)
    
    sample_text = " ".join(test_reviews)
    keywords = analyzer_tb.extract_keywords(sample_text, top_n=10)
    
    print("\nTop 10 Keywords:")
    for word, freq in keywords:
        print(f"  {word}: {freq}")
    
    # Test top reviews
    print("\n" + "=" * 70)
    print("TEST 4: Top Positive/Negative Reviews")
    print("=" * 70)
    
    top_positive = analyzer_tb.get_top_positive_reviews(analyzed_df, top_n=2)
    top_negative = analyzer_tb.get_top_negative_reviews(analyzed_df, top_n=2)
    
    print("\n✅ Top 2 Positive Reviews:")
    for idx, row in top_positive.iterrows():
        print(f"  - {row['reviewText']} (Confidence: {row['sentiment_confidence']:.4f})")
    
    print("\n❌ Top 2 Negative Reviews:")
    for idx, row in top_negative.iterrows():
        print(f"  - {row['reviewText']} (Confidence: {row['sentiment_confidence']:.4f})")
    
    # Test VADER if available
    if VADER_AVAILABLE:
        print("\n" + "=" * 70)
        print("TEST 5: VADER Sentiment Analysis")
        print("=" * 70)
        
        analyzer_vader = SentimentAnalyzer(method='vader')
        
        for i, review in enumerate(test_reviews[:3]):
            print(f"\nReview {i+1}: \"{review}\"")
            result = analyzer_vader.get_sentiment_vader(review)
            print(f"  Sentiment: {result['sentiment'].upper()}")
            print(f"  Compound: {result['compound']:.4f}")
            print(f"  Pos: {result.get('positive', 0):.4f} | "
                  f"Neg: {result.get('negative', 0):.4f} | "
                  f"Neu: {result.get('neutral', 0):.4f}")
    
    # Test both methods
    print("\n" + "=" * 70)
    print("TEST 6: Combined TextBlob + VADER")
    print("=" * 70)
    
    if VADER_AVAILABLE:
        analyzer_both = SentimentAnalyzer(method='both')
        
        review = test_reviews[0]
        print(f"\nReview: \"{review}\"")
        result = analyzer_both.get_sentiment_score(review)
        print(f"\nResults:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 70)
    
    print("\n📝 Usage Example:")
    print("""
from sentiment_analyzer import SentimentAnalyzer

# Initialize
analyzer = SentimentAnalyzer(method='textblob')

# Analyze single review
result = analyzer.classify_review_sentiment("Amazing product!")

# Analyze batch
analyzed_df = analyzer.analyze_reviews_batch(reviews_df)

# Get summary
summary = analyzer.get_sentiment_summary(analyzed_df)
    """)