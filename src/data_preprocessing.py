"""
Data Preprocessing Module
Handles loading, cleaning, and feature engineering for Amazon reviews
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.num_users = 0
        self.num_items = 0
        
    def load_data(self):
        """Load the Amazon reviews dataset"""
        print("\n" + "=" * 70)
        print("📂 STEP 1: LOADING DATA")
        print("=" * 70)
        
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✅ Loaded {len(self.df):,} reviews")
            print(f"\nColumns: {list(self.df.columns)}")
            print(f"\nFirst few rows:")
            print(self.df.head())
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.filepath}")
            print(f"   Please make sure amazon_reviews.csv is in data/raw/ folder")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def clean_data(self):
        print("\n" + "=" * 70)
        print("🧹 STEP 2: DATA CLEANING")
        print("=" * 70)
        
        initial_rows = len(self.df)
    
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")
    
        # Handle missing values
        print("\nHandling missing values...")
    
        # Fill missing text fields
        if 'reviewText' in self.df.columns:
          self.df['reviewText'] = self.df['reviewText'].fillna('')
        if 'summary' in self.df.columns:
          self.df['summary'] = self.df['summary'].fillna('')
    
        # Handle helpful column - it might be in format "[1, 2]" meaning 1 helpful out of 2 total
        if 'helpful' in self.df.columns and 'helpful_yes' not in self.df.columns:
            # Parse helpful column (format: "[helpful_yes, total_vote]")
           try:
               import ast
               self.df['helpful_parsed'] = self.df['helpful'].apply(
                   lambda x: ast.literal_eval(x) if isinstance(x, str) else [0, 0]
            )
               self.df['helpful_yes'] = self.df['helpful_parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
               self.df['total_vote'] = self.df['helpful_parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
               self.df = self.df.drop('helpful_parsed', axis=1)
           except:
            self.df['helpful_yes'] = 0
            self.df['total_vote'] = 0
    
        # Fill missing numeric fields
        if 'helpful_yes' in self.df.columns:
         self.df['helpful_yes'] = self.df['helpful_yes'].fillna(0)
        if 'total_vote' in self.df.columns:
         self.df['total_vote'] = self.df['total_vote'].fillna(0)
        if 'day_diff' in self.df.columns:
         self.df['day_diff'] = self.df['day_diff'].fillna(self.df['day_diff'].median())
    
        # Remove rows with missing ratings
        if 'overall' in self.df.columns:
         self.df = self.df.dropna(subset=['overall'])
    
        # Validate ratings (1-5 scale)
        self.df = self.df[(self.df['overall'] >= 1) & (self.df['overall'] <= 5)]
    
        # Remove empty reviews
        if 'reviewText' in self.df.columns:
         self.df = self.df[self.df['reviewText'].str.len() > 0]
    
        # Create user_id and item_id from reviewerID and asin
        if 'reviewerID' in self.df.columns:
        # Convert reviewerID to numeric user_id
         unique_users = self.df['reviewerID'].unique()
         user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
         self.df['user_id'] = self.df['reviewerID'].map(user_mapping)
        else:
          self.df['user_id'] = range(len(self.df))
    
        if 'asin' in self.df.columns:
        # Convert asin to numeric item_id
         unique_items = self.df['asin'].unique()
         item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}
         self.df['item_id'] = self.df['asin'].map(item_mapping)
        else:
         self.df['item_id'] = (self.df['overall'] * 1000 + 
                             (self.df['helpful_yes'] % 100)).astype(int)
    
        print(f"✅ Cleaned data: {len(self.df):,} reviews remaining")
        print(f"   Users: {self.df['user_id'].nunique():,}")
        print(f"   Items: {self.df['item_id'].nunique():,}")
    
        return self.df
    
    def create_features(self):
        """Engineer features for recommendation"""
        print("\n" + "=" * 70)
        print("🛠️ STEP 3: FEATURE ENGINEERING")
        print("=" * 70)
        
        # Create user and item IDs
        # Since dataset doesn't have user_id, we create synthetic ones
        print("\n1️⃣ Creating user and item identifiers...")
        self.df['user_id'] = range(len(self.df))
        
        # Create item_id based on rating patterns and helpfulness
        self.df['item_id'] = (
            self.df['overall'].astype(int) * 1000 + 
            (self.df['helpful_yes'].fillna(0) % 100).astype(int)
        )
        
        self.num_users = self.df['user_id'].nunique()
        self.num_items = self.df['item_id'].nunique()
        
        print(f"   Created {self.num_users:,} unique users")
        print(f"   Created {self.num_items:,} unique items")
        
        # Binary target: positive engagement (rating >= 4)
        print("\n2️⃣ Creating target variable...")
        self.df['is_positive'] = (self.df['overall'] >= 4).astype(int)
        positive_pct = self.df['is_positive'].mean() * 100
        print(f"   Positive reviews: {self.df['is_positive'].sum():,} ({positive_pct:.1f}%)")
        
        # Normalized rating (0-1 scale)
        self.df['rating_normalized'] = (self.df['overall'] - 1) / 4
        
        # Text features
        print("\n3️⃣ Creating text-based features...")
        if 'reviewText' in self.df.columns:
            self.df['review_length'] = self.df['reviewText'].str.len()
            self.df['review_word_count'] = self.df['reviewText'].str.split().str.len()
        if 'summary' in self.df.columns:
            self.df['summary_length'] = self.df['summary'].str.len()
        
        # Helpfulness ratio
        print("\n4️⃣ Creating helpfulness features...")
        self.df['helpfulness_ratio'] = np.where(
            self.df['total_vote'] > 0,
            self.df['helpful_yes'] / self.df['total_vote'],
            0
        )
        
        # Recency features
        print("\n5️⃣ Creating recency features...")
        if 'day_diff' in self.df.columns:
            self.df['recency_score'] = 1 / (1 + self.df['day_diff'] / 30)
            self.df['is_recent'] = (self.df['day_diff'] <= 30).astype(int)
        else:
            self.df['recency_score'] = 0.5
            self.df['is_recent'] = 0
        
        # Engagement score (composite metric)
        print("\n6️⃣ Creating engagement score...")
        self.df['engagement_score'] = (
            0.5 * self.df['rating_normalized'] +
            0.3 * self.df['helpfulness_ratio'] +
            0.2 * self.df['recency_score']
        )
        
        print(f"\n✅ Feature engineering complete!")
        print(f"   Total features: {self.df.shape[1]}")
        
        return self.df
    
    def create_user_features(self):
        """Create user-level aggregated features"""
        print("\n7️⃣ Creating USER features...")
        
        user_features = self.df.groupby('user_id').agg({
            'overall': ['mean', 'std', 'count'],
            'is_positive': 'mean',
            'item_id': 'nunique',
            'helpful_yes': 'sum',
            'review_word_count': 'mean'
        }).reset_index()
        
        user_features.columns = ['user_id', 'avg_rating', 'rating_std', 'total_reviews',
                                'positive_ratio', 'unique_items', 'total_helpful_votes',
                                'avg_review_length']
        
        # Fill NaN in std with 0
        user_features['rating_std'] = user_features['rating_std'].fillna(0)
        
        print(f"   ✅ Created features for {len(user_features):,} users")
        
        return user_features
    
    def create_item_features(self):
        """Create item-level aggregated features"""
        print("\n8️⃣ Creating ITEM features...")
        
        item_features = self.df.groupby('item_id').agg({
            'overall': ['mean', 'std', 'count'],
            'user_id': 'nunique',
            'is_positive': 'mean',
            'helpful_yes': 'sum',
            'helpfulness_ratio': 'mean'
        }).reset_index()
        
        item_features.columns = ['item_id', 'avg_rating', 'rating_std', 'review_count',
                                'unique_users', 'positive_ratio', 'total_helpful',
                                'avg_helpfulness']
        
        # Fill NaN in std with 0
        item_features['rating_std'] = item_features['rating_std'].fillna(0)
        
        print(f"   ✅ Created features for {len(item_features):,} items")
        
        return item_features
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split into train/val/test sets"""
        print("\n" + "=" * 70)
        print("✂️ STEP 4: TRAIN/VAL/TEST SPLIT")
        print("=" * 70)
        
        # First split: train+val vs test
        train_val, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['is_positive']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.train_df, self.val_df = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val['is_positive']
        )
        
        print(f"✅ Split complete:")
        print(f"   Train: {len(self.train_df):,} ({len(self.train_df)/len(self.df)*100:.1f}%)")
        print(f"   Val:   {len(self.val_df):,} ({len(self.val_df)/len(self.df)*100:.1f}%)")
        print(f"   Test:  {len(self.test_df):,} ({len(self.test_df)/len(self.df)*100:.1f}%)")
        
        return self.train_df, self.val_df, self.test_df
    
    def save_processed_data(self, output_dir):
        """Save processed datasets"""
        print("\n" + "=" * 70)
        print("💾 STEP 5: SAVING PROCESSED DATA")
        print("=" * 70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        self.train_df.to_csv(output_dir / 'train_data.csv', index=False)
        self.val_df.to_csv(output_dir / 'val_data.csv', index=False)
        self.test_df.to_csv(output_dir / 'test_data.csv', index=False)
        
        # Save aggregated features
        user_features = self.create_user_features()
        item_features = self.create_item_features()
        
        user_features.to_csv(output_dir / 'user_features.csv', index=False)
        item_features.to_csv(output_dir / 'item_features.csv', index=False)
        
        print(f"✅ All data saved to: {output_dir}")
        print(f"   - train_data.csv")
        print(f"   - val_data.csv")
        print(f"   - test_data.csv")
        print(f"   - user_features.csv")
        print(f"   - item_features.csv")
    
    def visualize_data(self, save_path=None):
        """Create exploratory visualizations"""
        print("\n" + "=" * 70)
        print("📊 STEP 6: DATA VISUALIZATION")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Amazon Reviews - Data Exploration', fontsize=16, fontweight='bold')
        
        # 1. Rating distribution
        self.df['overall'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], color='skyblue', edgecolor='black'
        )
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Review length distribution
        axes[0, 1].hist(self.df['review_word_count'], bins=50, color='coral', edgecolor='black')
        axes[0, 1].set_title('Review Length Distribution')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlim(0, 500)
        
        # 3. Helpfulness ratio
        axes[0, 2].hist(self.df['helpfulness_ratio'], bins=50, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Helpfulness Ratio')
        axes[0, 2].set_xlabel('Ratio')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Positive vs Negative reviews
        labels = ['Negative (1-3)', 'Positive (4-5)']
        colors = ['#ff9999', '#66b3ff']
        self.df['is_positive'].value_counts().plot(
            kind='pie', ax=axes[1, 0], autopct='%1.1f%%',
            labels=labels, colors=colors, startangle=90
        )
        axes[1, 0].set_title('Positive vs Negative Reviews')
        axes[1, 0].set_ylabel('')
        
        # 5. Engagement score distribution
        axes[1, 1].hist(self.df['engagement_score'], bins=50, color='mediumpurple', edgecolor='black')
        axes[1, 1].set_title('Engagement Score Distribution')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Recency vs Rating (if day_diff exists)
        if 'day_diff' in self.df.columns:
            sample = self.df.sample(min(1000, len(self.df)))
            axes[1, 2].scatter(sample['day_diff'], sample['overall'], alpha=0.3, s=10)
            axes[1, 2].set_title('Recency vs Rating')
            axes[1, 2].set_xlabel('Days Since Review')
            axes[1, 2].set_ylabel('Rating')
        else:
            axes[1, 2].text(0.5, 0.5, 'No recency data', ha='center', va='center')
            axes[1, 2].set_title('Recency vs Rating')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        
        return fig
    
    def get_stats(self):
        """Get dataset statistics"""
        stats = {
            'total_reviews': len(self.df),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'positive_rate': self.df['is_positive'].mean(),
            'avg_rating': self.df['overall'].mean(),
            'avg_review_length': self.df['review_word_count'].mean() if 'review_word_count' in self.df.columns else 0
        }
        return stats


# Test the preprocessor
if __name__ == "__main__":
    from src.config import Config
    
    print("Testing Data Preprocessor...")
    
    # Initialize
    preprocessor = DataPreprocessor(Config.RAW_DATA_DIR / 'amazon_reviews.csv')
    
    # Run pipeline
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.create_features()
    preprocessor.split_data()
    preprocessor.save_processed_data(Config.PROCESSED_DATA_DIR)
    preprocessor.visualize_data(Config.OUTPUTS_DIR / 'data_exploration.png')
    
    # Print stats
    stats = preprocessor.get_stats()
    print("\n" + "=" * 70)
    print("📊 DATASET STATISTICS")
    print("=" * 70)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Preprocessing complete!")