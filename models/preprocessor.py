"""
Text preprocessing utilities for fake news detection.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing class for cleaning and preparing news articles."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
    
    def clean_text(self, text):
        """
        Clean and preprocess a single text string.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def fit_vectorizer(self, texts, max_features=10000):
        """
        Fit TF-IDF vectorizer on the training texts.
        
        Args:
            texts (list): List of cleaned texts
            max_features (int): Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english'
        )
        self.vectorizer.fit(texts)
    
    def transform_texts(self, texts):
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts (list): List of cleaned texts
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names from the fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        
        return self.vectorizer.get_feature_names_out()

def load_and_preprocess_data(true_path, fake_path):
    """
    Load and preprocess the fake news dataset.
    
    Args:
        true_path (str): Path to true news CSV file
        fake_path (str): Path to fake news CSV file
        
    Returns:
        tuple: (features, labels, preprocessor)
    """
    # Load data
    try:
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {e}")
    
    # Add labels
    true_df['label'] = 1  # Real news
    fake_df['label'] = 0  # Fake news
    
    # Combine datasets
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean text data
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Real news articles: {df[df['label'] == 1].shape[0]}")
    print(f"Fake news articles: {df[df['label'] == 0].shape[0]}")
    
    return df['cleaned_text'].tolist(), df['label'].tolist(), preprocessor