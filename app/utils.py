"""
Utility functions for the Streamlit application.
"""

import joblib
import os
import sys
import numpy as np

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

def load_models():
    """Load trained models and preprocessor."""
    models = {}
    model_files = {
        'logistic_regression': 'saved_models/logistic_regression_model.joblib',
        'random_forest': 'saved_models/random_forest_model.joblib',
        'svm': 'saved_models/svm_model.joblib'
    }
    
    # Load models
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load preprocessor
    preprocessor_path = 'saved_models/preprocessor.joblib'
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    else:
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    return models, preprocessor

def predict_article(article_text, model, preprocessor):
    """
    Predict whether an article is fake or real.
    
    Args:
        article_text (str): The article text to classify
        model: Trained ML model
        preprocessor: Fitted text preprocessor
        
    Returns:
        tuple: (prediction, confidence, probabilities)
    """
    # Clean and preprocess the text
    cleaned_text = preprocessor.clean_text(article_text)
    
    # Transform to features
    article_features = preprocessor.transform_texts([cleaned_text])
    
    # Make prediction
    prediction_numeric = model.predict(article_features)[0]
    prediction_proba = model.predict_proba(article_features)[0]
    
    # Convert to human-readable format
    prediction = "Real" if prediction_numeric == 1 else "Fake"
    
    # Calculate confidence (probability of predicted class)
    confidence = prediction_proba[prediction_numeric]
    
    return prediction, confidence, prediction_proba

def get_prediction_explanation(prediction, confidence):
    """
    Generate an explanation for the prediction.
    
    Args:
        prediction (str): "Real" or "Fake"
        confidence (float): Confidence score
        
    Returns:
        str: Explanation text
    """
    if prediction == "Real":
        if confidence > 0.9:
            return """
            🟢 **High Confidence - Likely Real News**
            
            The model is very confident that this article is legitimate news. The text patterns, 
            language style, and content structure are consistent with authentic journalism. 
            However, always verify information from multiple reliable sources.
            """
        elif confidence > 0.7:
            return """
            🔵 **Medium Confidence - Probably Real News**
            
            The model believes this is likely real news, but with moderate confidence. 
            The article shows characteristics of legitimate journalism, though some 
            elements might be ambiguous. Consider checking additional sources for verification.
            """
        else:
            return """
            🟡 **Low Confidence - Uncertain**
            
            The model leans toward classifying this as real news, but with low confidence. 
            The article contains mixed signals that make classification difficult. 
            Exercise extra caution and verify through multiple reliable sources.
            """
    else:  # Fake
        if confidence > 0.9:
            return """
            🔴 **High Confidence - Likely Fake News**
            
            The model is very confident that this article contains misinformation or 
            misleading content. The text shows strong patterns associated with fake news, 
            such as sensational language, lack of credible sources, or suspicious claims.
            """
        elif confidence > 0.7:
            return """
            🟠 **Medium Confidence - Probably Fake News**
            
            The model believes this is likely fake or misleading news. The article 
            exhibits several characteristics commonly found in misinformation, but 
            some elements are ambiguous. Fact-check before sharing or believing.
            """
        else:
            return """
            🟡 **Low Confidence - Uncertain**
            
            The model leans toward classifying this as fake news, but with low confidence. 
            The article contains mixed signals. While there may be concerning elements, 
            verification through fact-checking websites and reliable sources is recommended.
            """

def get_article_statistics(article_text):
    """
    Calculate basic statistics about the article.
    
    Args:
        article_text (str): The article text
        
    Returns:
        dict: Dictionary containing article statistics
    """
    words = article_text.split()
    sentences = article_text.split('.')
    
    stats = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'character_count': len(article_text),
        'exclamation_marks': article_text.count('!'),
        'question_marks': article_text.count('?'),
        'uppercase_ratio': sum(1 for c in article_text if c.isupper()) / max(len(article_text), 1)
    }
    
    return stats

def analyze_text_features(article_text):
    """
    Analyze text features that might indicate fake news.
    
    Args:
        article_text (str): The article text
        
    Returns:
        dict: Dictionary containing feature analysis
    """
    # Suspicious words/phrases often found in fake news
    suspicious_words = [
        'breaking', 'shocking', 'unbelievable', 'you won\'t believe',
        'secret', 'hidden truth', 'they don\'t want you to know',
        'wake up', 'share before', 'deleted', 'banned', 'censored'
    ]
    
    # Emotional words
    emotional_words = [
        'amazing', 'terrible', 'horrible', 'incredible', 'outrageous',
        'disgusting', 'brilliant', 'devastating', 'shocking', 'stunning'
    ]
    
    article_lower = article_text.lower()
    
    features = {
        'suspicious_word_count': sum(1 for word in suspicious_words if word in article_lower),
        'emotional_word_count': sum(1 for word in emotional_words if word in article_lower),
        'all_caps_words': sum(1 for word in article_text.split() if word.isupper() and len(word) > 2),
        'exclamation_density': article_text.count('!') / max(len(article_text.split()), 1),
        'question_density': article_text.count('?') / max(len(article_text.split()), 1)
    }
    
    return features