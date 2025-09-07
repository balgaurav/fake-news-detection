"""
Streamlit web application for fake news detection.
"""

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import load_models, predict_article, get_prediction_explanation

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .real-news {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .fake-news {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .confidence-meter {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses machine learning to classify news articles as **Real** or **Fake**.
    Enter a news article below to get an instant prediction with confidence scores.
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Configuration")
        
        # Model selection
        available_models = ['random_forest', 'logistic_regression', 'svm']
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0,
            help="Choose which trained model to use for predictions"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Performance")
        try:
            results = joblib.load('saved_models/results_summary.joblib')
            if selected_model in results:
                model_results = results[selected_model]
                # Support multiple possible key names depending on how results were saved
                accuracy = model_results.get('test_accuracy') or model_results.get('accuracy')
                if accuracy is not None:
                    st.metric("Accuracy", f"{accuracy:.1%}")
                else:
                    st.metric("Accuracy", "N/A")

                precision = model_results.get('precision')
                f1 = model_results.get('f1_score')
                if precision is not None:
                    st.metric("Precision", f"{precision:.1%}")
                else:
                    st.metric("Precision", "N/A")

                if f1 is not None:
                    st.metric("F1-Score", f"{f1:.1%}")
                else:
                    st.metric("F1-Score", "N/A")
        except Exception:
            st.warning("Model results not found or could not be read. Please ensure saved_models/results_summary.joblib exists and is readable.")
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses machine learning to detect fake news articles.
        
        **How it works:**
        - Analyzes text patterns and language
        - Uses three different ML algorithms
        - Provides confidence scores
        """)
    
    # Load models
    try:
        models, preprocessor = load_models()
        st.success(f"✅ Models loaded successfully! Using: {selected_model.replace('_', ' ').title()}")
    except Exception as e:
        st.error("❌ Error loading models. Please train the models first by running: `python models/model_trainer.py`")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter News Article")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Example Articles"],
            horizontal=True
        )
        
        if input_method == "Type/Paste Text":
            article_text = st.text_area(
                "News Article Text",
                height=300,
                placeholder="Paste your news article here...",
                help="Enter the full text of the news article you want to analyze"
            )
        else:
            # Simplified examples
            examples = {
                "Suspicious News Example": """
                BREAKING: Scientists have discovered that eating chocolate for breakfast 
                can make you lose 20 pounds in one week! Doctors are amazed by this 
                simple trick that the weight loss industry doesn't want you to know!
                """,
                "Legitimate News Example": """
                The Federal Reserve announced a 0.25 percentage point increase in interest 
                rates following their meeting this week. The decision reflects ongoing efforts 
                to combat inflation while supporting economic growth.
                """
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            article_text = examples[selected_example]
            st.text_area("Selected Example:", value=article_text, height=200, disabled=True)
        
        # Prediction button
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if article_text.strip():
                with st.spinner("Analyzing article..."):
                    # Make prediction
                    prediction, confidence, probabilities = predict_article(
                        article_text, models[selected_model], preprocessor
                    )
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Create prediction box
                    if prediction == "Real":
                        st.markdown(f'''
                        <div class="prediction-box real-news">
                            <h3>✅ REAL NEWS</h3>
                            <div class="confidence-meter">
                                Confidence: {confidence:.1%}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="prediction-box fake-news">
                            <h3>❌ FAKE NEWS</h3>
                            <div class="confidence-meter">
                                Confidence: {confidence:.1%}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Confidence breakdown
                    st.markdown("### 📈 Confidence Score")
                    
                    # Simple confidence bar
                    st.progress(confidence)
                    st.write(f"**Confidence: {confidence:.1%}**")
                    
                    if confidence > 0.8:
                        st.success("High confidence prediction")
                    elif confidence > 0.6:
                        st.warning("Medium confidence prediction")
                    else:
                        st.error("Low confidence - result may be uncertain")
                    
                    # Explanation
                    explanation = get_prediction_explanation(prediction, confidence)
                    st.markdown("### 💡 Explanation")
                    st.info(explanation)
                    
                    # Save prediction log
                    log_prediction(article_text[:100], prediction, confidence, selected_model)
            
            else:
                st.warning("⚠️ Please enter a news article to analyze.")
    
    with col2:
        st.markdown("### 📊 Model Performance")
        
        # Display model comparison if results are available
        try:
            results = joblib.load('saved_models/results_summary.joblib')
            
            # Simple metrics display
            for model_name, model_results in results.items():
                accuracy = model_results.get('test_accuracy', 0)
                st.metric(
                    f"{model_name.replace('_', ' ').title()}",
                    f"{accuracy:.1%} accurate"
                )
            
        except:
            st.warning("📊 Model performance data not available.")
        
        st.markdown("### 📈 How It Works")
        st.markdown("""
        **1. Text Processing**
        - Cleans and prepares the article text
        - Removes unnecessary characters and words
        
        **2. Feature Extraction**  
        - Converts text into numerical features
        - Identifies important word patterns
        
        **3. Machine Learning**
        - Three algorithms analyze the text
        - Each provides a prediction and confidence
        """)

def log_prediction(article_preview, prediction, confidence, model_used):
    """Log prediction for statistics."""
    try:
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'article_preview': article_preview,
            'prediction': prediction,
            'confidence': confidence,
            'model_used': model_used
        }
        
        # Create or append to log file
        if os.path.exists('prediction_logs.csv'):
            logs_df = pd.read_csv('prediction_logs.csv')
            logs_df = pd.concat([logs_df, pd.DataFrame([log_data])], ignore_index=True)
        else:
            logs_df = pd.DataFrame([log_data])
        
        logs_df.to_csv('prediction_logs.csv', index=False)
        
    except Exception as e:
        st.error(f"Error logging prediction: {e}")

if __name__ == "__main__":
    main()