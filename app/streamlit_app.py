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
                st.metric("Accuracy", f"{model_results['test_accuracy']:.1%}")
                st.metric("Precision", f"{model_results['precision']:.1%}")
                st.metric("F1-Score", f"{model_results['f1_score']:.1%}")
        except:
            st.warning("Model results not found. Please train models first.")
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system analyzes text patterns to detect potentially fake news articles.
        
        **Features:**
        - Multiple ML models
        - Real-time predictions
        - Confidence scoring
        - Performance visualization
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
            # Example articles
            examples = {
                "Real News Example": """
                Scientists at MIT have developed a new breakthrough in renewable energy technology. 
                The research team, led by Dr. Sarah Johnson, has created a more efficient solar panel 
                that can generate 40% more electricity than traditional panels. The study, published 
                in Nature Energy, shows promising results for large-scale implementation. The technology 
                uses advanced materials and could significantly reduce the cost of solar power generation.
                """,
                "Fake News Example": """
                BREAKING: Government secretly plans to replace all birds with drones by 2025! 
                Insider sources reveal shocking truth about surveillance program. Birds aren't real - 
                they're all government spy drones! Wake up people! Share this before they delete it! 
                The mainstream media won't tell you this because they're in on it too! 
                #BirdsArentReal #GovernmentConspiracy
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
                    st.markdown("### 📈 Confidence Breakdown")
                    
                    # Create gauge chart for confidence
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Prediction Confidence (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Probability distribution
                    prob_df = pd.DataFrame({
                        'Classification': ['Fake News', 'Real News'],
                        'Probability': [probabilities[0], probabilities[1]]
                    })
                    
                    fig_bar = px.bar(
                        prob_df, 
                        x='Classification', 
                        y='Probability',
                        color='Classification',
                        color_discrete_map={'Fake News': '#dc3545', 'Real News': '#28a745'},
                        title="Classification Probabilities"
                    )
                    fig_bar.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Explanation
                    explanation = get_prediction_explanation(prediction, confidence)
                    st.markdown("### 💡 Explanation")
                    st.info(explanation)
                    
                    # Save prediction log
                    log_prediction(article_text[:100], prediction, confidence, selected_model)
            
            else:
                st.warning("⚠️ Please enter a news article to analyze.")
    
    with col2:
        st.markdown("### 📊 Model Comparison")
        
        # Display model comparison if results are available
        try:
            results = joblib.load('saved_models/results_summary.joblib')
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, model_results in results.items():
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': model_results['test_accuracy'],
                    'Precision': model_results['precision'],
                    'F1-Score': model_results['f1_score']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Bar chart comparison
            fig_comparison = px.bar(
                df_comparison, 
                x='Model', 
                y='Accuracy',
                title='Model Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            fig_comparison.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Metrics table
            st.markdown("#### Model Metrics")
            styled_df = df_comparison.style.format({
                'Accuracy': '{:.1%}',
                'Precision': '{:.1%}',
                'F1-Score': '{:.1%}'
            }).background_gradient(subset=['Accuracy', 'Precision', 'F1-Score'])
            
            st.dataframe(styled_df, use_container_width=True)
            
        except:
            st.warning("📊 Model comparison data not available. Train models to see comparison.")
        
        st.markdown("### 📈 Usage Statistics")
        
        # Display prediction logs if available
        try:
            if os.path.exists('prediction_logs.csv'):
                logs_df = pd.read_csv('prediction_logs.csv')
                
                # Recent predictions count
                st.metric("Total Predictions", len(logs_df))
                
                # Prediction distribution
                pred_counts = logs_df['prediction'].value_counts()
                fig_pie = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Prediction Distribution"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            else:
                st.info("No prediction history available yet.")
                
        except:
            st.info("Prediction logs not available.")

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