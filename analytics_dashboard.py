"""
Interactive analytics dashboard for fake news detection models.
Run with: streamlit run analytics_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.model_comparison import ModelAnalyzer
except ImportError:
    # Fallback to a simple results loader
    ModelAnalyzer = None

# Page configuration
st.set_page_config(
    page_title="Fake News Detection Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analyzer():
    """Load the model analyzer with caching."""
    if ModelAnalyzer is None:
        st.warning("ModelAnalyzer not available. Using fallback mode.")
        return None
    
    try:
        return ModelAnalyzer()
    except FileNotFoundError:
        st.error("Model results not found. Please train models first using the main pipeline.")
        return None
    except Exception as e:
        st.warning(f"Could not load ModelAnalyzer: {e}. Using fallback mode.")
        return None

@st.cache_data
def load_real_results():
    """Load real model results from saved files."""
    try:
        # Try to load the results summary
        results_path = 'saved_models/results_summary.joblib'
        if os.path.exists(results_path):
            results = joblib.load(results_path)
            st.success("✅ Loaded real model results!")
            return results
        else:
            st.warning("⚠️ Results summary not found. Using sample data for demonstration.")
            return load_sample_data()
    except Exception as e:
        # Common issue: importing binary packages with the wrong Python (e.g. numpy._core errors)
        msg = str(e)
        if 'numpy._core' in msg or 'ImportError' in msg or 'No module named' in msg:
            st.error(
                "⚠️ Error loading real results: a native dependency failed to import. "
                "This often happens when Streamlit is run with a different Python than the one where packages were installed.\n\n"
                "Fix: run Streamlit with the Python executable you used to install dependencies. For example:\n"
                "/opt/homebrew/bin/python3.10 -m streamlit run analytics_dashboard.py\n\n"
                "Or activate your virtualenv before running Streamlit: source venv/bin/activate && streamlit run analytics_dashboard.py"
            )
        else:
            st.warning(f"⚠️ Error loading real results: {e}. Using sample data.")
        return load_sample_data()

@st.cache_data
def load_sample_data():
    """Load sample data for testing."""
    # Create sample data if real data not available
    np.random.seed(42)
    sample_data = {
        'logistic_regression': {
            'accuracy': 0.87,
            'precision': 0.85,
            'recall': 0.89,
            'f1_score': 0.87,
            'training_time': 12.5,
            'y_test': np.random.choice([0, 1], 100),
            'y_pred': np.random.choice([0, 1], 100),
            'y_proba': np.random.uniform(0, 1, 100)
        },
        'random_forest': {
            'accuracy': 0.91,
            'precision': 0.89,
            'recall': 0.93,
            'f1_score': 0.91,
            'training_time': 45.2,
            'y_test': np.random.choice([0, 1], 100),
            'y_pred': np.random.choice([0, 1], 100),
            'y_proba': np.random.uniform(0, 1, 100)
        },
        'svm': {
            'accuracy': 0.88,
            'precision': 0.86,
            'recall': 0.90,
            'f1_score': 0.88,
            'training_time': 32.1,
            'y_test': np.random.choice([0, 1], 100),
            'y_pred': np.random.choice([0, 1], 100),
            'y_proba': np.random.uniform(0, 1, 100)
        }
    }
    return sample_data

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detection Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load analyzer and results
    analyzer = load_analyzer()
    results = load_real_results()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">🎛️ Dashboard Controls</div>', unsafe_allow_html=True)
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        list(results.keys()),
        default=list(results.keys())
    )
    
    # Visualization type
    viz_type = st.sidebar.selectbox(
        "Choose Visualization",
        ["Overview", "Performance Metrics", "Confusion Matrices", "Model Comparison"]
    )
    
    # Main content area
    if not selected_models:
        st.warning("Please select at least one model to analyze.")
        return
    
    if viz_type == "Overview":
        show_overview(results, selected_models)
    elif viz_type == "Performance Metrics":
        show_performance_metrics(results, selected_models)
    elif viz_type == "Confusion Matrices":
        show_confusion_matrices(results, selected_models)
    elif viz_type == "Model Comparison":
        show_model_comparison(results, selected_models)

def show_overview(results, selected_models):
    """Display overview dashboard."""
    st.header("📊 Model Performance Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate best metrics
    # Support both 'test_accuracy' and 'accuracy' depending on saving
    def _get_accuracy(r):
        return r.get('test_accuracy') or r.get('accuracy') or 0.0

    best_accuracy = max([_get_accuracy(results[model]) for model in selected_models])
    best_f1 = max([results[model].get('f1_score', 0.0) for model in selected_models])
    avg_training_time = np.mean([results[model].get('training_time', 0.0) for model in selected_models])
    total_models = len(selected_models)
    
    with col1:
        st.metric(
            label="🎯 Best Accuracy",
            value=f"{best_accuracy:.3f}",
            delta=f"{(best_accuracy - 0.5):.3f} above random"
        )
    
    with col2:
        st.metric(
            label="⚡ Best F1-Score",
            value=f"{best_f1:.3f}",
            delta="Higher is better"
        )
    
    with col3:
        st.metric(
            label="⏱️ Avg Training Time",
            value=f"{avg_training_time:.1f}s",
            delta="Per model"
        )
    
    with col4:
        st.metric(
            label="🤖 Models Analyzed",
            value=str(total_models),
            delta="Active models"
        )
    
    # Performance comparison chart
    st.subheader("📈 Performance Comparison")
    
    metrics_df = pd.DataFrame([
        {
            'Model': model.replace('_', ' ').title(),
            'Accuracy': _get_accuracy(results[model]),
            'Precision': results[model]['precision'],
            'Recall': results[model]['recall'],
            'F1-Score': results[model]['f1_score']
        }
        for model in selected_models
    ])
    
    # Create radar chart
    fig = go.Figure()
    
    for _, row in metrics_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    st.dataframe(metrics_df, use_container_width=True)

def show_performance_metrics(results, selected_models):
    """Display detailed performance metrics."""
    st.header("📊 Performance Metrics Analysis")
    
    # Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # helper to retrieve metric values defensively
    def _get_metric(model_name, metric_name):
        if metric_name == 'accuracy':
            return results[model_name].get('test_accuracy') or results[model_name].get('accuracy') or 0.0
        return results[model_name].get(metric_name, 0.0)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score']
    )
    
    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        values = [_get_metric(model, metric) for model in selected_models]
        model_names = [model.replace('_', ' ').title() for model in selected_models]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                name=metric.title(),
                marker_color=[colors[j % len(colors)] for j in range(len(selected_models))],
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training time comparison
    st.subheader("⏱️ Training Time Analysis")
    
    training_times = [results[model]['training_time'] for model in selected_models]
    model_names = [model.replace('_', ' ').title() for model in selected_models]
    
    fig_time = go.Figure(data=go.Bar(
        x=model_names,
        y=training_times,
        marker_color='lightcoral',
        text=[f'{t:.1f}s' for t in training_times],
        textposition='auto'
    ))
    
    fig_time.update_layout(
        title='Model Training Time Comparison',
        xaxis_title='Models',
        yaxis_title='Training Time (seconds)'
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

def show_confusion_matrices(results, selected_models):
    """Display confusion matrices."""
    st.header("🔢 Confusion Matrix Analysis")
    
    cols = st.columns(len(selected_models))
    
    for i, model in enumerate(selected_models):
        with cols[i]:
            if 'y_test' in results[model] and 'y_pred' in results[model]:
                cm = confusion_matrix(results[model]['y_test'], results[model]['y_pred'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Fake', 'Predicted Real'],
                    y=['Actual Fake', 'Actual Real'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    hoverangles="skip"
                ))
                
                fig.update_layout(
                    title=f'{model.replace("_", " ").title()}',
                    width=300, height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_model_comparison(results, selected_models):
    """Display simplified model comparison."""
    st.header("⚔️ Model Comparison")
    
    # Ranking table
    st.subheader("🏆 Model Rankings")
    
    ranking_data = []
    for model in selected_models:
        ranking_data.append({
            'Model': model.replace('_', ' ').title(),
            'Accuracy': results[model].get('test_accuracy') or results[model].get('accuracy'),
            'Precision': results[model].get('precision'),
            'Recall': results[model].get('recall'),
            'F1-Score': results[model].get('f1_score')
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Sort by F1-Score by default
    ranking_df = ranking_df.sort_values('F1-Score', ascending=False)
    ranking_df.reset_index(drop=True, inplace=True)
    ranking_df.index += 1
    
    st.dataframe(ranking_df, use_container_width=True)

if __name__ == "__main__":
    main()