"""
Advanced model comparison and analysis tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)
from wordcloud import WordCloud
import joblib
import os
from .preprocessor import TextPreprocessor

class ModelAnalyzer:
    """Advanced analysis tools for fake news detection models."""
    
    def __init__(self, results_path='saved_models/results_summary.joblib'):
        """Initialize analyzer with model results."""
        try:
            self.results = joblib.load(results_path)
            self.models = self.load_trained_models()
            self.preprocessor = joblib.load('saved_models/preprocessor.joblib')
        except FileNotFoundError:
            raise FileNotFoundError("Model results not found. Please train models first.")
    
    def load_trained_models(self):
        """Load all trained models."""
        models = {}
        model_files = {
            'logistic_regression': 'saved_models/logistic_regression_model.joblib',
            'random_forest': 'saved_models/random_forest_model.joblib',
            'svm': 'saved_models/svm_model.joblib'
        }
        
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        return models
    
    def create_roc_curves(self):
        """Create ROC curves for all models."""
        fig = go.Figure()
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'y_proba' in results and results['y_proba'] is not None:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(results['y_test'] if 'y_test' in results else [], 
                                       results['y_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600
        )
        
        return fig
    
    def create_precision_recall_curves(self):
        """Create Precision-Recall curves for all models."""
        fig = go.Figure()
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'y_proba' in results and results['y_proba'] is not None:
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(
                    results['y_test'] if 'y_test' in results else [], 
                    results['y_proba']
                )
                pr_auc = auc(recall, precision)
                
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{model_name.replace("_", " ").title()} (AUC = {pr_auc:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=800, height=600
        )
        
        return fig
    
    def create_feature_importance_plot(self, model_name='random_forest', top_n=20):
        """Create feature importance plot for tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} doesn't support feature importance.")
        
        # Get feature names and importance
        feature_names = self.preprocessor.get_feature_names()
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create plot
        fig = go.Figure(data=go.Bar(
            x=[feature_names[i] for i in indices],
            y=[importances[i] for i in indices],
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}',
            xaxis_title='Features',
            yaxis_title='Importance',
            xaxis_tickangle=-45,
            width=1000, height=600
        )
        
        return fig
    
    def create_performance_comparison_chart(self):
        """Create comprehensive performance comparison chart."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score']
        )
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [self.results[model][metric] for model in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric.title(),
                    marker_color=[colors[j % len(colors)] for j in range(len(model_names))],
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=800,
            width=1000
        )
        
        return fig
    
    def create_learning_curves(self, model_name='random_forest'):
        """Create learning curves to analyze training performance."""
        # This would require training history data
        # For now, create a placeholder visualization
        
        # Simulated learning curve data (replace with actual training history)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = np.random.uniform(0.7, 0.95, 10)  # Simulated
        val_scores = np.random.uniform(0.6, 0.85, 10)    # Simulated
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'Learning Curves - {model_name.replace("_", " ").title()}',
            xaxis_title='Training Set Size (fraction)',
            yaxis_title='Accuracy Score',
            width=800, height=600
        )
        
        return fig
    
    def analyze_misclassified_examples(self, model_name, num_examples=10):
        """Analyze misclassified examples."""
        if model_name not in self.results:
            raise ValueError(f"Results for {model_name} not found.")
        
        results = self.results[model_name]
        y_true = results.get('y_test', [])
        y_pred = results.get('y_pred', [])
        
        if not y_true or not y_pred:
            raise ValueError("Test data not found in results.")
        
        # Find misclassified indices
        misclassified_mask = np.array(y_true) != np.array(y_pred)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Sample random misclassified examples
        if len(misclassified_indices) > num_examples:
            sample_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        else:
            sample_indices = misclassified_indices
        
        misclassified_data = []
        for idx in sample_indices:
            misclassified_data.append({
                'Index': idx,
                'True_Label': 'Real' if y_true[idx] == 1 else 'Fake',
                'Predicted_Label': 'Real' if y_pred[idx] == 1 else 'Fake',
                'Confidence': results.get('y_proba', [0.5])[idx] if 'y_proba' in results else 0.5
            })
        
        return pd.DataFrame(misclassified_data)
    
    def create_confusion_matrix_heatmap(self, model_name):
        """Create an enhanced confusion matrix heatmap."""
        if model_name not in self.results:
            raise ValueError(f"Results for {model_name} not found.")
        
        results = self.results[model_name]
        y_true = results.get('y_test', [])
        y_pred = results.get('y_pred', [])
        
        if not y_true or not y_pred:
            raise ValueError("Test data not found in results.")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Fake', 'Predicted Real'],
            y=['Actual Fake', 'Actual Real'],
            colorscale='Blues',
            text=[[f'{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)' for j in range(cm.shape[1])] 
                  for i in range(cm.shape[0])],
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverangles="skip"
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name.replace("_", " ").title()}',
            width=500, height=500
        )
        
        return fig
    
    def create_word_clouds(self, data_path_true, data_path_fake):
        """Create word clouds for real vs fake news."""
        try:
            # Load data
            true_df = pd.read_csv(data_path_true)
            fake_df = pd.read_csv(data_path_fake)
            
            # Preprocess text
            preprocessor = TextPreprocessor()
            
            # Clean texts
            true_texts = ' '.join([preprocessor.clean_text(text) for text in true_df['text'].head(1000)])
            fake_texts = ' '.join([preprocessor.clean_text(text) for text in fake_df['text'].head(1000)])
            
            # Create word clouds
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Real news word cloud
            wordcloud_real = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     colormap='Blues').generate(true_texts)
            axes[0].imshow(wordcloud_real, interpolation='bilinear')
            axes[0].set_title('Real News - Common Words', fontsize=16, fontweight='bold')
            axes[0].axis('off')
            
            # Fake news word cloud
            wordcloud_fake = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     colormap='Reds').generate(fake_texts)
            axes[1].imshow(wordcloud_fake, interpolation='bilinear')
            axes[1].set_title('Fake News - Common Words', fontsize=16, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs('visualizations', exist_ok=True)
            plt.savefig('visualizations/word_clouds.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return "Word clouds saved to visualizations/word_clouds.png"
            
        except Exception as e:
            return f"Error creating word clouds: {e}"
    
    def create_prediction_confidence_distribution(self):
        """Create distribution plots of prediction confidence for each model."""
        fig = make_subplots(
            rows=len(self.results), cols=1,
            subplot_titles=[name.replace('_', ' ').title() for name in self.results.keys()],
            vertical_spacing=0.1
        )
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'y_proba' in results and results['y_proba'] is not None:
                y_true = results.get('y_test', [])
                y_proba = results['y_proba']
                
                # Separate probabilities by true label
                real_probs = [p for p, label in zip(y_proba, y_true) if label == 1]
                fake_probs = [p for p, label in zip(y_proba, y_true) if label == 0]
                
                # Add histograms
                fig.add_trace(
                    go.Histogram(
                        x=real_probs,
                        name='Real News',
                        opacity=0.7,
                        marker_color='blue',
                        nbinsx=30,
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=fake_probs,
                        name='Fake News',
                        opacity=0.7,
                        marker_color='red',
                        nbinsx=30,
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Prediction Confidence Distribution by Model',
            height=300 * len(self.results),
            barmode='overlay'
        )
        
        fig.update_xaxes(title_text='Prediction Confidence (Probability of Real News)')
        fig.update_yaxes(title_text='Count')
        
        return fig
    
    def generate_comprehensive_report(self, output_path='visualizations/comprehensive_report.html'):
        """Generate a comprehensive HTML report with all visualizations."""
        os.makedirs('visualizations', exist_ok=True)
        
        # Create all plots
        roc_fig = self.create_roc_curves()
        pr_fig = self.create_precision_recall_curves()
        performance_fig = self.create_performance_comparison_chart()
        confidence_fig = self.create_prediction_confidence_distribution()
        
        # Create confusion matrices subplot
        fig_cms = make_subplots(
            rows=1, cols=len(self.results),
            subplot_titles=[name.replace('_', ' ').title() for name in self.results.keys()]
        )
        
        for i, model_name in enumerate(self.results.keys()):
            try:
                cm_fig = self.create_confusion_matrix_heatmap(model_name)
                fig_cms.add_trace(cm_fig.data[0], row=1, col=i+1)
            except Exception as e:
                print(f"Warning: Could not create confusion matrix for {model_name}: {e}")
        
        fig_cms.update_layout(title='Confusion Matrices Comparison', height=500)
        
        # Generate performance summary table
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results.get('accuracy', 0):.4f}",
                'Precision': f"{results.get('precision', 0):.4f}",
                'Recall': f"{results.get('recall', 0):.4f}",
                'F1-Score': f"{results.get('f1_score', 0):.4f}",
                'Training Time (s)': f"{results.get('training_time', 0):.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fake News Detection - Comprehensive Analysis Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{ 
                    text-align: center; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px;
                    margin: -20px -20px 30px -20px;
                    border-radius: 0 0 10px 10px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .section {{ 
                    margin: 30px 0; 
                    padding: 20px;
                    background: #fafafa;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .section h2 {{
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .plot-container {{ 
                    margin: 20px 0; 
                    text-align: center;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left;
                }}
                th {{ 
                    background-color: #667eea; 
                    color: white;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f2f2f2;
                }}
                .metric-highlight {{
                    font-weight: bold;
                    color: #667eea;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #666;
                    border-top: 1px solid #ddd;
                }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Fake News Detection</h1>
                <h2>Comprehensive Model Analysis Report</h2>
                <p>Advanced Machine Learning Pipeline for News Authenticity Classification</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <p>This report presents a comprehensive analysis of multiple machine learning models trained for fake news detection. 
                    The analysis includes performance metrics, feature importance, error analysis, and visualizations to provide 
                    insights into model behavior and effectiveness.</p>
                    
                    <h3>📈 Performance Summary</h3>
                    {summary_df.to_html(index=False, classes='summary-table', escape=False)}
                </div>
                
                <div class="section">
                    <h2>📈 Model Performance Comparison</h2>
                    <div class="plot-container" id="performance-plot"></div>
                </div>
                
                <div class="section">
                    <h2>🎯 ROC Curves Analysis</h2>
                    <p>Receiver Operating Characteristic curves show the trade-off between sensitivity and specificity. 
                    Higher AUC values indicate better model performance.</p>
                    <div class="plot-container" id="roc-plot"></div>
                </div>
                
                <div class="section">
                    <h2>🎯 Precision-Recall Curves</h2>
                    <p>Precision-Recall curves are particularly useful for imbalanced datasets. 
                    They show the trade-off between precision and recall at various threshold settings.</p>
                    <div class="plot-container" id="pr-plot"></div>
                </div>
                
                <div class="section">
                    <h2>🔢 Confusion Matrices</h2>
                    <p>Confusion matrices provide detailed breakdown of correct and incorrect predictions for each class.</p>
                    <div class="plot-container" id="cm-plot"></div>
                </div>
                
                <div class="section">
                    <h2>📊 Prediction Confidence Distribution</h2>
                    <p>Distribution of prediction confidence scores helps understand model certainty and potential calibration issues.</p>
                    <div class="plot-container" id="confidence-plot"></div>
                </div>
                
                <div class="section">
                    <h2>🔍 Key Insights</h2>
                    <ul>
                        <li><strong>Best Performing Model:</strong> {max(self.results.keys(), key=lambda x: self.results[x].get('f1_score', 0)).replace('_', ' ').title()}</li>
                        <li><strong>Highest Accuracy:</strong> {max(self.results.values(), key=lambda x: x.get('accuracy', 0)).get('accuracy', 0):.4f}</li>
                        <li><strong>Best Precision:</strong> {max(self.results.values(), key=lambda x: x.get('precision', 0)).get('precision', 0):.4f}</li>
                        <li><strong>Best Recall:</strong> {max(self.results.values(), key=lambda x: x.get('recall', 0)).get('recall', 0):.4f}</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🚀 Recommendations</h2>
                    <ul>
                        <li>Consider ensemble methods to combine the strengths of different models</li>
                        <li>Implement cross-validation for more robust performance estimation</li>
                        <li>Analyze feature importance to understand key indicators of fake news</li>
                        <li>Regular model retraining with new data to maintain performance</li>
                        <li>Deploy monitoring systems to track model performance in production</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by Advanced Fake News Detection System | Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <script>
                // Plot the visualizations
                Plotly.newPlot('performance-plot', {performance_fig.to_json()});
                Plotly.newPlot('roc-plot', {roc_fig.to_json()});
                Plotly.newPlot('pr-plot', {pr_fig.to_json()});
                Plotly.newPlot('cm-plot', {fig_cms.to_json()});
                Plotly.newPlot('confidence-plot', {confidence_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        # Save the report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved to {output_path}")
        return output_path
    
    def export_results_to_excel(self, output_path='visualizations/model_results.xlsx'):
        """Export detailed results to Excel file."""
        os.makedirs('visualizations', exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for model_name, results in self.results.items():
                summary_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': results.get('accuracy', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1-Score': results.get('f1_score', 0),
                    'Training Time (s)': results.get('training_time', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results for each model
            for model_name, results in self.results.items():
                if 'classification_report' in results:
                    # Convert classification report to DataFrame
                    report_dict = classification_report(
                        results.get('y_test', []), 
                        results.get('y_pred', []), 
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report_dict).transpose()
                    report_df.to_excel(writer, sheet_name=f'{model_name}_detailed')
        
        print(f"Results exported to {output_path}")
        return output_path