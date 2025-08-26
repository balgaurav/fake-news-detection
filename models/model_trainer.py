"""
Model training and evaluation for fake news detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from models.preprocessor import load_and_preprocess_data

class FakeNewsTrainer:
    """Trainer class for fake news detection models."""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42)
        }
        self.trained_models = {}
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def load_data(self, true_path, fake_path, test_size=0.2):
        """Load and split the dataset."""
        print("Loading and preprocessing data...")
        
        # Load and preprocess data
        texts, labels, self.preprocessor = load_and_preprocess_data(true_path, fake_path)
        
        # Split the data
        X_train_text, X_test_text, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Fit vectorizer and transform texts
        print("Fitting TF-IDF vectorizer...")
        self.preprocessor.fit_vectorizer(X_train_text)
        
        self.X_train = self.preprocessor.transform_texts(X_train_text)
        self.X_test = self.preprocessor.transform_texts(X_test_text)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
    
    def train_single_model(self, model_name):
        """Train a single model."""
        print(f"Training {model_name}...")
        
        model = self.models[model_name]
        model.fit(self.X_train, self.y_train)
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(self):
        """Train all models."""
        print("Training all models...")
        
        for model_name in self.models.keys():
            self.train_single_model(model_name)
        
        print("All models trained successfully!")
    
    def evaluate_single_model(self, model_name):
        """Evaluate a single model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.trained_models[model_name]
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Probabilities for test set
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(self.X_test)[:, 1]
        else:
            y_proba = None
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred_test, average='weighted'
        )
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        results = {
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred_test,
            'y_proba': y_proba
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained models."""
        print("Evaluating all models...")
        
        for model_name in self.trained_models.keys():
            self.evaluate_single_model(model_name)
        
        self.print_comparison_table()
        self.plot_model_comparison()
    
    def print_comparison_table(self):
        """Print a comparison table of all models."""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': f"{results['test_accuracy']:.3f}",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}",
                'CV Mean': f"{results['cv_mean']:.3f}",
                'CV Std': f"{results['cv_std']:.3f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        best_accuracy = self.results[best_model]['test_accuracy']
        
        print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()}")
        print(f"🎯 Best Accuracy: {best_accuracy:.1%}")
    
    def plot_model_comparison(self):
        """Create visualization comparing model performances."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data for plotting
        models = list(self.results.keys())
        model_names = [name.replace('_', ' ').title() for name in models]
        
        accuracies = [self.results[model]['test_accuracy'] for model in models]
        precisions = [self.results[model]['precision'] for model in models]
        recalls = [self.results[model]['recall'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        
        # Colors for each model
        colors = ['#2E8B57', '#4169E1', '#DC143C']
        
        # Plot 1: Accuracy Comparison
        axes[0, 0].bar(model_names, accuracies, color=colors)
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0.8, 1.0)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot 2: Precision vs Recall
        axes[0, 1].scatter(recalls, precisions, c=colors, s=100)
        for i, model in enumerate(model_names):
            axes[0, 1].annotate(model, (recalls[i], precisions[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: F1-Score Comparison
        axes[1, 0].bar(model_names, f1_scores, color=colors)
        axes[1, 0].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim(0.8, 1.0)
        for i, v in enumerate(f1_scores):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot 4: Confusion Matrix for Best Model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        cm = confusion_matrix(self.y_test, self.results[best_model]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model.replace("_", " ").title()}', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Model comparison plot saved to 'visualizations/model_comparison.png'")
    
    def save_models(self):
        """Save trained models and preprocessor."""
        os.makedirs('saved_models', exist_ok=True)
        
        # Save models
        for model_name, model in self.trained_models.items():
            model_path = f'saved_models/{model_name}_model.joblib'
            joblib.dump(model, model_path)
            print(f"✅ {model_name} saved to {model_path}")
        
        # Save preprocessor
        preprocessor_path = 'saved_models/preprocessor.joblib'
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✅ Preprocessor saved to {preprocessor_path}")
        
        # Save results summary
        results_path = 'saved_models/results_summary.joblib'
        joblib.dump(self.results, results_path)
        print(f"✅ Results summary saved to {results_path}")
    
    def get_detailed_report(self, model_name):
        """Get detailed classification report for a specific model."""
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")
        
        y_pred = self.results[model_name]['y_pred']
        report = classification_report(self.y_test, y_pred, 
                                     target_names=['Fake News', 'Real News'])
        
        print(f"\n📋 DETAILED REPORT FOR {model_name.upper()}")
        print("=" * 50)
        print(report)
        
        return report

def main():
    """Main training pipeline."""
    trainer = FakeNewsTrainer()
    
    try:
        # Load data
        trainer.load_data('data/processed/True.csv', 'data/processed/Fake.csv')
        
        # Train all models
        trainer.train_all_models()
        
        # Evaluate models
        trainer.evaluate_all_models()
        
        # Get detailed report for best model
        best_model = max(trainer.results.keys(), 
                        key=lambda x: trainer.results[x]['test_accuracy'])
        trainer.get_detailed_report(best_model)
        
        # Save models
        trainer.save_models()
        
        print("\n🎉 Training pipeline completed successfully!")
        
    except FileNotFoundError:
        print("❌ Error: Dataset files not found.")
        print("Please download True.csv and Fake.csv from Kaggle and place them in the 'data/processed/' directory.")
        print("Dataset URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()


