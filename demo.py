#!/usr/bin/env python3
"""
Demo script showing the fake news detection system in action.
"""

import os
import sys
import joblib

def demo_prediction():
    """Demonstrate the prediction functionality with sample texts."""
    print("🔍 Fake News Detection System Demo")
    print("=" * 50)
    
    # Load models and preprocessor
    try:
        models = {}
        model_files = {
            'Logistic Regression': 'saved_models/logistic_regression_model.joblib',
            'Random Forest': 'saved_models/random_forest_model.joblib',
            'SVM': 'saved_models/svm_model.joblib'
        }
        
        for name, path in model_files.items():
            models[name] = joblib.load(path)
        
        preprocessor = joblib.load('saved_models/preprocessor.joblib')
        print("✅ Models loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Sample news articles for testing
    sample_articles = [
        {
            "title": "Breaking: Miracle Drug Cures All Diseases!",
            "text": "Scientists have discovered a revolutionary new drug that can cure all known diseases including cancer, AIDS, and even aging. The drug, made from a secret combination of herbs, has been tested on millions of people with 100% success rate. Doctors are amazed by the results and say this will change medicine forever. The drug will be available in stores next week for just $19.99.",
            "expected": "FAKE"
        },
        {
            "title": "Federal Reserve Adjusts Interest Rates",
            "text": "The Federal Reserve announced a 0.25 percentage point increase in the federal funds rate following their two-day meeting this week. The decision was unanimous among voting members and reflects the central bank's ongoing efforts to combat inflation while supporting economic growth. The rate increase brings the federal funds rate to 5.25-5.50 percent, its highest level in over a decade.",
            "expected": "REAL"
        },
        {
            "title": "Celebrity Endorses Miracle Weight Loss",
            "text": "Famous actress loses 50 pounds in 2 weeks using this one weird trick that doctors hate! You won't believe what she did. This secret method has helped thousands of people lose weight without diet or exercise. Click here to discover the shocking truth that the weight loss industry doesn't want you to know!",
            "expected": "FAKE"
        }
    ]
    
    # Process each article
    for i, article in enumerate(sample_articles, 1):
        print(f"📰 Article {i}: {article['title']}")
        print(f"Expected: {article['expected']}")
        print("-" * 40)
        print(f"Text: {article['text'][:100]}...")
        print("\nModel Predictions:")
        
        # Clean text
        cleaned_text = preprocessor.clean_text(article['text'])
        features = preprocessor.transform_texts([cleaned_text])
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            
            label = "REAL" if pred == 1 else "FAKE"
            confidence = max(prob) * 100
            
            predictions[model_name] = {
                'label': label,
                'confidence': confidence,
                'correct': label == article['expected']
            }
            
            status = "✅" if predictions[model_name]['correct'] else "❌"
            print(f"  {status} {model_name}: {label} ({confidence:.1f}% confidence)")
        
        # Summary
        correct_predictions = sum(1 for p in predictions.values() if p['correct'])
        print(f"\nAccuracy: {correct_predictions}/{len(models)} models correct")
        print("=" * 50)
        print()

def demo_interactive():
    """Interactive demo where user can input their own text."""
    print("🎯 Interactive Demo - Enter your own text!")
    print("(Type 'quit' to exit)")
    print("-" * 40)
    
    # Load models
    try:
        models = {}
        model_files = {
            'Logistic Regression': 'saved_models/logistic_regression_model.joblib',
            'Random Forest': 'saved_models/random_forest_model.joblib',
            'SVM': 'saved_models/svm_model.joblib'
        }
        
        for name, path in model_files.items():
            models[name] = joblib.load(path)
        
        preprocessor = joblib.load('saved_models/preprocessor.joblib')
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    while True:
        user_text = input("\nEnter news text to analyze: ").strip()
        
        if user_text.lower() == 'quit':
            print("👋 Thanks for using the demo!")
            break
        
        if not user_text:
            print("Please enter some text.")
            continue
        
        print(f"\nAnalyzing: {user_text[:50]}...")
        print("-" * 30)
        
        # Process text
        cleaned_text = preprocessor.clean_text(user_text)
        features = preprocessor.transform_texts([cleaned_text])
        
        # Get predictions
        for model_name, model in models.items():
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            
            label = "REAL" if pred == 1 else "FAKE"
            confidence = max(prob) * 100
            
            print(f"  {model_name}: {label} ({confidence:.1f}% confidence)")

def main():
    """Run the demo."""
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Welcome to the Fake News Detection System!")
    print("Choose demo mode:")
    print("1. Sample articles demo")
    print("2. Interactive demo")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        demo_prediction()
    elif choice == "2":
        demo_interactive()
    elif choice == "3":
        demo_prediction()
        demo_interactive()
    else:
        print("Invalid choice. Running sample demo...")
        demo_prediction()

if __name__ == "__main__":
    main()
