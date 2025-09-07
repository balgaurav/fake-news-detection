#!/usr/bin/env python3
"""
Test script to verify the fake news detection system functionality.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

def test_models_exist():
    """Test if all required model files exist."""
    print("🔍 Testing model files...")
    
    required_files = [
        'saved_models/logistic_regression_model.joblib',
        'saved_models/random_forest_model.joblib',
        'saved_models/svm_model.joblib',
        'saved_models/preprocessor.joblib',
        'saved_models/results_summary.joblib'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All model files found!")
    return True

def test_model_loading():
    """Test if models can be loaded successfully."""
    print("\n🤖 Testing model loading...")
    
    try:
        # Load preprocessor
        preprocessor = joblib.load('saved_models/preprocessor.joblib')
        print("✅ Preprocessor loaded")
        
        # Load models
        models = {}
        model_files = {
            'logistic_regression': 'saved_models/logistic_regression_model.joblib',
            'random_forest': 'saved_models/random_forest_model.joblib',
            'svm': 'saved_models/svm_model.joblib'
        }
        
        for name, path in model_files.items():
            models[name] = joblib.load(path)
            print(f"✅ {name} model loaded")
        
        # Load results
        results = joblib.load('saved_models/results_summary.joblib')
        print("✅ Results summary loaded")
        
        return preprocessor, models, results
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None

def test_prediction():
    """Test if predictions work correctly."""
    print("\n🎯 Testing predictions...")
    
    preprocessor, models, results = test_model_loading()
    if not all([preprocessor, models, results]):
        return False
    
    # Test samples
    test_texts = [
        "Scientists have discovered a revolutionary new treatment for cancer that works in 100% of cases.",
        "The Federal Reserve announced a 0.25% interest rate increase following their latest meeting."
    ]
    
    try:
        for i, text in enumerate(test_texts):
            print(f"\nTesting sample {i+1}: {text[:50]}...")
            
            # Clean and preprocess text
            cleaned_text = preprocessor.clean_text(text)
            processed_text = preprocessor.transform_texts([cleaned_text])
            
            # Make predictions
            for model_name, model in models.items():
                prediction = model.predict(processed_text)[0]
                probability = model.predict_proba(processed_text)[0]
                
                label = "REAL" if prediction == 1 else "FAKE"
                confidence = max(probability) * 100
                
                print(f"  {model_name}: {label} ({confidence:.1f}% confidence)")
        
        print("✅ Predictions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return False

def test_results_summary():
    """Test if results summary contains expected data."""
    print("\n📊 Testing results summary...")
    
    try:
        results = joblib.load('saved_models/results_summary.joblib')
        
        expected_models = ['logistic_regression', 'random_forest', 'svm']
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for model in expected_models:
            if model in results:
                print(f"✅ {model} results found")
                for metric in expected_metrics:
                    if metric in results[model]:
                        value = results[model][metric]
                        print(f"    {metric}: {value:.3f}")
                    else:
                        print(f"    ❌ Missing metric: {metric}")
            else:
                print(f"❌ Missing model: {model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Fake News Detection System Test Suite")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests = [
        test_models_exist,
        test_results_summary,
        test_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to use.")
        print("\nNext steps:")
        print("1. Run main app: python run_app.py app")
        print("2. Run analytics: python run_app.py analytics")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
