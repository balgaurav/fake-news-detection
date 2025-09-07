#!/usr/bin/env python3
"""
Startup guide and quick launcher for the Fake News Detection System.
"""

import os
import sys
import subprocess

def show_banner():
    """Display the welcome banner."""
    print("🔍" * 20)
    print("🔍" + " " * 36 + "🔍")
    print("🔍   FAKE NEWS DETECTION SYSTEM   🔍")
    print("🔍" + " " * 36 + "🔍")
    print("🔍" * 20)
    print()

def show_menu():
    """Display the main menu."""
    print("📋 Available Options:")
    print("=" * 40)
    print("1. 🚀 Launch Main Prediction App")
    print("2. 📊 Launch Analytics Dashboard")
    print("3. 🧪 Run System Tests")
    print("4. 🎯 Run Demo")
    print("5. 🤖 Train Models (if needed)")
    print("6. 📦 Install Dependencies")
    print("7. 📖 View Documentation")
    print("8. 🌐 Open Both Apps in Browser")
    print("9. ❌ Exit")
    print("=" * 40)

def get_python_cmd():
    """Get the correct Python command."""
    return "/opt/homebrew/bin/python3.10"

def launch_main_app():
    """Launch the main prediction app."""
    print("🚀 Starting Main Prediction App...")
    print("   URL: http://localhost:8501")
    cmd = [get_python_cmd(), "-m", "streamlit", "run", "app/streamlit_app.py"]
    subprocess.run(cmd)

def launch_analytics():
    """Launch the analytics dashboard."""
    print("📊 Starting Analytics Dashboard...")
    print("   URL: http://localhost:8502")
    cmd = [get_python_cmd(), "-m", "streamlit", "run", "analytics_dashboard.py", "--server.port", "8502"]
    subprocess.run(cmd)

def run_tests():
    """Run system tests."""
    print("🧪 Running System Tests...")
    cmd = [get_python_cmd(), "test_system.py"]
    subprocess.run(cmd)

def run_demo():
    """Run the demo."""
    print("🎯 Starting Demo...")
    cmd = [get_python_cmd(), "demo.py"]
    subprocess.run(cmd)

def train_models():
    """Train models."""
    print("🤖 Starting Model Training...")
    print("⚠️  This may take several minutes...")
    cmd = [get_python_cmd(), "-m", "models.model_trainer"]
    subprocess.run(cmd)

def install_deps():
    """Install dependencies."""
    print("📦 Installing Dependencies...")
    cmd = [get_python_cmd(), "-m", "pip", "install", "-r", "requirements.txt"]
    subprocess.run(cmd)

def show_docs():
    """Show documentation."""
    print("📖 Documentation")
    print("=" * 40)
    print()
    print("📁 Project Structure:")
    print("  ├── app/streamlit_app.py     - Main prediction interface")
    print("  ├── analytics_dashboard.py  - Model analysis dashboard")
    print("  ├── models/                 - ML model implementations")
    print("  ├── saved_models/           - Trained model files")
    print("  └── data/processed/         - Dataset files")
    print()
    print("🔧 Quick Commands:")
    print("  • Main App:      streamlit run app/streamlit_app.py")
    print("  • Analytics:     streamlit run analytics_dashboard.py")
    print("  • Train Models:  python -m models.model_trainer")
    print("  • Run Tests:     python test_system.py")
    print("  • Demo:          python demo.py")
    print()
    print("🌐 URLs (when running):")
    print("  • Main App:      http://localhost:8501")
    print("  • Analytics:     http://localhost:8502")
    print()
    print("📊 Features:")
    print("  • Real-time fake news detection")
    print("  • Multiple ML models (Logistic Regression, Random Forest, SVM)")
    print("  • Interactive analytics dashboard")
    print("  • Performance metrics and visualizations")
    print("  • Batch processing capabilities")
    print("  • Model comparison tools")
    print()

def open_browsers():
    """Open both apps in browser."""
    print("🌐 Opening both applications...")
    print("Starting Main App on port 8501...")
    
    # Start main app in background
    cmd1 = [get_python_cmd(), "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true"]
    proc1 = subprocess.Popen(cmd1)
    
    print("Starting Analytics Dashboard on port 8502...")
    
    # Start analytics in background
    cmd2 = [get_python_cmd(), "-m", "streamlit", "run", "analytics_dashboard.py", "--server.port", "8502", "--server.headless", "true"]
    proc2 = subprocess.Popen(cmd2)
    
    print("\n✅ Both applications are starting...")
    print("🌐 URLs:")
    print("   Main App:      http://localhost:8501")
    print("   Analytics:     http://localhost:8502")
    print("\nPress Ctrl+C to stop both applications")
    
    try:
        proc1.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping applications...")
        proc1.terminate()
        proc2.terminate()

def main():
    """Main function."""
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    show_banner()
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            launch_main_app()
        elif choice == "2":
            launch_analytics()
        elif choice == "3":
            run_tests()
        elif choice == "4":
            run_demo()
        elif choice == "5":
            train_models()
        elif choice == "6":
            install_deps()
        elif choice == "7":
            show_docs()
        elif choice == "8":
            open_browsers()
        elif choice == "9":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to return to menu...")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
