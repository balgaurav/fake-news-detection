#!/usr/bin/env python3
"""
Launcher script for the Fake News Detection System.
This script provides easy access to different components of the application.
"""

import os
import sys
import subprocess
import argparse

def run_main_app():
    """Run the main Streamlit application."""
    print("🚀 Starting main Fake News Detection app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])

def run_analytics_dashboard():
    """Run the analytics dashboard."""
    print("📊 Starting Analytics Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "analytics_dashboard.py"])

def run_model_training():
    """Run model training pipeline."""
    print("🤖 Starting model training...")
    subprocess.run([sys.executable, "-m", "models.model_trainer"])

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    parser = argparse.ArgumentParser(description="Fake News Detection System Launcher")
    parser.add_argument(
        "command",
        choices=["app", "analytics", "train", "install"],
        help="Command to run: app (main app), analytics (dashboard), train (model training), install (dependencies)"
    )
    
    args = parser.parse_args()
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.command == "app":
        run_main_app()
    elif args.command == "analytics":
        run_analytics_dashboard()
    elif args.command == "train":
        run_model_training()
    elif args.command == "install":
        install_dependencies()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔍 Fake News Detection System Launcher")
        print("=====================================")
        print("Available commands:")
        print("  python run_app.py app        - Run main application")
        print("  python run_app.py analytics  - Run analytics dashboard")
        print("  python run_app.py train      - Train models")
        print("  python run_app.py install    - Install dependencies")
        print("\nExample: python run_app.py app")
    else:
        main()
