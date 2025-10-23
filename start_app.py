#!/usr/bin/env python3
"""
Startup script for AI Data Cleaner & Web Scraper
Handles common startup issues and provides helpful guidance.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up."""
    print("🔧 Checking environment...")
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Please run this script from the AI-DATA-CLEANER directory.")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print("✅ Environment looks good!")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def install_spacy_model():
    """Install spaCy model if needed."""
    print("🧠 Checking spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("✅ spaCy model already installed!")
        return True
    except OSError:
        print("📦 Installing spaCy model...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✅ spaCy model installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install spaCy model: {e}")
            return False

def start_streamlit():
    """Start the Streamlit application."""
    print("🚀 Starting Streamlit application...")
    
    try:
        # Try to start on port 8501
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        print("💡 Try running manually: streamlit run streamlit_app.py")

def main():
    """Main startup function."""
    print("🔧 AI Data Cleaner & Web Scraper - Startup Assistant")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return False
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Install dependencies and start app")
    print("2. Just start the app (dependencies already installed)")
    print("3. Run diagnostic check")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if install_dependencies() and install_spacy_model():
            start_streamlit()
    elif choice == "2":
        start_streamlit()
    elif choice == "3":
        print("🔍 Running diagnostic check...")
        subprocess.run([sys.executable, "quick_start.py"])
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")
    
    return True

if __name__ == "__main__":
    main()

