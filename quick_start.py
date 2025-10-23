#!/usr/bin/env python3
"""
Quick Start Script for AI Data Cleaner & Web Scraper
Diagnoses common issues and provides solutions.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'polars', 'plotly', 'numpy',
        'fastapi', 'uvicorn', 'beautifulsoup4', 'lxml', 'rapidfuzz'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"OK {package}")
        except ImportError:
            print(f"MISSING {package}")
            missing.append(package)
    
    return missing

def check_spacy_model():
    """Check if spaCy model is installed."""
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("OK spaCy model (en_core_web_sm)")
        return True
    except OSError:
        print("MISSING spaCy model (en_core_web_sm)")
        return False

def check_file_structure():
    """Check if required files exist."""
    required_files = [
        'streamlit_app.py',
        'cli.py',
        'api.py',
        'core/__init__.py',
        'core/data_cleaner.py',
        'core/cleaner_policy.py',
        'core/quality_gates.py',
        'core/web_scraper.py',
        'requirements.txt'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"OK {file_path}")
        else:
            print(f"MISSING {file_path}")
            missing.append(file_path)
    
    return missing

def create_test_data():
    """Create a small test dataset."""
    try:
        import pandas as pd
        
        test_data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'email': ['john@email.com', 'jane@email.com', 'bob@company.org', 'alice@test.net', 'charlie@example.com'],
            'age': [25, 30, 45, 60, 35],
            'status': ['Delivered', 'DELIVERED', 'delivered', 'Completed', 'Shipped']
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv('test_data.csv', index=False)
        print("OK Created test_data.csv")
        return True
    except Exception as e:
        print(f"FAILED to create test data: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core modules."""
    try:
        from core.data_cleaner import DataCleaner, CleaningConfig
        from core.cleaner_policy import CleanerPolicy
        from core.quality_gates import QualityGates
        
        print("OK Core modules import successfully")
        
        # Test basic initialization
        policy = CleanerPolicy()
        quality_gates = QualityGates()
        config = CleaningConfig()
        cleaner = DataCleaner(config)
        
        print("OK Core classes initialize successfully")
        return True
    except Exception as e:
        print(f"FAILED Core functionality test: {e}")
        return False

def install_missing_packages(missing_packages):
    """Install missing packages."""
    if not missing_packages:
        return True
    
    print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
    
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"OK Installed {package}")
        except subprocess.CalledProcessError:
            print(f"FAILED to install {package}")
            return False
    
    return True

def install_spacy_model():
    """Install spaCy model."""
    print("\nInstalling spaCy model...")
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        print("OK Installed spaCy model")
        return True
    except subprocess.CalledProcessError:
        print("FAILED to install spaCy model")
        return False

def main():
    """Main diagnostic function."""
    print("AI Data Cleaner & Web Scraper - Quick Start Diagnostic")
    print("=" * 60)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        print("\nPlease upgrade to Python 3.8 or higher")
        return False
    
    # Check file structure
    print("\n2. Checking file structure...")
    missing_files = check_file_structure()
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        print("Please ensure you're in the correct directory (AI-DATA-CLEANER)")
        return False
    
    # Check dependencies
    print("\n3. Checking dependencies...")
    missing_packages = check_dependencies()
    
    # Check spaCy model
    print("\n4. Checking spaCy model...")
    spacy_ok = check_spacy_model()
    
    # Install missing packages
    if missing_packages:
        if not install_missing_packages(missing_packages):
            return False
    
    # Install spaCy model if needed
    if not spacy_ok:
        if not install_spacy_model():
            return False
    
    # Test basic functionality
    print("\n5. Testing basic functionality...")
    if not test_basic_functionality():
        return False
    
    # Create test data
    print("\n6. Creating test data...")
    if not create_test_data():
        return False
    
    print("\n" + "=" * 60)
    print("All checks passed! You're ready to use the AI Data Cleaner & Web Scraper")
    print("\nNext steps:")
    print("1. Start Streamlit: streamlit run streamlit_app.py")
    print("2. Or use CLI: python cli.py clean-data --input test_data.csv --output results")
    print("3. Or start API: python api.py")
    print("\nFor troubleshooting, see TROUBLESHOOTING.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
