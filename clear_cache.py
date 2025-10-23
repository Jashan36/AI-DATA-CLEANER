#!/usr/bin/env python3
"""
Clear Streamlit cache and restart the application
This script helps clear any cached content that might be showing old GitHub references
"""

import os
import shutil
import subprocess
import sys

def clear_streamlit_cache():
    """Clear Streamlit cache directories"""
    print("🧹 Clearing Streamlit cache...")
    
    # Common Streamlit cache locations
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        os.path.expanduser("~/.cache/streamlit"),
        ".streamlit",
        "__pycache__",
        ".pytest_cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir)
                    print(f"✅ Removed cache directory: {cache_dir}")
                else:
                    os.remove(cache_dir)
                    print(f"✅ Removed cache file: {cache_dir}")
            except Exception as e:
                print(f"⚠️  Could not remove {cache_dir}: {e}")
        else:
            print(f"ℹ️  Cache directory not found: {cache_dir}")

def clear_python_cache():
    """Clear Python cache files"""
    print("\n🐍 Clearing Python cache...")
    
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"✅ Removed Python cache: {cache_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove {cache_path}: {e}")
        
        for file_name in files:
            if file_name.endswith(".pyc"):
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"✅ Removed Python cache file: {file_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")

def restart_streamlit():
    """Restart Streamlit application"""
    print("\n🚀 Restarting Streamlit application...")
    
    try:
        # Kill any existing Streamlit processes
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/f", "/im", "streamlit.exe"], 
                         capture_output=True, text=True)
        else:  # Unix-like systems
            subprocess.run(["pkill", "-f", "streamlit"], 
                         capture_output=True, text=True)
        
        print("✅ Stopped existing Streamlit processes")
        
        # Start Streamlit
        print("🔄 Starting fresh Streamlit application...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "data3.py"])
        
    except Exception as e:
        print(f"⚠️  Could not restart Streamlit automatically: {e}")
        print("💡 Please manually run: streamlit run data3.py")

def main():
    """Main function to clear all caches and restart"""
    print("🔧 AI Data Cleaner - Cache Clearing Utility")
    print("=" * 50)
    
    # Clear all caches
    clear_streamlit_cache()
    clear_python_cache()
    
    print("\n✅ Cache clearing completed!")
    print("\n💡 To ensure GitHub references are completely removed:")
    print("   1. Restart your Streamlit application")
    print("   2. Clear your browser cache (Ctrl+F5 or Cmd+Shift+R)")
    print("   3. Check the sidebar - it should now show 'Source Code' instead of GitHub links")
    
    # Ask if user wants to restart Streamlit
    restart = input("\n🔄 Would you like to restart Streamlit now? (y/n): ").lower().strip()
    if restart in ['y', 'yes']:
        restart_streamlit()
    else:
        print("\n📝 To restart manually, run: streamlit run data3.py")

if __name__ == "__main__":
    main()


