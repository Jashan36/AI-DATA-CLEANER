# 🔧 Troubleshooting Guide

## Common Interface Issues & Solutions

### 1. Streamlit App Won't Start

**Symptoms:**
- `streamlit run streamlit_app.py` fails
- Import errors or module not found
- Port already in use

**Solutions:**
```bash
# Check if port 8501 is in use
netstat -an | findstr :8501  # Windows
lsof -i :8501                # Mac/Linux

# Kill existing Streamlit processes
taskkill /f /im streamlit.exe  # Windows
pkill -f streamlit            # Mac/Linux

# Install missing dependencies
pip install -r requirements.txt

# Run with different port
streamlit run streamlit_app.py --server.port 8502
```

### 2. Import Errors

**Common Errors:**
- `ModuleNotFoundError: No module named 'core'`
- `ImportError: cannot import name 'DataCleaner'`

**Solutions:**
```bash
# Ensure you're in the correct directory
cd AI-DATA-CLEANER

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# For Windows, ensure Python path is correct
python -c "import sys; print(sys.path)"
```

### 3. Data Upload Issues

**Symptoms:**
- File upload button not working
- "No data was uploaded" error
- File format not supported

**Solutions:**
- Use CSV or Parquet files only
- Ensure file size < 200MB
- Check file encoding (UTF-8 recommended)
- Try with a small sample file first

### 4. Cleaning Process Fails

**Symptoms:**
- "Data cleaning failed" error
- Empty results (0 rows, 0 columns)
- Process hangs indefinitely

**Solutions:**
```bash
# Check system resources
# Windows: Task Manager
# Mac/Linux: htop or top

# Reduce chunk size in sidebar
# Try different output format (CSV instead of Parquet)
# Disable quality gates temporarily
# Use smaller sample data
```

### 5. Quality Report Not Loading

**Symptoms:**
- "Quality report not available" message
- Empty quality metrics
- Missing DQR file

**Solutions:**
- Ensure Quality Gates are enabled in sidebar
- Check if cleaning completed successfully
- Verify output directory has write permissions
- Try with CSV output format

### 6. Web Scraping Issues

**Symptoms:**
- URLs not loading
- "Connection timeout" errors
- Empty scraping results

**Solutions:**
- Check internet connection
- Try with different URLs
- Increase timeout in sidebar settings
- Disable robots.txt checking temporarily
- Use fewer concurrent requests

### 7. Memory Issues

**Symptoms:**
- App crashes with large files
- "Out of memory" errors
- Slow performance

**Solutions:**
```bash
# Reduce chunk size in sidebar (try 1000-5000)
# Use CSV output instead of Parquet
# Process smaller files
# Close other applications
# Restart the app
```

### 8. Browser Issues

**Symptoms:**
- Interface not loading in browser
- Styling issues
- JavaScript errors

**Solutions:**
- Clear browser cache
- Try different browser (Chrome, Firefox, Edge)
- Disable browser extensions
- Check browser console for errors (F12)
- Try incognito/private mode

## Quick Diagnostic Commands

```bash
# Check Python version (3.8+ required)
python --version

# Check installed packages
pip list | grep -E "(streamlit|pandas|polars|plotly)"

# Test basic imports
python -c "import streamlit, pandas, polars, plotly; print('All imports successful')"

# Check file permissions
ls -la streamlit_app.py  # Mac/Linux
dir streamlit_app.py     # Windows

# Test with minimal data
python -c "
import pandas as pd
df = pd.DataFrame({'test': [1,2,3]})
df.to_csv('test.csv', index=False)
print('Test file created')
"
```

## Performance Optimization

### For Large Files:
1. **Reduce chunk size** to 1000-5000 rows
2. **Use CSV output** instead of Parquet
3. **Disable quality gates** for initial testing
4. **Process in smaller batches**

### For Better Performance:
1. **Close other applications**
2. **Use SSD storage** for temp files
3. **Increase system RAM** if possible
4. **Use faster internet** for web scraping

## Error Logs Location

```bash
# Streamlit logs
# Check terminal/console where you ran streamlit

# Python logs
# Look for error messages in the terminal

# Browser logs
# Press F12 → Console tab for JavaScript errors
```

## Getting Help

If issues persist:

1. **Check the terminal** where Streamlit is running for error messages
2. **Try the CLI version** instead of the web interface:
   ```bash
   python cli.py clean-data --input your_file.csv --output results
   ```
3. **Test with sample data** to isolate the issue
4. **Check system requirements** (Python 3.8+, 4GB+ RAM)

## Common File Path Issues

**Windows:**
- Use forward slashes or raw strings: `r"C:\path\to\file"`
- Avoid spaces in file paths
- Use short paths when possible

**Mac/Linux:**
- Ensure proper file permissions: `chmod 644 filename`
- Check for hidden characters in filenames
- Use absolute paths if relative paths fail

## Reset Everything

If all else fails:

```bash
# Stop Streamlit
Ctrl+C

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Restart Streamlit
streamlit run streamlit_app.py
```
