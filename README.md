# AI Data Cleaner & Evaluator

This project is an advanced data cleaning, modeling, and web scraping tool powered by Streamlit, JAX, Plotly, and modern NLP libraries. It supports both data cleaning/analysis and web scraping/company info extraction with advanced metrics and visualizations.

## Features
- Automated data cleaning (outlier handling, imputation, normalization) using JAX
- One-hot encoding for categorical features
- Advanced visualizations (PCA, t-SNE, correlation heatmaps, pair plots)
- Web scraping and NLP entity extraction (spaCy, Transformers)
- Downloadable cleaned datasets and extracted info
- Model training and evaluation (Logistic Regression, etc.)

## How to Run

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage

Run the Streamlit app:
```bash
streamlit run data3.py
```

The app will open in your browser. You can upload CSV files for cleaning, modeling, and analysis, or enter URLs for web scraping and NLP analysis.

### Notes
- For NLP features, install `en_core_web_sm` for spaCy:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- For best HTML parsing, install `lxml`:
  ```bash
  pip install lxml
  ```
- Some features require optional packages (see requirements.txt).

## Project Structure
- `data3.py`: Main Streamlit app with all features


## License
MIT
