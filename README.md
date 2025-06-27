# AI Data Cleaner & Evaluator

This project is an advanced data cleaning and visualization tool powered by Streamlit, JAX, Polars, Plotly, and modern NLP libraries. It supports both data cleaning/analysis and web scraping/company info extraction with advanced metrics and visualizations.

## Features
- Automated data cleaning (outlier handling, imputation, normalization)
- Advanced visualizations (PCA, t-SNE, correlation heatmaps, pair plots)
- Web scraping and NLP entity extraction (spaCy, Transformers)
- Downloadable cleaned datasets and extracted info
- Gradio interface for quick data summaries

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
streamlit run data1.py
```

The app will open in your browser. You can upload CSV files for cleaning or enter URLs for web scraping and NLP analysis.

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
- `data1.py`: Main Streamlit app with all features
- `data.py`: (Optional) Additional scripts or modules

## License
MIT
