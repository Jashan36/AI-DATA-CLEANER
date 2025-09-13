import streamlit as st
import jax
import jax.numpy as jnp
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, precision_recall_curve,
    average_precision_score
)
# --- START: ADDED FOR MODELING ---
from sklearn.linear_model import LogisticRegression
# --- END: ADDED FOR MODELING ---
import io
import warnings
import time
from typing import Tuple, Dict, Any, List, Optional, Callable
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
import requests
from bs4 import BeautifulSoup
import re
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = None
try:
    import robotexclusionrulesparser
    RobotsParser = robotexclusionrulesparser.RobotFileParserLookalike
except ImportError:
    RobotsParser = None

# Check if lxml is importable and warn if not
try:
    import lxml
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    st.warning("lxml is not installed or not available in this environment. Please ensure you are running Streamlit with the correct Python environment and that lxml is installed. Try: pip install lxml")

# Clear JAX caches to avoid stale JIT state
jax.clear_caches()

# Suppress warnings
warnings.filterwarnings('ignore')

# JAX Configuration
jax.config.update("jax_enable_x64", True)

# ======================== ENHANCED DATA CLEANER ========================
class JAXDataCleaner:
    """Enhanced JAX-powered data cleaning pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        self.outlier_threshold = 3.0
        self.imputation_method = "mean"
        # Special cleaners for known messy columns
        self.special_cleaners = {
            'Price': self._clean_price_column,
            'Rating': self._clean_rating_column
        }

    def _clean_price_column(self, series):
        def clean_price(val):
            try:
                if pd.isnull(val): return np.nan
                val_str = str(val).lower()
                if 'free' in val_str: return 0.0
                # Remove currency symbols and words
                for word in ['rs', '$', 'usd', 'inr', '₹', 'approx.', 'approx', 'eur', 'gbp', 'cad', 'aud', 'n/a', 'na', 'none']:
                    val_str = val_str.replace(word, '')
                digits = ''.join(c for c in val_str if c.isdigit() or c=='.')
                return float(digits)
            except Exception as e:
                print(f"Error cleaning price: {e}")
                return np.nan
        return series.apply(clean_price)

    def _clean_rating_column(self, series):
        rating_map = {'bad': 1, 'poor': 2, 'ok': 3, 'average': 3, 'good': 4, 'excellent': 5, 'n/a': np.nan, 'na': np.nan, 'none': np.nan}
        def clean_rating(val):
            try:
                if pd.isnull(val): return np.nan
                if isinstance(val, (int, float)): return val
                val_str = str(val).strip().lower()
                if val_str in rating_map: return rating_map[val_str]
                return float(val_str)
            except Exception as e:
                print(f"Error cleaning rating: {e}")
                return np.nan
        return series.apply(clean_rating)

    def set_parameters(self, outlier_threshold: float = 3.0,
                      imputation_method: str = "mean"):
        self.outlier_threshold = outlier_threshold
        self.imputation_method = imputation_method
        
    def detect_outliers_jax(self, data: jnp.ndarray) -> jnp.ndarray:
        """Detect outliers using JAX-optimized Z-score method"""
        @jax.jit
        def z_score_outliers(x):
            mean = jnp.nanmean(x, axis=0)
            std = jnp.nanstd(x, axis=0)
            z_scores = jnp.abs((x - mean) / (std + 1e-8))
            return jnp.any(z_scores > self.outlier_threshold, axis=1)
        
        return z_score_outliers(data)
    
    def fill_missing_values_jax(self, data: jnp.ndarray) -> jnp.ndarray:
        """Fill missing values using selected method"""
        @jax.jit
        def impute(x):
            if self.imputation_method == "mean":
                fill_value = jnp.nanmean(x, axis=0)
            elif self.imputation_method == "median":
                fill_value = jnp.nanmedian(x, axis=0)
            elif self.imputation_method == "zero":
                fill_value = jnp.zeros(x.shape[1])
            return jnp.where(jnp.isnan(x), fill_value, x)
        
        return impute(data)
    
    def handle_outliers(self, data: jnp.ndarray, strategy: str = "cap") -> jnp.ndarray:
        """Handle outliers using specified strategy (column-wise)"""
        @jax.jit
        def process(x):
            q1 = jnp.nanpercentile(x, 25, axis=0)
            q3 = jnp.nanpercentile(x, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if strategy == "remove":
                mask = (x < lower_bound) | (x > upper_bound)
                return jnp.where(mask, jnp.nan, x)
            elif strategy == "cap":
                x = jnp.where(x < lower_bound, lower_bound, x)
                x = jnp.where(x > upper_bound, upper_bound, x)
                return x
            else:  # "ignore"
                return x
        
        return process(data)
    
    def preprocess_data(self, df: pd.DataFrame,
                       outlier_strategy: str = "cap") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete preprocessing pipeline with outlier handling and messy string cleaning"""
        try:
            processed_df = df.copy()
            stats = {}

            # Step 1: Special cleaning for known messy columns (Price, Rating, etc.)
            for col_name, cleaner_func in self.special_cleaners.items():
                if col_name in processed_df.columns:
                    processed_df[col_name] = cleaner_func(processed_df[col_name])

            # Step 2: Separate numeric and categorical columns
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns

            # Step 3: Handle categorical data
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))

            # Step 4: Process numeric columns
            if len(numeric_cols) > 0:
                numeric_data = jnp.array(processed_df[numeric_cols].values, dtype=jnp.float32)
                
                # Handle outliers
                numeric_data = self.handle_outliers(numeric_data, outlier_strategy)
                
                # Fill missing values
                numeric_data_filled = self.fill_missing_values_jax(numeric_data)
                
                # Detect outliers after handling
                outlier_mask = self.detect_outliers_jax(numeric_data_filled)
                stats['outlier_count'] = int(jnp.sum(outlier_mask))
                stats['outlier_percentage'] = float(jnp.mean(outlier_mask) * 100)
                
                # Update dataframe
                processed_df[numeric_cols] = numeric_data_filled
            
            # Store feature statistics
            stats['original_shape'] = df.shape
            stats['processed_shape'] = processed_df.shape
            stats['missing_values_original'] = df.isnull().sum().sum()
            stats['missing_values_processed'] = processed_df.isnull().sum().sum()
            stats['dtypes'] = str(df.dtypes.value_counts().to_dict())
            
            self.feature_stats = stats
            return processed_df, stats
        except Exception as e:
            print(f"Error in preprocess_data: {e}")
            return df, {}

# ======================== ENHANCED VISUALIZATIONS & METRICS ========================
def calculate_advanced_metrics(y_true, y_pred, y_prob, problem_type="binary"):
    """Calculate modern ML metrics"""
    metrics = {}
    
    # Common metrics
    metrics['accuracy'] = np.mean(y_true == y_pred)
    
    if problem_type == "binary":
        # Binary classification metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['precision'] = tp / (tp + fp + 1e-8)
        metrics['recall'] = tp / (tp + fn + 1e-8)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-8)
        try:
            # y_prob should be the probability of the positive class
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        metrics['log_loss'] = -np.mean(y_true * np.log(y_prob + 1e-8) + (1 - y_true) * np.log(1 - y_prob + 1e-8))
    else:
        # Multi-class metrics
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        try:
            # y_prob should be the probability matrix of shape (n_samples, n_classes)
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['roc_auc'] = np.nan
    
    # Class imbalance analysis
    unique, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = dict(zip(unique, counts))
    metrics['imbalance_ratio'] = max(counts) / min(counts) if len(counts) > 1 else 1.0
    
    return metrics

def create_evaluation_plots(y_true, y_pred, y_prob, target_name="Target", is_binary=True):
    """Create comprehensive evaluation plots"""
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                      title="Confusion Matrix",
                      labels=dict(x="Predicted", y="Actual", color="Count"))
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC and PR Curves (only if binary)
    if is_binary:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                    name=f'ROC Curve (AUC = {roc_auc:.3f}'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                    line=dict(dash='dash'), name='Random'))
        fig_roc.update_layout(title='ROC Curve', 
                             xaxis_title='False Positive Rate', 
                             yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)
    
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', 
                              name=f'PR Curve (AUC = {pr_auc:.3f}'))
        fig_pr.update_layout(title='Precision-Recall Curve',
                            xaxis_title='Recall',
                            yaxis_title='Precision')
        st.plotly_chart(fig_pr, use_container_width=True)
    
        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model'))
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   line=dict(dash='dash'), name='Ideal'))
        fig_cal.update_layout(title='Calibration Curve',
                             xaxis_title='Mean Predicted Probability',
                             yaxis_title='Fraction of Positives')
        st.plotly_chart(fig_cal, use_container_width=True)
    
def plot_feature_importance(importance_df: pd.DataFrame):
    """Visualize feature importance"""
    fig = px.bar(importance_df.head(20),
                 x='Importance', y='Feature',
                 orientation='h',
                 title="Top 20 Feature Importances (from Logistic Regression)")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
def visualize_embeddings(X_emb: np.ndarray, y: np.ndarray, method: str = "PCA"):
    """Visualize high-dimensional data in 2D"""
    if method == "PCA":
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(X_emb)
        title = "PCA Projection of Cleaned Data"
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=min(30, len(X_emb)-1), random_state=42)
        emb = reducer.fit_transform(X_emb)
        title = "t-SNE Projection of Cleaned Data"
        
    fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=pd.Series(y).astype(str),
                   title=title, labels={'color': 'Target Class'})
    st.plotly_chart(fig, use_container_width=True)

# ======================== HELPER FUNCTIONS ... (omitting for brevity, no changes here) ========================
# ... The rest of the non-UI functions like web scraping utils are unchanged ...
# All other functions from your original code would remain here. I am omitting them for a more concise answer.

import nest_asyncio
nest_asyncio.apply()
import httpx
import asyncio
from transformers import pipeline
TRANSFORMERS_AVAILABLE = True
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

def is_allowed_by_robots(url):
    if RobotsParser is None:
        return True
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotsParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("DataAnalyzerBot/1.0 (contact@example.com)", url)
    except Exception:
        return True

def web_scraping_ui():
    st.header("🌐 Web Scraping & Company Info Extractor")
    st.sidebar.subheader("Web Scraping/NLP Parameters")
    nlp_model = st.sidebar.selectbox("NLP Model", ["spaCy", "Transformer (summarization)"])
    sentiment_enabled = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
    entity_types = st.sidebar.multiselect(
        "Entity Types to Extract",
        ["ORG", "PERSON", "GPE", "LOC", "EMAIL", "PHONE", "ADDRESS", "FAC", "NORP", "PRODUCT", "EVENT", "LAW", "LANGUAGE"],
        default=["ORG", "PERSON", "GPE", "EMAIL", "PHONE"]
    )
    timeout = st.sidebar.slider("HTTP Timeout (seconds)", 5, 60, 30)
    retries = st.sidebar.slider("HTTP Retries", 1, 5, 3)
    batch_size = st.sidebar.slider("Batch Size (async)", 1, 20, 5)

    url_input = st.text_area("Enter one or more URLs (one per line):")
    if st.button("Fetch & Analyze URLs"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        results = []
        async def fetch_all(urls):
            async with httpx.AsyncClient(timeout=timeout) as client:
                tasks = []
                for url in urls:
                    if not is_allowed_by_robots(url):
                        st.warning(f"Blocked by robots.txt: {url}")
                        continue
                    tasks.append(client.get(url, headers={"User-Agent": "DataAnalyzerBot/1.0 (contact@example.com)"}))
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                return responses
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            try:
                responses = asyncio.get_event_loop().run_until_complete(fetch_all(batch))
            except Exception as e:
                st.error(f"Async fetch error: {e}")
                continue
            for url, resp in zip(batch, responses):
                if isinstance(resp, Exception):
                    st.error(f"Failed to fetch {url}: {resp}")
                    continue
                html = resp.text if hasattr(resp, 'text') else str(resp)
                info = parse_and_extract_info_adv(html, url, nlp_model, entity_types, sentiment_enabled)
                results.append(info)
                st.write(f"**Summary:** {info['summary']}")
                st.write(f"**Entities:** {info['entities']}")
                st.write(f"**Sentiment:** {info.get('sentiment', '')}")
                st.write(f"**About Section:** {info['about']}")
                st.write(f"**Contact Section:** {info['contact']}")
                st.write(f"**Main Text (first 500 chars):** {info['main_text'][:500]}")
                if info['tables']:
                    for i, tbl in enumerate(info['tables']):
                        st.write(f"Table {i+1}")
                        st.dataframe(tbl.head())
        # Download structured results
        if results:
            def format_entities(entities):
                return "; ".join([f"{label}:{text}" for text, label in entities])
            def clean_text(text, maxlen=300):
                if not text:
                    return ""
                text = str(text).replace("\n", " ").replace("\r", " ").strip()
                return text[:maxlen] + ("..." if len(text) > maxlen else "")
            # Normal extracted info CSV
            df_struct = pd.DataFrame([
                {
                    "url": r["url"],
                    "summary": clean_text(r["summary"], 300),
                    "entities": format_entities(r["entities"]),
                    "about": clean_text(r["about"], 300),
                    "contact": clean_text(r["contact"], 300),
                    "sentiment": r.get("sentiment", "")
                } for r in results
            ])
            csv = df_struct.to_csv(index=False).encode('utf-8')
            st.download_button("Download Extracted Info as CSV", csv, "extracted_info.csv", "text/csv")

            # Cleaned data for model training
            def extract_by_label(entities, label):
                return ";".join([text for text, l in entities if l == label])
            df_ml = pd.DataFrame([
                {
                    "url": r["url"],
                    "orgs": extract_by_label(r["entities"], "ORG"),
                    "persons": extract_by_label(r["entities"], "PERSON"),
                    "locations": extract_by_label(r["entities"], "GPE") + (";" + extract_by_label(r["entities"], "LOC") if extract_by_label(r["entities"], "LOC") else ""),
                    "emails": extract_by_label(r["entities"], "EMAIL"),
                    "phones": extract_by_label(r["entities"], "PHONE"),
                    "address": extract_by_label(r["entities"], "ADDRESS"),
                    "fac": extract_by_label(r["entities"], "FAC"),
                    "norp": extract_by_label(r["entities"], "NORP"),
                    "product": extract_by_label(r["entities"], "PRODUCT"),
                    "event": extract_by_label(r["entities"], "EVENT"),
                    "law": extract_by_label(r["entities"], "LAW"),
                    "language": extract_by_label(r["entities"], "LANGUAGE")
                } for r in results
            ])
            csv_ml = df_ml.to_csv(index=False).encode('utf-8')
            st.download_button("Download ML-Ready Data as CSV", csv_ml, "ml_ready_data.csv", "text/csv")

def parse_and_extract_info_adv(html, url, nlp_model, entity_types, sentiment_enabled):
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
        if 'st' in globals():
            st.warning("lxml parser not found, using slower 'html.parser'. Install lxml for best results: pip install lxml")
    texts = [t.get_text(separator=" ", strip=True) for t in soup.find_all(['p', 'li'])]
    main_text = " ".join(texts)
    main_text_lines = [line.strip() for line in main_text.splitlines() if len(line.strip()) > 30]
    main_text_clean = " ".join(main_text_lines)
    tables = []
    try:
        tables = pd.read_html(html)
    except Exception:
        pass
    def extract_section(soup, keyword):
        tag = soup.find(text=re.compile(keyword, re.I))
        if tag:
            parent = tag.find_parent(['section', 'div', 'p'])
            if parent:
                return parent.get_text(separator=" ", strip=True)
            return tag.strip()
        return None
    about = extract_section(soup, r"about")
    contact = extract_section(soup, r"contact")
    emails = re.findall(r"[\w\.-]+@[\w\.-]+", html)
    phones = re.findall(r"\+?\d[\d\s\-]{7,}\d", html)
    entities = []
    summary = None
    sentiment = None
    # --- NLP Model Selection ---
    if nlp_model == "Transformer (summarization)" and TRANSFORMERS_AVAILABLE and main_text_clean:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(main_text_clean[:1024])[0]['summary_text']
        except Exception as e:
            summary = main_text_clean[:300]
        if sentiment_enabled:
            try:
                sentiment_pipe = pipeline("sentiment-analysis")
                sentiment = sentiment_pipe(main_text_clean[:512])[0]['label']
            except Exception:
                sentiment = ""
    elif nlp is not None and main_text_clean:
        doc = nlp(main_text_clean)
        sents = list(doc.sents)
        summary = " ".join([sent.text for sent in sents[:3]])
        # Filter entities by user selection
        wanted_labels = set(entity_types)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in wanted_labels]
        # Add regex emails/phones as entities
        if "EMAIL" in wanted_labels:
            entities += [(email, "EMAIL") for email in emails]
        if "PHONE" in wanted_labels:
            entities += [(phone, "PHONE") for phone in phones]
        if sentiment_enabled and TEXTBLOB_AVAILABLE:
            try:
                sentiment = TextBlob(main_text_clean[:512]).sentiment.polarity
            except Exception:
                sentiment = ""
        else:
            sentiment = ""
    else:
        summary = " ".join(main_text_clean.split(". ")[:2])
        entities = [(email, "EMAIL") for email in emails] + [(phone, "PHONE") for phone in phones]
    return {
        "url": url,
        "summary": summary,
        "entities": entities,
        "about": about,
        "contact": contact,
        "main_text": main_text_clean,
        "tables": tables,
        "sentiment": sentiment
    }

# --- Modern Entrypoint ---
def main():
    st.write("App started")
    st.set_page_config(page_title="🔧 AI Data Cleaner & Evaluator", layout="wide")
    st.title("🔧 AI-Powered Data Cleaner, Modeler & Evaluator")
    st.markdown("An End-to-End Automated Pipeline for Data Cleaning, Modeling, and Evaluation.")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Data Cleaning & Analysis", "Web Scraping & Company Info"])
    if page == "Web Scraping & Company Info":
        web_scraping_ui()
        return

    # --- Automated Data Cleaning Parameters ---
    outlier_threshold = 3.0
    imputation_method = "mean"
    outlier_strategy = "cap"

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return
        try:
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            st.subheader("📊 Raw Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            with st.expander("Data Summary"):
                try:
                    st.write(f"Columns: {len(df.columns)}")
                    st.write(f"Rows: {len(df)}")
                    st.write(f"Missing Values: {df.isnull().sum().sum()}")
                    st.write(f"Duplicate Rows: {df.duplicated().sum()}")
                    st.write("Data Types:")
                    st.write(df.dtypes.value_counts())
                except Exception as e:
                    st.error(f"Error in Data Summary: {e}")

            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 0:
                target_col = cat_cols[-1]
            else:
                target_col = df.columns[-1]
            st.info(f"Automatically selected target column: **{target_col}**. All other columns are treated as features.")

            with st.spinner("⏳ Running automated data cleaning pipeline..."):
                try:
                    cleaner = JAXDataCleaner()
                    cleaner.set_parameters(outlier_threshold, imputation_method)
                    X_raw = df.drop(columns=[target_col])
                    y_raw = df[target_col]
                    X_clean, stats = cleaner.preprocess_data(X_raw, outlier_strategy)
                    le_target = LabelEncoder()
                    y_clean_np = le_target.fit_transform(y_raw)
                    X_clean_np = cleaner.scaler.fit_transform(X_clean)
                except Exception as e:
                    st.error(f"Error in data cleaning: {e}")
                    return
            
            st.subheader("🧹 Data Cleaning Results")
            col1, col2, col3 = st.columns(3)
            outlier_count = stats.get('outlier_count', 0)
            outlier_percentage = stats.get('outlier_percentage', 0.0)
            col1.metric("Outliers Handled", outlier_count)
            col2.metric("Missing Values Filled", stats.get('missing_values_original', 0))
            col3.metric("Features Scaled", X_clean_np.shape[1])
            st.success("✅ Data cleaning complete!")
            
            with st.expander("Cleaned Data Preview"):
                try:
                    st.dataframe(pd.DataFrame(X_clean_np, columns=X_clean.columns).head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying cleaned data: {e}")

            try:
                cleaned_df = pd.DataFrame(X_clean_np, columns=X_clean.columns)
                cleaned_df[target_col] = y_clean_np
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned & Scaled Dataset as CSV",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing download: {e}")
            
            st.markdown("---")

            ### --- START: ADDED FOR MODELING --- ###
            
            st.subheader("🤖 Automated Model Training & Evaluation")
            st.markdown("A `Logistic Regression` model will be trained on the cleaned data to predict the target variable.")

            # Model Training Controls
            st.sidebar.subheader("Model Parameters")
            test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.sidebar.number_input("Random State for Splitting", value=42)

            with st.spinner(f"Splitting data and training model..."):
                try:
                    # 1. Split Data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_clean_np, y_clean_np, test_size=test_size, random_state=random_state, stratify=y_clean_np
                    )
                    st.write(f"Data Split: Training set has {X_train.shape[0]} samples, Test set has {X_test.shape[0]} samples.")

                    # 2. Train Model
                    model = LogisticRegression(random_state=random_state, max_iter=1000)
                    model.fit(X_train, y_train)

                    # 3. Get Predictions
                    y_pred = model.predict(X_test)
                    y_prob_full = model.predict_proba(X_test)
                    
                    # Determine problem type for evaluation
                    n_classes = len(le_target.classes_)
                    is_binary = n_classes == 2
                    problem_type = "binary" if is_binary else "multiclass"

                    # For binary classification, roc_auc and pr_auc need prob of positive class
                    y_prob_for_auc = y_prob_full[:, 1] if is_binary else y_prob_full

                except Exception as e:
                    st.error(f"❌ An error occurred during model training: {e}")
                    return

            st.success("✅ Model training and prediction complete!")
            st.markdown("---")

            st.subheader("📈 Model Performance Metrics")
            
            # 4. Calculate and Display Metrics
            metrics = calculate_advanced_metrics(y_test, y_pred, y_prob_for_auc, problem_type=problem_type)
            
            # Display key metrics in columns
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            m_col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            m_col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            m_col4.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
            
            # Display AUC scores if available
            if not np.isnan(metrics.get('roc_auc', np.nan)):
                auc_col1, auc_col2 = st.columns(2)
                auc_col1.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
                if 'pr_auc' in metrics:
                   auc_col2.metric("Precision-Recall AUC", f"{metrics.get('pr_auc', 0):.3f}")

            with st.expander("View Detailed Classification Report"):
                 # Get unique classes in test set to avoid mismatch
                 unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
                 target_names_filtered = [str(le_target.classes_[i]) for i in unique_test_classes if i < len(le_target.classes_)]
                 report = classification_report(y_test, y_pred, target_names=target_names_filtered, labels=unique_test_classes)
                
                 st.text(report)

            # 5. Create and Display Evaluation Plots
            st.subheader("📊 Model Evaluation Plots")
            create_evaluation_plots(y_test, y_pred, y_prob_for_auc, target_name=target_col, is_binary=is_binary)
            
            st.markdown("---")
            
            # 6. Feature Importance
            st.subheader("🌟 Feature Importance Analysis")
            try:
                if hasattr(model, 'coef_'):
                    # Handle both binary and multi-class coefficients
                    if model.coef_.shape[0] == 1: # Binary
                        importances = np.abs(model.coef_[0])
                    else: # Multi-class, take mean of absolute coefficients across classes
                        importances = np.mean(np.abs(model.coef_), axis=0)
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': X_clean.columns,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    
                    plot_feature_importance(feature_importance_df)
                else:
                    st.info("This model type does not provide direct feature importance coefficients.")
            except Exception as e:
                st.error(f"Could not generate feature importance plot: {e}")
                
            ### --- END: ADDED FOR MODELING --- ###

            st.markdown("---")
            st.subheader("🔍 Visualize High-Dimensional Data")
            # Automated: Use PCA if features > 10, else t-SNE
            vis_method = "PCA" if X_clean_np.shape[1] > 10 else "t-SNE"
            st.info(f"Using **{vis_method}** to visualize the cleaned feature space.")
            try:
                visualize_embeddings(X_clean_np, y_clean_np, method=vis_method)
            except Exception as e:
                st.error(f"Error in embedding visualization: {e}")
        
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            return
            
    # About section
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app is powered by Streamlit, JAX, and Scikit-learn. "
        "It provides an end-to-end solution for data cleaning, modeling, and evaluation."
    )
    st.sidebar.subheader("GitHub Repository")
    st.sidebar.markdown(
        "[View Source Code](https://github.com/yourusername/your-repo)"
    )

if __name__ == "__main__":
    # Ensure all helper/webscraping functions are defined or pasted back in before running
    # This is a placeholder for a complete run
    main()
