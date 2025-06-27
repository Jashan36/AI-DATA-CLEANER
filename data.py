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
import io
import warnings
import time
from typing import Tuple, Dict, Any, List, Optional, Callable
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

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
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['roc_auc'] = np.nan
    
    # Class imbalance analysis
    unique, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = dict(zip(unique, counts))
    metrics['imbalance_ratio'] = max(counts) / min(counts) if len(counts) > 1 else 1.0
    
    return metrics

def create_evaluation_plots(y_true, y_pred, y_prob, metrics, target_name="Target"):
    """Create comprehensive evaluation plots"""
    # ROC Curve
    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                    name=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f}'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                    line=dict(dash='dash'), name='Random'))
        fig_roc.update_layout(title='ROC Curve', 
                             xaxis_title='False Positive Rate', 
                             yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Precision-Recall Curve
    if 'pr_auc' in metrics:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', 
                              name=f'PR Curve (AUC = {metrics["pr_auc"]:.3f}'))
        fig_pr.update_layout(title='Precision-Recall Curve',
                            xaxis_title='Recall',
                            yaxis_title='Precision')
        st.plotly_chart(fig_pr, use_container_width=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                      title="Confusion Matrix",
                      labels=dict(x="Predicted", y="Actual", color="Count"))
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Calibration Curve
    if len(np.unique(y_true)) == 2:  # Only for binary classification
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model'))
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   line=dict(dash='dash'), name='Ideal'))
        fig_cal.update_layout(title='Calibration Curve',
                             xaxis_title='Mean Predicted Probability',
                             yaxis_title='Fraction of Positives')
        st.plotly_chart(fig_cal, use_container_width=True)
    
    # Class Distribution
    fig_dist = px.bar(x=list(metrics['class_distribution'].keys()), 
                     y=list(metrics['class_distribution'].values()),
                     title=f"{target_name} Distribution")
    fig_dist.update_layout(xaxis_title='Class', yaxis_title='Count')
    st.plotly_chart(fig_dist, use_container_width=True)

def plot_feature_importance(importance_df: pd.DataFrame):
    """Visualize feature importance"""
    fig = px.bar(importance_df.head(20), 
                x='Importance', y='Feature', 
                orientation='h',
                title="Top Feature Importance")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with target
    st.subheader("Feature-Target Correlation")
    fig_corr = px.bar(importance_df.sort_values('Correlation', ascending=False).head(20),
                     x='Correlation', y='Feature', orientation='h')
    fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_corr, use_container_width=True)

def visualize_embeddings(X_emb: np.ndarray, y: np.ndarray, method: str = "PCA"):
    """Visualize high-dimensional data in 2D"""
    if method == "PCA":
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(X_emb)
        title = "PCA Projection"
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        emb = reducer.fit_transform(X_emb)
        title = "t-SNE Projection"
        
    fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=y,
                   title=title, labels={'color': 'Target'})
    st.plotly_chart(fig, use_container_width=True)

# ======================== HELPER FUNCTIONS FOR ROBUST TYPE CONVERSION ========================
def to_pyint(x):
    if hasattr(x, 'item'):
        return int(x.item())
    if isinstance(x, (np.generic, np.ndarray)):
        return int(np.asarray(x))
    return int(x)

def to_py_tuple(t):
    if isinstance(t, (np.ndarray, jnp.ndarray)):
        t = np.asarray(t)
    return tuple(to_pyint(x) for x in t)

def to_py_str(s):
    if isinstance(s, (np.generic, np.ndarray, jnp.ndarray)):
        return str(np.asarray(s))
    return str(s)

# --- Data Cleaning with Polars and JAX ---
# Used only for Gradio interface

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    try:
        # Clean 'Price' column
        if 'Price' in df.columns:
            df = df.with_columns(
                pl.col('Price')
                .str.replace_all(r'[^0-9.]', '')
                .cast(pl.Float64)
                .fill_null(0.0)
                .alias('Price')
            )
        # Clean 'Rating' column
        if 'Rating' in df.columns:
            rating_map = {'bad': 1, 'poor': 2, 'ok': 3, 'average': 3, 'good': 4, 'excellent': 5}
            df = df.with_columns(
                pl.col('Rating')
                .apply(lambda x: rating_map.get(str(x).lower(), float(x) if str(x).replace('.','',1).isdigit() else None))
                .cast(pl.Float64)
                .alias('Rating')
            )
        # Fill missing values with mean using JAX
        for col in df.columns:
            if df[col].dtype in [pl.Float64, pl.Int64]:
                col_data = jnp.array(df[col].to_numpy())
                mean_val = float(jnp.nanmean(col_data))
                df = df.with_columns(
                    pl.col(col).fill_null(mean_val)
                )
        return df
    except Exception as e:
        print(f"Error in clean_data: {e}")
        return df

# --- Modern Entrypoint ---
def main():
    st.set_page_config(page_title="🔧 AI Data Cleaner & Evaluator", layout="wide")
    st.title("🔧 AI-Powered Data Cleaner & Evaluator")
    st.markdown("Enhanced with Automated Preprocessing and Visualization")

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

            # --- Automated Target Selection: last column if categorical, else first categorical ---
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 0:
                target_col = cat_cols[-1]
            else:
                target_col = df.columns[-1]
            st.info(f"Automatically selected target column: {target_col}")

            # --- Automated Data Processing ---
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
            col1.metric("Outliers Detected", outlier_count)
            col2.metric("Outlier Percentage", f"{outlier_percentage:.2f}%")
            col3.metric("Missing Values", stats.get('missing_values_processed', 0))

            with st.expander("Cleaned Data Preview"):
                try:
                    st.dataframe(pd.DataFrame(X_clean_np, columns=X_clean.columns).head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying cleaned data: {e}")

            # --- Download Cleaned Data ---
            try:
                cleaned_df = pd.DataFrame(X_clean_np, columns=X_clean.columns)
                cleaned_df[target_col] = y_clean_np
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Dataset as CSV",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing download: {e}")

            st.subheader("Target Variable Preview")
            try:
                st.write(pd.Series(y_clean_np).value_counts())
            except Exception as e:
                st.error(f"Error displaying target variable: {e}")

            st.success("✅ Data cleaning complete! No model training performed.")

            st.subheader("🔍 Visualize High-Dimensional Data")
            # Automated: Use PCA if features > 10, else t-SNE
            vis_method = "PCA" if X_clean_np.shape[1] > 10 else "t-SNE"
            st.info(f"Automatically selected visualization method: {vis_method}")
            try:
                visualize_embeddings(X_clean_np, y_clean_np, method=vis_method)
            except Exception as e:
                st.error(f"Error in embedding visualization: {e}")

            # --- Modern Visualization and Reporting ---
            st.subheader("📈 Advanced Data Profiling")
            try:
                # Convert to pandas DataFrame first to set columns, then to Polars
                df_cleaned_pd = pd.DataFrame(X_clean_np, columns=X_clean.columns)
                df_cleaned = pl.from_pandas(df_cleaned_pd)
                if 'Price' in df_cleaned.columns:
                    df_cleaned = df_cleaned.with_columns(
                        ((pl.col('Price') - pl.col('Price').mean()) / pl.col('Price').std()).alias('Price')
                    )
                fig_price_dist = px.histogram(df_cleaned.to_pandas(), x="Price", title="Normalized Price Distribution")
                st.plotly_chart(fig_price_dist, use_container_width=True)
            except Exception as e:
                st.error(f"Error in price distribution plot: {e}")
            st.subheader("🔗 Feature Correlation Heatmap")
            try:
                corr = df_cleaned.corr()
                fig_corr_heatmap = px.imshow(corr.to_pandas(), text_auto=True, aspect="auto", 
                                            title="Feature Correlation Heatmap",
                                            labels=dict(x="Features", y="Features", color="Correlation"))
                st.plotly_chart(fig_corr_heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"Error in correlation heatmap: {e}")
            st.subheader("🔍 Feature Pair Plot")
            try:
                fig_pair_plot = sns.pairplot(df_cleaned.to_pandas())
                st.pyplot(fig_pair_plot)
            except Exception as e:
                st.error(f"Error in pair plot: {e}")

            st.subheader("📊 Gradio Data Summary")
            def gradio_summary(file):
                try:
                    df_gradio = pl.read_csv(file.name)
                    df_clean_gradio = clean_data(df_gradio)
                    summary_gradio = df_clean_gradio.describe().to_pandas().to_markdown()
                    return summary_gradio
                except Exception as e:
                    return f"Error in Gradio summary: {e}"
            try:
                gr.Interface(
                    fn=gradio_summary,
                    inputs=gr.File(label="Upload CSV"),
                    outputs=gr.Markdown(label="Data Summary"),
                    title="Gradio Data Summary Example"
                ).launch(inline=True)
            except Exception as e:
                st.error(f"Error launching Gradio interface: {e}")
        except Exception as e:
            st.error(f"❌ Error loading or processing data: {e}")
            return

    # About section
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app is powered by Streamlit, JAX, and Plotly. "
        "It provides an end-to-end solution for data cleaning and visualization using advanced techniques."
    )
    st.sidebar.subheader("GitHub Repository")
    st.sidebar.markdown(
        "[View Source Code](https://github.com/yourusername/your-repo)"
    )

if __name__ == "__main__":
    main()

# Install command for required packages
# pip install polars jax plotly matplotlib seaborn gradio dash dash-bootstrap-components
