"""
Streamlit UI for AI Data Cleaner & Web Scraper.
Provides tabs for Upload/Connect, Quality Report, Explore, Web Summaries, Explain, and Downloads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil
from datetime import datetime
import logging

from core.data_cleaner import DataCleaner, CleaningConfig, CleaningResult
from core.web_scraper import WebScraper
from core.quality_gates import QualityGates

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Data Cleaner & Web Scraper",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🔧 AI Data Cleaner & Web Scraper</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'cleaning_result' not in st.session_state:
        st.session_state.cleaning_result = None
    if 'scraping_results' not in st.session_state:
        st.session_state.scraping_results = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'audit_log' not in st.session_state:
        st.session_state.audit_log = None
    
    # Add troubleshooting info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🆘 Need Help?")
        if st.button("Run Quick Diagnostic"):
            st.info("Run: `python quick_start.py` in terminal")
        if st.button("View Troubleshooting Guide"):
            st.info("See: TROUBLESHOOTING.md")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data cleaning settings
        st.subheader("Data Cleaning Settings")
        domain = st.selectbox(
            "Data Domain",
            ["general", "healthcare", "finance", "retail"],
            help="Select domain for specialized cleaning rules"
        )
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Chunk size for streaming processing"
        )
        
        max_memory = st.slider(
            "Max Memory (MB)",
            min_value=512,
            max_value=4096,
            value=1024,
            step=256,
            help="Maximum memory usage"
        )
        
        output_format = st.selectbox(
            "Output Format",
            ["parquet", "csv", "both"],
            help="Output format for cleaned data"
        )
        
        enable_quality_gates = st.checkbox(
            "Enable Quality Gates",
            value=True,
            help="Enable data quality validation"
        )
        
        enable_quarantine = st.checkbox(
            "Enable Quarantine",
            value=True,
            help="Quarantine problematic rows"
        )
        
        # Web scraping settings
        st.subheader("Web Scraping Settings")
        max_concurrent = st.slider(
            "Max Concurrent Requests",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum concurrent web requests"
        )
        
        timeout = st.slider(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=60,
            value=30,
            help="Timeout for web requests"
        )
        
        retry_attempts = st.slider(
            "Retry Attempts",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of retry attempts"
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Upload/Connect",
        "📊 Quality Report", 
        "🔍 Explore",
        "🌐 Web Summaries",
        "🔍 Explain (Audit)",
        "📥 Downloads"
    ])
    
    with tab1:
        upload_connect_tab(domain, chunk_size, max_memory, output_format, enable_quality_gates, enable_quarantine)
    
    with tab2:
        quality_report_tab()
    
    with tab3:
        explore_tab()
    
    with tab4:
        web_summaries_tab(max_concurrent, timeout, retry_attempts)
    
    with tab5:
        explain_audit_tab()
    
    with tab6:
        downloads_tab()

def upload_connect_tab(domain: str, chunk_size: int, max_memory: int, output_format: str, 
                      enable_quality_gates: bool, enable_quarantine: bool):
    """Upload/Connect tab for data input."""
    st.header("📁 Upload/Connect Data")
    
    # Add file format guidance
    st.info("💡 **Supported formats:** CSV, Parquet | **Max size:** 200MB | **Encoding:** UTF-8 recommended")
    
    # File upload section
    st.subheader("Upload Data File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Parquet file",
        type=['csv', 'parquet'],
        help="Upload your data file for cleaning"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and preview data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_parquet(tmp_path)
            
            st.session_state.current_data = df
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            with st.expander("Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Run cleaning button
            st.subheader("Run Data Cleaning")
            
            if st.button("🚀 Run Cleaning", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    # Create cleaning configuration
                    config = CleaningConfig(
                        chunk_size=chunk_size,
                        max_memory_mb=max_memory,
                        enable_quality_gates=enable_quality_gates,
                        enable_quarantine=enable_quarantine,
                        output_format=output_format
                    )
                    
                    # Initialize data cleaner
                    cleaner = DataCleaner(config)
                    
                    # Create output directory
                    output_dir = tempfile.mkdtemp(prefix='cleaning_output_')
                    
                    # Clean the data
                    result = cleaner.clean_file(tmp_path, output_dir, domain)
                    
                    if result.success:
                        st.session_state.cleaning_result = result
                        st.success("✅ Data cleaning completed successfully!")
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rows Processed", f"{result.total_rows_processed:,}")
                        with col2:
                            st.metric("Rows Cleaned", f"{result.total_rows_cleaned:,}")
                        with col3:
                            st.metric("Rows Quarantined", f"{result.total_rows_quarantined:,}")
                        with col4:
                            st.metric("Processing Time", f"{result.processing_time:.2f}s")
                        
                        # Load audit log
                        if result.audit_log_path:
                            with open(result.audit_log_path, 'r') as f:
                                st.session_state.audit_log = json.load(f)
                    else:
                        st.error(f"❌ Data cleaning failed: {result.error_message}")
            
            # Clean up temporary file
            Path(tmp_path).unlink()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Database connection section (placeholder)
    st.subheader("Connect to Database")
    st.info("Database connection feature coming soon. Currently supports file upload only.")

def quality_report_tab():
    """Quality Report tab for data quality analysis."""
    st.header("📊 Data Quality Report")
    
    if st.session_state.cleaning_result is None:
        st.info("Please upload and clean data first to view quality report.")
        return
    
    result = st.session_state.cleaning_result
    
    if not result.quality_report_path or not Path(result.quality_report_path).exists():
        # Try to locate a quality report in the same output directory as audit log
        guess_path = None
        if result.audit_log_path:
            try:
                guess_path = str(Path(result.audit_log_path).with_name('quality_report.json'))
            except Exception:
                guess_path = None
        if guess_path and Path(guess_path).exists():
            result.quality_report_path = guess_path
        else:
            st.warning("Quality report not available. Quality gates may have been disabled or report not generated.")
            return
    
    try:
        # Load quality report
        with open(result.quality_report_path, 'r') as f:
            quality_metrics = json.load(f)
        
        # Display quality score
        quality_score = quality_metrics.get('quality_score', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality Score", f"{quality_score:.1%}")
        with col2:
            st.metric("Total Violations", quality_metrics.get('total_violations', 0))
        with col3:
            st.metric("Errors", quality_metrics.get('error_count', 0))
        with col4:
            st.metric("Warnings", quality_metrics.get('warning_count', 0))
        
        # Quality score visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = quality_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Data Quality Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Violations breakdown
        if quality_metrics.get('total_violations', 0) > 0:
            st.subheader("Violations Breakdown")
            
            violations_data = {
                'Type': ['Errors', 'Warnings', 'Info'],
                'Count': [
                    quality_metrics.get('error_count', 0),
                    quality_metrics.get('warning_count', 0),
                    quality_metrics.get('info_count', 0)
                ]
            }
            
            fig_violations = px.bar(
                pd.DataFrame(violations_data),
                x='Type',
                y='Count',
                title="Violations by Type",
                color='Type',
                color_discrete_map={'Errors': 'red', 'Warnings': 'orange', 'Info': 'blue'}
            )
            st.plotly_chart(fig_violations, use_container_width=True)
        
        # Domain information
        if quality_metrics.get('domain'):
            st.subheader("Domain Information")
            st.info(f"Detected domain: **{quality_metrics['domain']}**")
        
    except Exception as e:
        st.error(f"Error loading quality report: {e}")

def explore_tab():
    """Explore tab for post-cleaning data exploration."""
    st.header("🔍 Post-Cleaning Data Exploration")
    
    if st.session_state.cleaning_result is None:
        st.info("Please upload and clean data first to explore results.")
        st.markdown("### Quick Test")
        if st.button("Create Sample Data"):
            try:
                sample_data = {
                    'id': ['ID001', 'ID002', 'ID003'],
                    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                    'email': ['john@email.com', 'jane@email.com', 'bob@company.org'],
                    'age': [25, 30, 45],
                    'status': ['Delivered', 'DELIVERED', 'delivered']
                }
                df = pd.DataFrame(sample_data)
                st.session_state.current_data = df
                st.success("Sample data created! You can now run cleaning.")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error creating sample data: {e}")
        return
    
    result = st.session_state.cleaning_result
    
    try:
        # Load cleaned data with robust handling of format
        if result.cleaned_data_path and Path(result.cleaned_data_path).exists():
            try:
                if result.cleaned_data_path.endswith('.parquet'):
                    df_clean = pd.read_parquet(result.cleaned_data_path)
                elif result.cleaned_data_path.endswith('.csv'):
                    df_clean = pd.read_csv(result.cleaned_data_path)
                else:
                    # Fallback: try parquet then csv
                    try:
                        df_clean = pd.read_parquet(result.cleaned_data_path)
                    except Exception:
                        df_clean = pd.read_csv(result.cleaned_data_path)
            except ValueError as ve:
                if "No columns to parse from file" in str(ve):
                    st.warning("Cleaned data file is empty or has no columns to parse. Nothing to explore yet.")
                    df_clean = pd.DataFrame()
                else:
                    raise
            except Exception as e:
                # Final fallback: if extension mismatch, try the alternate file in the same directory
                cleaned_path = Path(result.cleaned_data_path)
                alt_path_parquet = cleaned_path.with_suffix('.parquet')
                alt_path_csv = cleaned_path.with_suffix('.csv')
                try:
                    if alt_path_parquet.exists():
                        try:
                            df_clean = pd.read_parquet(alt_path_parquet)
                        except ValueError as ve:
                            if "No columns to parse from file" in str(ve):
                                st.warning("Cleaned data file is empty or has no columns to parse. Nothing to explore yet.")
                                df_clean = pd.DataFrame()
                            else:
                                raise
                        result.cleaned_data_path = str(alt_path_parquet)
                    elif alt_path_csv.exists():
                        try:
                            df_clean = pd.read_csv(alt_path_csv)
                        except ValueError as ve:
                            if "No columns to parse from file" in str(ve):
                                st.warning("Cleaned data file is empty or has no columns to parse. Nothing to explore yet.")
                                df_clean = pd.DataFrame()
                            else:
                                raise
                        result.cleaned_data_path = str(alt_path_csv)
                    else:
                        raise e
                except ValueError as ve2:
                    st.error(f"Error exploring data: {ve2}")
                    return
                except Exception as e2:
                    st.error(f"Error exploring data: {e2}")
                    return

            st.subheader("Cleaned Data Overview")

            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df_clean):,}")
            with col2:
                st.metric("Columns", len(df_clean.columns))
            with col3:
                st.metric("Missing Values", df_clean.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df_clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df_clean.head(10), use_container_width=True)

            if df_clean.empty or len(df_clean.columns) == 0:
                st.info("Cleaned dataset is empty after processing. Adjust cleaning settings or review quarantine results.")
                return

            # Column analysis
            st.subheader("Column Analysis")
            
            # Separate numeric and categorical columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numeric Columns:**")
                for col in numeric_cols:
                    st.write(f"- {col}")
            
            with col2:
                st.write("**Categorical Columns:**")
                for col in categorical_cols:
                    st.write(f"- {col}")
            
            # Visualizations
            if numeric_cols:
                st.subheader("Numeric Columns Distribution")
                
                # Select columns to visualize
                selected_numeric = st.multiselect(
                    "Select numeric columns to visualize",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if selected_numeric:
                    n_cols = min(3, len(selected_numeric))
                    n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
                    
                    for i in range(0, len(selected_numeric), n_cols):
                        cols = st.columns(n_cols)
                        for j, col in enumerate(cols):
                            if i + j < len(selected_numeric):
                                feature = selected_numeric[i + j]
                                with col:
                                    fig = px.histogram(df_clean, x=feature, title=f"{feature} Distribution")
                                    st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                st.subheader("Categorical Columns Distribution")
                
                # Select columns to visualize
                selected_categorical = st.multiselect(
                    "Select categorical columns to visualize",
                    categorical_cols,
                    default=categorical_cols[:3] if len(categorical_cols) >= 3 else categorical_cols
                )
                
                if selected_categorical:
                    for feature in selected_categorical:
                        value_counts = df_clean[feature].value_counts().head(10)
                        fig = px.bar(
                            x=value_counts.values,
                            y=value_counts.index,
                            orientation='h',
                            title=f"{feature} Distribution (Top 10)",
                            labels={'x': 'Count', 'y': feature}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df_clean[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix of Numeric Features",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Cleaned data file not found.")
    
    except Exception as e:
        st.error(f"Error exploring data: {e}")

def web_summaries_tab(max_concurrent: int, timeout: int, retry_attempts: int):
    """Web Summaries tab for web scraping and summarization."""
    st.header("🌐 Web Scraping & Summarization")
    
    # URL input
    st.subheader("Enter URLs to Scrape")
    
    url_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com\nhttps://another-site.com",
        height=100
    )
    
    if url_input:
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        st.write(f"Found {len(urls)} URLs to scrape")
        
        if st.button("🚀 Start Scraping", type="primary"):
            with st.spinner("Scraping URLs..."):
                try:
                    # Initialize web scraper
                    scraper = WebScraper(
                        max_concurrent=max_concurrent,
                        timeout=timeout,
                        retry_attempts=retry_attempts
                    )
                    
                    # Scrape URLs
                    results = asyncio.run(scraper.scrape_urls(urls))
                    st.session_state.scraping_results = results
                    
                    # Display results
                    successful = sum(1 for r in results if r.success)
                    failed = len(results) - successful
                    
                    st.success(f"✅ Scraping completed: {successful} successful, {failed} failed")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total URLs", len(urls))
                    with col2:
                        st.metric("Successful", successful)
                    with col3:
                        st.metric("Failed", failed)
                    
                except Exception as e:
                    st.error(f"Error during scraping: {e}")
    
    # Display scraping results
    if st.session_state.scraping_results:
        st.subheader("Scraping Results")
        
        results = st.session_state.scraping_results
        
        # Results table
        results_data = []
        for result in results:
            results_data.append({
                'URL': result.url,
                'Status': '✅ Success' if result.success else '❌ Failed',
                'Title': result.title[:50] + '...' if len(result.title) > 50 else result.title,
                'Response Time': f"{result.response_time:.2f}s" if result.response_time else 'N/A',
                'Error': result.error_message if not result.success else ''
            })
        
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
        
        # Detailed results
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1}: {result.url}"):
                if result.success:
                    st.write(f"**Title:** {result.title}")
                    st.write(f"**Response Time:** {result.response_time:.2f}s")
                    
                    if result.summary:
                        st.write("**Summary:**")
                        st.write(result.summary)
                    
                    if result.structured_cards:
                        st.write("**Structured Cards:**")
                        for card in result.structured_cards:
                            st.write(f"- **{card.card_type.title()}:** {card.title}")
                            if card.content:
                                st.write(f"  Content: {card.content[0][:100]}...")
                else:
                    st.error(f"Failed: {result.error_message}")

def explain_audit_tab():
    """Explain (Audit) tab for detailed audit information."""
    st.header("🔍 Explain (Audit)")
    
    if st.session_state.audit_log is None:
        st.info("Please upload and clean data first to view audit information.")
        return
    
    # Attempt to load audit log from file if session missing
    audit_log = st.session_state.audit_log
    if (audit_log is None) and st.session_state.cleaning_result and st.session_state.cleaning_result.audit_log_path:
        try:
            with open(st.session_state.cleaning_result.audit_log_path, 'r') as f:
                audit_log = json.load(f)
                st.session_state.audit_log = audit_log
        except Exception:
            audit_log = None
    
    # Audit summary
    st.subheader("Audit Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", len(audit_log.get('decisions', [])))
    with col2:
        st.metric("Input Shape", f"{audit_log.get('input_shape', (0, 0))[0]} × {audit_log.get('input_shape', (0, 0))[1]}")
    with col3:
        st.metric("Output Shape", f"{audit_log.get('output_shape', (0, 0))[0]} × {audit_log.get('output_shape', (0, 0))[1]}")
    with col4:
        st.metric("Quarantined Rows", audit_log.get('quarantined_rows', 0))
    
    # Detailed decisions
    st.subheader("Detailed Cleaning Decisions")
    
    decisions = audit_log.get('decisions', [])
    
    if decisions:
        for i, decision in enumerate(decisions, 1):
            with st.expander(f"Decision {i}: {decision.get('column', 'Unknown')} - {decision.get('strategy', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Column Information:**")
                    st.write(f"- Column: {decision.get('column', 'Unknown')}")
                    st.write(f"- Type: {decision.get('column_type', 'Unknown')}")
                    st.write(f"- Strategy: {decision.get('strategy', 'Unknown')}")
                    st.write(f"- Confidence: {decision.get('confidence', 0):.2f}")
                
                with col2:
                    st.write("**Rules Applied:**")
                    for rule in decision.get('rules_applied', []):
                        st.write(f"- {rule}")
                
                # Before/After samples
                if decision.get('before_sample') and decision.get('after_sample'):
                    st.write("**Before/After Samples:**")
                    
                    sample_data = {
                        'Before': decision['before_sample'][:5],
                        'After': decision['after_sample'][:5]
                    }
                    
                    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
                
                # Quarantine reason
                if decision.get('quarantine_reason'):
                    st.warning(f"Quarantine Reason: {decision['quarantine_reason']}")
    else:
        st.info("No detailed decisions available in audit log. Ensure cleaning completed successfully and audit was written.")

def downloads_tab():
    """Downloads tab for exporting results."""
    st.header("📥 Download Results")
    
    # Data cleaning results
    if st.session_state.cleaning_result:
        st.subheader("Data Cleaning Results")
        
        result = st.session_state.cleaning_result
        
        col1, col2 = st.columns(2)
        
        with col1:
            if result.cleaned_data_path and Path(result.cleaned_data_path).exists():
                cleaned_path = Path(result.cleaned_data_path)
                file_bytes = cleaned_path.read_bytes()
                if cleaned_path.suffix.lower() == '.csv':
                    st.download_button(
                        label="📊 Download Cleaned Data (CSV)",
                        data=file_bytes,
                        file_name=cleaned_path.name,
                        mime="text/csv"
                    )
                else:
                    st.download_button(
                        label="📊 Download Cleaned Data (Parquet)",
                        data=file_bytes,
                        file_name=cleaned_path.name,
                        mime="application/octet-stream"
                    )
            
            if result.quarantined_data_path and Path(result.quarantined_data_path).exists():
                q_path = Path(result.quarantined_data_path)
                q_bytes = q_path.read_bytes()
                if q_path.suffix.lower() == '.csv':
                    st.download_button(
                        label="🚫 Download Quarantined Data (CSV)",
                        data=q_bytes,
                        file_name=q_path.name,
                        mime="text/csv"
                    )
                else:
                    st.download_button(
                        label="🚫 Download Quarantined Data (Parquet)",
                        data=q_bytes,
                        file_name=q_path.name,
                        mime="application/octet-stream"
                    )
        
        with col2:
            if result.audit_log_path and Path(result.audit_log_path).exists():
                with open(result.audit_log_path, 'rb') as f:
                    st.download_button(
                        label="📋 Download Audit Log",
                        data=f.read(),
                        file_name="audit_log.json",
                        mime="application/json"
                    )
            
            if result.dqr_path and Path(result.dqr_path).exists():
                with open(result.dqr_path, 'rb') as f:
                    st.download_button(
                        label="📊 Download Quality Report",
                        data=f.read(),
                        file_name="dqr.md",
                        mime="text/markdown"
                    )
    
    # Web scraping results
    if st.session_state.scraping_results:
        st.subheader("Web Scraping Results")
        
        # Create summary file
        summary_data = []
        for result in st.session_state.scraping_results:
            summary_data.append({
                'URL': result.url,
                'Title': result.title,
                'Success': result.success,
                'Summary': result.summary[:200] + '...' if len(result.summary) > 200 else result.summary,
                'Error': result.error_message if not result.success else ''
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Scraping Summary",
            data=summary_csv,
            file_name="scraping_summary.csv",
            mime="text/csv"
        )
    
    # Combined download
    if st.session_state.cleaning_result or st.session_state.scraping_results:
        st.subheader("Download All Results")
        st.info("Use the individual download buttons above to get specific files, or download all results as a ZIP file.")
        
        # Note: In a real implementation, you would create a ZIP file with all results
        st.warning("ZIP download feature requires server-side implementation.")

if __name__ == "__main__":
    main()
