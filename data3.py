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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
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
import json
import logging
from datetime import datetime
from dateutil import parser as date_parser
import re
from collections import defaultdict
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

# ======================== CACHING AND UTILITY FUNCTIONS ========================

def get_cleaner_instance():
    """Get JAXDataCleaner instance"""
    return JAXDataCleaner()

@st.cache_data
def load_data_from_csv(uploaded_file):
    """Cached data loading function"""
    return pd.read_csv(uploaded_file)

def make_arrow_compatible(df):
    """Convert DataFrame to be Arrow-compatible for Streamlit display"""
    df_clean = df.copy()
    
    # Convert object columns to string to avoid Arrow serialization issues
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str)
    
    # Handle mixed types by converting to string
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                # Try to convert to numeric first
                pd.to_numeric(df_clean[col], errors='raise')
            except (ValueError, TypeError):
                # If conversion fails, keep as string
                df_clean[col] = df_clean[col].astype(str)
    
    return df_clean

def analyze_data_quality(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """Analyze data quality issues and provide recommendations"""
    issues = {}
    recommendations = []
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        issues['missing_values'] = missing_data[missing_data > 0].to_dict()
        recommendations.append("Consider handling missing values before training")
    
    # Check for case inconsistencies in categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    case_issues = {}
    
    for col in categorical_cols:
        if col == target_col:
            continue
        values = df[col].dropna().astype(str)
        unique_lower = set(values.str.lower().unique())
        unique_original = set(values.unique())
        
        if len(unique_lower) < len(unique_original):
            case_issues[col] = {
                'original_unique': len(unique_original),
                'normalized_unique': len(unique_lower),
                'examples': list(unique_original)[:10]  # Show first 10 examples
            }
    
    if case_issues:
        issues['case_inconsistencies'] = case_issues
        recommendations.append("Normalize case in categorical columns to merge duplicate categories")
    
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues['duplicate_rows'] = duplicate_rows
        recommendations.append(f"Consider removing {duplicate_rows} duplicate rows")
    
    return {
        'issues': issues,
        'recommendations': recommendations,
        'total_issues': len(issues)
    }

def demonstrate_status_cleaning():
    """Demonstrate how the enhanced cleaner handles STATUS column issues"""
    st.subheader("🎯 STATUS Column Cleaning Demo")
    st.markdown("""
    **Your specific case:**
    - Delivered (57) + DELIVERED (31) → **delivered** (88)
    - Pending (51) + PENDING (25) → **pending** (76)  
    - Returned (56) + RETURNED (21) → **returned** (77)
    - Cancelled (45) + CANCELLED (17) → **cancelled** (62)
    - 17 NaN values → **unknown** (17)
    
    **Result:** 8 variants + NaN → **5 clean categories**
    """)
    
    # Create a sample to demonstrate
    sample_data = pd.DataFrame({
        'STATUS': ['Delivered', 'DELIVERED', 'Pending', 'PENDING', 'Returned', 'RETURNED', 
                   'Cancelled', 'CANCELLED', None, None] * 32  # 320 rows
    })
    
    st.write("**Before cleaning:**")
    st.write(sample_data['STATUS'].value_counts(dropna=False))
    
    # Apply the cleaner
    cleaner = JAXDataCleaner()
    cleaned_status = cleaner._clean_status_column(sample_data['STATUS'])
    
    st.write("**After cleaning:**")
    st.write(cleaned_status.value_counts(dropna=False))
    
    st.success("✅ Case normalization reduces 8 variants to 4 clean categories!")

def demonstrate_comprehensive_cleaning():
    """Demonstrate comprehensive data cleaning with audit logging"""
    st.subheader("🔧 Comprehensive Data Cleaning Demo")
    st.markdown("""
    **Sample messy dataset with common issues:**
    - Mixed case categorical values
    - Inconsistent date formats
    - Invalid email addresses
    - Corrupted numeric values
    - Missing values
    """)
    
    # Create a comprehensive sample dataset
    sample_data = pd.DataFrame({
        'Customer ID': ['C001', 'C002', 'C003', 'C004', 'C005'] * 20,
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'] * 20,
        'Email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net'] * 20,
        'Phone': ['+1-555-123-4567', '(555) 987-6543', '555.111.2222', '555-333-4444', 'invalid-phone'] * 20,
        'Age': [25, 30, 'invalid', 35, 28] * 20,
        'Status': ['Delivered', 'DELIVERED', 'Pending', 'PENDING', 'Cancelled'] * 20,
        'Order Date': ['2024-01-15', '15/01/2024', 'Jan 15, 2024', '2024-01-15', '15-01-2024'] * 20,
        'Price': ['$25.99', '30.50', 'invalid', '45.00', 'N/A'] * 20,
        'Rating': ['Excellent', 'Good', 'Average', 'Poor', 'N/A'] * 20
    })
    
    st.write("**Original messy data (first 10 rows):**")
    st.dataframe(sample_data.head(10), use_container_width=True)
    
    # Apply comprehensive cleaning
    cleaner = JAXDataCleaner()
    cleaner.set_parameters(
        outlier_threshold=3.0,
        imputation_method="mean",
        categorical_missing_strategy="unknown"
    )
    
    with st.spinner("Applying comprehensive cleaning pipeline..."):
        cleaned_data, stats = cleaner.preprocess_data(sample_data, outlier_strategy="cap")
    
    st.write("**After comprehensive cleaning (first 10 rows):**")
    st.dataframe(cleaned_data.head(10), use_container_width=True)
    
    # Display cleaning statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Shape", f"{stats.get('original_shape', (0,0))[0]} × {stats.get('original_shape', (0,0))[1]}")
    with col2:
        st.metric("Cleaned Shape", f"{stats.get('processed_shape', (0,0))[0]} × {stats.get('processed_shape', (0,0))[1]}")
    with col3:
        st.metric("Missing Values Fixed", stats.get('missing_values_original', 0) - stats.get('missing_values_processed', 0))
    with col4:
        st.metric("Domain Detected", stats.get('domain', 'Unknown'))
    
    # Display cleaning log
    cleaning_log = cleaner.get_cleaning_log()
    
    with st.expander("View Detailed Cleaning Log"):
        st.subheader("🔄 Transformation Steps")
        for i, step in enumerate(cleaning_log.get('steps', []), 1):
            with st.expander(f"Step {i}: {step.get('operation', 'Unknown')} - {step.get('column', 'Unknown')}"):
                st.json(step)
        
        if cleaning_log.get('warnings'):
            st.subheader("⚠️ Warnings")
            for warning in cleaning_log['warnings']:
                st.warning(warning)
    
    st.success("✅ Comprehensive cleaning complete with full audit trail!")

def detect_target_type(y_series: pd.Series, threshold: float = 0.05) -> str:
    """Automatically detect if target is continuous (regression) or discrete (classification)"""
    try:
        # Check if it's numeric
        if not pd.api.types.is_numeric_dtype(y_series):
            return "classification"
        
        # Remove NaN values for analysis
        y_clean = y_series.dropna()
        
        if len(y_clean) == 0:
            return "classification"  # Default fallback
        
        # Check if values are all integers (discrete)
        if all(y_clean == y_clean.astype(int)):
            # Check if number of unique values is small relative to total
            unique_ratio = len(y_clean.unique()) / len(y_clean)
            if unique_ratio <= threshold:
                return "classification"
        
        # Check for very few unique values (likely categorical)
        if len(y_clean.unique()) <= 10:
            return "classification"
        
        return "regression"
    except Exception:
        return "classification"  # Default fallback

def get_model_configs():
    """Return available models and their configurations"""
    return {
        "classification": {
            "Logistic Regression": {
                "model": LogisticRegression,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "max_iter": [100, 500, 1000],
                    "solver": ["liblinear", "lbfgs"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "Support Vector Machine": {
                "model": SVC,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier,
                "params": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"]
                }
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier,
                "params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"]
                }
            }
        },
        "regression": {
            "Linear Regression": {
                "model": LinearRegression,
                "params": {}
            },
            "Ridge Regression": {
                "model": Ridge,
                "params": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                }
            },
            "Lasso Regression": {
                "model": Lasso,
                "params": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                }
            },
            "Random Forest": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor,
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "Support Vector Regression": {
                "model": SVR,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsRegressor,
                "params": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"]
                }
            }
        }
    }

# ======================== ENHANCED DATA CLEANER ========================
class JAXDataCleaner:
    """Enhanced JAX-powered data cleaning pipeline with comprehensive audit logging"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        self.outlier_threshold = 3.0
        self.imputation_method = "mean"
        self.categorical_missing_strategy = "unknown"
        
        # Comprehensive audit logging
        self.cleaning_log = {
            'timestamp': datetime.now().isoformat(),
            'steps': [],
            'transformations': {},
            'statistics': {},
            'warnings': [],
            'errors': []
        }
        
        # Domain-specific cleaning rules
        self.domain_rules = {
            'healthcare': {
                'symptoms': ['fever', 'cough', 'headache', 'nausea', 'fatigue'],
                'diagnoses': ['hypertension', 'diabetes', 'asthma', 'pneumonia'],
                'medications': ['aspirin', 'ibuprofen', 'acetaminophen']
            },
            'finance': {
                'currencies': ['usd', 'eur', 'gbp', 'jpy', 'cad', 'aud'],
                'account_types': ['checking', 'savings', 'credit', 'investment'],
                'transaction_types': ['deposit', 'withdrawal', 'transfer', 'payment']
            },
            'retail': {
                'categories': ['electronics', 'clothing', 'books', 'home', 'sports'],
                'statuses': ['delivered', 'pending', 'cancelled', 'returned', 'shipped']
            }
        }
        
        # Special cleaners for known messy columns
        self.special_cleaners = {
            'Price': self._clean_price_column,
            'Rating': self._clean_rating_column,
            'Status': self._clean_status_column,
            'STATUS': self._clean_status_column,
            'Date': self._clean_date_column,
            'DATE': self._clean_date_column,
            'Age': self._clean_age_column,
            'AGE': self._clean_age_column,
            'Email': self._clean_email_column,
            'EMAIL': self._clean_email_column,
            'Phone': self._clean_phone_column,
            'PHONE': self._clean_phone_column,
            'Score': self._clean_score_column,
            'SCORE': self._clean_score_column,
            'AdmissionDate': self._clean_date_column,
            'ADMISSIONDATE': self._clean_date_column,
            'Admission_Date': self._clean_date_column,
            'ADMISSION_DATE': self._clean_date_column
        }
        
        # Synonym mappings for categorical standardization
        self.synonym_mappings = {
            'status': {
                'delivered': ['delivered', 'completed', 'shipped', 'fulfilled'],
                'pending': ['pending', 'processing', 'in_progress', 'awaiting'],
                'cancelled': ['cancelled', 'canceled', 'failed', 'aborted'],
                'returned': ['returned', 'refunded', 'rejected']
            },
            'gender': {
                'male': ['male', 'm', 'man', 'masculine'],
                'female': ['female', 'f', 'woman', 'feminine'],
                'other': ['other', 'non-binary', 'nb', 'prefer_not_to_say']
            },
            'rating': {
                'excellent': ['excellent', 'outstanding', 'amazing', 'perfect', '5'],
                'good': ['good', 'great', 'satisfactory', '4'],
                'average': ['average', 'ok', 'okay', 'fair', '3'],
                'poor': ['poor', 'bad', 'terrible', 'awful', '2'],
                'very_poor': ['very poor', 'worst', 'horrible', '1']
            }
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

    def _clean_status_column(self, series):
        """Clean STATUS column by normalizing case and handling missing values"""
        original_unique = series.nunique()
        original_missing = series.isnull().sum()
        
        def clean_status(val):
            try:
                if pd.isnull(val): 
                    return np.nan  # Keep NaN for now, will be handled by user choice
                
                val_str = str(val).strip().lower()
                
                # Use synonym mappings for better standardization
                for standard_value, synonyms in self.synonym_mappings.get('status', {}).items():
                    if val_str in synonyms:
                        return standard_value
                
                # Fallback to direct mapping
                status_mapping = {
                    'delivered': 'delivered',
                    'pending': 'pending', 
                    'returned': 'returned',
                    'cancelled': 'cancelled',
                    'canceled': 'cancelled',  # Handle American spelling
                    'processing': 'pending',
                    'shipped': 'delivered',
                    'completed': 'delivered',
                    'failed': 'cancelled',
                    'refunded': 'returned'
                }
                
                return status_mapping.get(val_str, val_str)
            except Exception as e:
                print(f"Error cleaning status: {e}")
                return np.nan
        
        cleaned_series = series.apply(clean_status)
        cleaned_unique = cleaned_series.nunique()
        
        # Log the transformation
        self._log_transformation(
            column='STATUS',
            operation='status_normalization',
            original_unique=original_unique,
            cleaned_unique=cleaned_unique,
            missing_count=original_missing,
            details=f"Normalized case and merged synonyms. Reduced from {original_unique} to {cleaned_unique} unique values."
        )
        
        return cleaned_series

    def _clean_date_column(self, series):
        """Clean DATE column with comprehensive date parsing - NO DROPPING"""
        original_missing = series.isnull().sum()
        parsing_errors = 0
        successful_parses = 0
        
        def clean_date(val):
            nonlocal parsing_errors, successful_parses
            try:
                if pd.isnull(val):
                    return np.nan
                
                val_str = str(val).strip()
                
                # Handle common date formats with more comprehensive patterns
                date_formats = [
                    '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y',
                    '%Y/%m/%d', '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
                    '%d %m %Y', '%m %d %Y', '%Y %m %d',
                    '%B %d, %Y', '%d %B %Y', '%b %d, %Y', '%d %b %Y',
                    '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M',
                    '%Y%m%d', '%d%m%Y', '%m%d%Y'  # Compact formats
                ]
                
                # Try pandas date parsing first (most flexible)
                try:
                    parsed_date = pd.to_datetime(val_str, infer_datetime_format=True, errors='raise')
                    successful_parses += 1
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
                
                # Try dateutil parser for very flexible parsing
                try:
                    parsed_date = date_parser.parse(val_str, fuzzy=True)
                    successful_parses += 1
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
                
                # Try manual format matching with more aggressive parsing
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(val_str, fmt)
                        successful_parses += 1
                        return parsed_date.strftime('%Y-%m-%d')
                    except:
                        continue
                
                # Last resort: try to extract year-month-day patterns
                import re
                # Look for 4-digit year patterns
                year_match = re.search(r'(\d{4})', val_str)
                if year_match:
                    year = year_match.group(1)
                    # Look for month and day
                    month_day_match = re.search(r'(\d{1,2})[\/\-\.\s]+(\d{1,2})', val_str)
                    if month_day_match:
                        month, day = month_day_match.groups()
                        try:
                            # Try different year-month-day combinations
                            for fmt in ['%Y-%m-%d', '%Y-%d-%m']:
                                try:
                                    parsed_date = datetime.strptime(f"{year}-{month.zfill(2)}-{day.zfill(2)}", fmt)
                                    successful_parses += 1
                                    return parsed_date.strftime('%Y-%m-%d')
                                except:
                                    continue
                        except:
                            pass
                
                # If all parsing fails, try to salvage partial date info
                # Extract any recognizable date components
                date_components = re.findall(r'\d{1,4}', val_str)
                if len(date_components) >= 3:
                    # Try to construct a reasonable date
                    try:
                        # Assume YYYY-MM-DD format if we have 3+ numbers
                        year = date_components[0] if len(date_components[0]) == 4 else date_components[-1]
                        month = date_components[1] if len(date_components[0]) == 4 else date_components[0]
                        day = date_components[2] if len(date_components[0]) == 4 else date_components[1]
                        
                        # Validate and construct date
                        if len(year) == 4 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                            parsed_date = datetime(int(year), int(month), int(day))
                            successful_parses += 1
                            return parsed_date.strftime('%Y-%m-%d')
                    except:
                        pass
                
                # If we still can't parse, use forward fill or impute with median date
                parsing_errors += 1
                return None  # Will be handled by missing value strategy
                
            except Exception as e:
                parsing_errors += 1
                return None
        
        cleaned_series = series.apply(clean_date)
        
        # Apply missing value strategy for unparseable dates
        if parsing_errors > 0:
            # Strategy 1: Forward fill from last valid date
            cleaned_series = cleaned_series.fillna(method='ffill')
            
            # Strategy 2: If still NaN, use median date from successfully parsed dates
            valid_dates = pd.to_datetime(cleaned_series.dropna(), errors='coerce')
            if not valid_dates.empty:
                median_date = valid_dates.median()
                cleaned_series = cleaned_series.fillna(median_date.strftime('%Y-%m-%d'))
            else:
                # Last resort: use current date
                current_date = datetime.now().strftime('%Y-%m-%d')
                cleaned_series = cleaned_series.fillna(current_date)
        
        # Log the transformation
        self._log_transformation(
            column='DATE',
            operation='date_standardization',
            original_missing=original_missing,
            parsing_errors=parsing_errors,
            successful_parses=successful_parses,
            final_missing=cleaned_series.isnull().sum(),
            details=f"Standardized dates to YYYY-MM-DD format. {successful_parses} successful, {parsing_errors} required imputation."
        )
        
        return cleaned_series

    def _clean_age_column(self, series):
        """Clean AGE column with validation and outlier detection"""
        original_missing = series.isnull().sum()
        invalid_ages = 0
        
        def clean_age(val):
            nonlocal invalid_ages
            try:
                if pd.isnull(val):
                    return np.nan
                
                # Convert to numeric
                age = pd.to_numeric(val, errors='coerce')
                
                if pd.isnull(age):
                    invalid_ages += 1
                    return np.nan
                
                # Validate age range (0-150)
                if age < 0 or age > 150:
                    invalid_ages += 1
                    return np.nan
                
                return int(age)
                
            except Exception as e:
                invalid_ages += 1
                return np.nan
        
        cleaned_series = series.apply(clean_age)
        
        # Log the transformation
        self._log_transformation(
            column='AGE',
            operation='age_validation',
            original_missing=original_missing,
            invalid_values=invalid_ages,
            details=f"Validated age range (0-150). {invalid_ages} invalid values found."
        )
        
        return cleaned_series

    def _clean_email_column(self, series):
        """Clean EMAIL column with validation"""
        original_missing = series.isnull().sum()
        invalid_emails = 0
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        def clean_email(val):
            nonlocal invalid_emails
            try:
                if pd.isnull(val):
                    return np.nan
                
                val_str = str(val).strip().lower()
                
                if email_pattern.match(val_str):
                    return val_str
                else:
                    invalid_emails += 1
                    return np.nan
                    
            except Exception as e:
                invalid_emails += 1
                return np.nan
        
        cleaned_series = series.apply(clean_email)
        
        # Log the transformation
        self._log_transformation(
            column='EMAIL',
            operation='email_validation',
            original_missing=original_missing,
            invalid_values=invalid_emails,
            details=f"Validated email format. {invalid_emails} invalid emails found."
        )
        
        return cleaned_series

    def _clean_phone_column(self, series):
        """Clean PHONE column with standardization"""
        original_missing = series.isnull().sum()
        invalid_phones = 0
        
        def clean_phone(val):
            nonlocal invalid_phones
            try:
                if pd.isnull(val):
                    return np.nan
                
                val_str = str(val).strip()
                
                # Remove all non-digit characters
                digits_only = re.sub(r'\D', '', val_str)
                
                # Validate phone number length (7-15 digits)
                if len(digits_only) < 7 or len(digits_only) > 15:
                    invalid_phones += 1
                    return np.nan
                
                # Format as international format
                if len(digits_only) == 10:
                    return f"+1{digits_only}"
                elif len(digits_only) == 11 and digits_only.startswith('1'):
                    return f"+{digits_only}"
                else:
                    return f"+{digits_only}"
                    
            except Exception as e:
                invalid_phones += 1
                return np.nan
        
        cleaned_series = series.apply(clean_phone)
        
        # Log the transformation
        self._log_transformation(
            column='PHONE',
            operation='phone_standardization',
            original_missing=original_missing,
            invalid_values=invalid_phones,
            details=f"Standardized phone format. {invalid_phones} invalid phones found."
        )
        
        return cleaned_series

    def _clean_score_column(self, series):
        """Clean SCORE column with validation and standardization"""
        original_missing = series.isnull().sum()
        invalid_scores = 0
        
        def clean_score(val):
            nonlocal invalid_scores
            try:
                if pd.isnull(val):
                    return np.nan
                
                # Convert to string and clean
                val_str = str(val).strip().lower()
                
                # Handle percentage scores
                if '%' in val_str:
                    val_str = val_str.replace('%', '')
                    try:
                        score = float(val_str)
                        if 0 <= score <= 100:
                            return score
                        else:
                            invalid_scores += 1
                            return np.nan
                    except:
                        invalid_scores += 1
                        return np.nan
                
                # Handle fraction scores (e.g., "85/100")
                if '/' in val_str:
                    try:
                        parts = val_str.split('/')
                        if len(parts) == 2:
                            numerator = float(parts[0])
                            denominator = float(parts[1])
                            if denominator > 0:
                                score = (numerator / denominator) * 100
                                return score
                    except:
                        pass
                
                # Handle letter grades
                grade_mapping = {
                    'a+': 98, 'a': 95, 'a-': 92,
                    'b+': 88, 'b': 85, 'b-': 82,
                    'c+': 78, 'c': 75, 'c-': 72,
                    'd+': 68, 'd': 65, 'd-': 62,
                    'f': 0
                }
                if val_str in grade_mapping:
                    return grade_mapping[val_str]
                
                # Try direct numeric conversion
                try:
                    score = float(val_str)
                    # Validate score range (0-100 for most scoring systems)
                    if 0 <= score <= 100:
                        return score
                    elif 0 <= score <= 1:  # Decimal scores
                        return score * 100
                    elif 0 <= score <= 4:  # GPA scale
                        return (score / 4) * 100
                    else:
                        invalid_scores += 1
                        return np.nan
                except:
                    invalid_scores += 1
                    return np.nan
                    
            except Exception as e:
                invalid_scores += 1
                return np.nan
        
        cleaned_series = series.apply(clean_score)
        
        # Apply missing value strategy for invalid scores
        if invalid_scores > 0 or cleaned_series.isnull().any():
            # Use median of valid scores
            valid_scores = cleaned_series.dropna()
            if not valid_scores.empty:
                median_score = valid_scores.median()
                cleaned_series = cleaned_series.fillna(median_score)
            else:
                # Default to 75 (average score)
                cleaned_series = cleaned_series.fillna(75.0)
        
        # Log the transformation
        self._log_transformation(
            column='SCORE',
            operation='score_standardization',
            original_missing=original_missing,
            invalid_values=invalid_scores,
            final_missing=cleaned_series.isnull().sum(),
            details=f"Standardized scores to 0-100 scale. {invalid_scores} invalid values found and imputed."
        )
        
        return cleaned_series

    def _log_transformation(self, column, operation, **kwargs):
        """Log a transformation step with detailed information"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'column': column,
            'operation': operation,
            **kwargs
        }
        self.cleaning_log['steps'].append(log_entry)

    def _detect_domain(self, df):
        """Detect the domain of the dataset based on column names and values"""
        column_names = [col.lower() for col in df.columns]
        
        domain_scores = defaultdict(int)
        
        # Healthcare indicators
        healthcare_keywords = ['patient', 'diagnosis', 'symptom', 'medication', 'treatment', 'hospital', 'doctor', 'age', 'gender']
        for keyword in healthcare_keywords:
            if any(keyword in col for col in column_names):
                domain_scores['healthcare'] += 1
        
        # Finance indicators
        finance_keywords = ['account', 'balance', 'transaction', 'amount', 'currency', 'payment', 'credit', 'debit']
        for keyword in finance_keywords:
            if any(keyword in col for col in column_names):
                domain_scores['finance'] += 1
        
        # Retail indicators
        retail_keywords = ['product', 'price', 'category', 'order', 'customer', 'rating', 'review', 'inventory']
        for keyword in retail_keywords:
            if any(keyword in col for col in column_names):
                domain_scores['retail'] += 1
        
        return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else 'general'

    def set_parameters(self, outlier_threshold=3.0, imputation_method="mean", categorical_missing_strategy="unknown"):
        """Set cleaning parameters"""
        self.outlier_threshold = outlier_threshold
        self.imputation_method = imputation_method
        self.categorical_missing_strategy = categorical_missing_strategy
        
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
        """Complete preprocessing pipeline with comprehensive audit logging and schema validation"""
        try:
            # Initialize cleaning log
            self.cleaning_log = {
                'timestamp': datetime.now().isoformat(),
                'steps': [],
                'transformations': {},
                'statistics': {},
                'warnings': [],
                'errors': []
            }
            
            processed_df = df.copy()
            stats = {}
            
            # Step 0: Domain detection and schema validation
            detected_domain = self._detect_domain(df)
            self._log_transformation(
                column='DATASET',
                operation='domain_detection',
                detected_domain=detected_domain,
                details=f"Detected dataset domain: {detected_domain}"
            )
            
            # Schema validation
            schema_issues = self._validate_schema(df)
            if schema_issues:
                self.cleaning_log['warnings'].extend(schema_issues)
                self._log_transformation(
                    column='DATASET',
                    operation='schema_validation',
                    issues_found=len(schema_issues),
                    details=f"Found {len(schema_issues)} schema issues"
                )

            # Step 1: Header normalization
            original_columns = list(df.columns)
            processed_df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            if original_columns != list(processed_df.columns):
                self._log_transformation(
                    column='HEADERS',
                    operation='header_normalization',
                    original_columns=original_columns,
                    normalized_columns=list(processed_df.columns),
                    details="Normalized column headers to lowercase snake_case"
                )

            # Step 2: Special cleaning for known messy columns
            for col_name, cleaner_func in self.special_cleaners.items():
                if col_name in processed_df.columns:
                    original_stats = {
                        'unique_count': processed_df[col_name].nunique(),
                        'missing_count': processed_df[col_name].isnull().sum(),
                        'dtype': str(processed_df[col_name].dtype)
                    }
                    
                    processed_df[col_name] = cleaner_func(processed_df[col_name])
                    
                    new_stats = {
                        'unique_count': processed_df[col_name].nunique(),
                        'missing_count': processed_df[col_name].isnull().sum(),
                        'dtype': str(processed_df[col_name].dtype)
                    }
                    
                    self._log_transformation(
                        column=col_name,
                        operation='special_cleaning',
                        original_stats=original_stats,
                        new_stats=new_stats,
                        details=f"Applied {cleaner_func.__name__} to {col_name}"
                    )

            # Step 3: Categorical value standardization
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col not in self.special_cleaners:  # Skip already cleaned columns
                    original_unique = processed_df[col].nunique()
                    original_missing = processed_df[col].isnull().sum()
                    
                    # Apply synonym mapping if available
                    col_lower = col.lower()
                    for category_type, mappings in self.synonym_mappings.items():
                        if category_type in col_lower:
                            processed_df[col] = self._apply_synonym_mapping(processed_df[col], mappings)
                            break
                    
                    # Case normalization
                    processed_df[col] = processed_df[col].astype(str).str.strip().str.lower()
                    
                    new_unique = processed_df[col].nunique()
                    
                    self._log_transformation(
                        column=col,
                        operation='categorical_standardization',
                        original_unique=original_unique,
                        new_unique=new_unique,
                        missing_count=original_missing,
                        details=f"Standardized categorical values. Reduced from {original_unique} to {new_unique} unique values."
                    )

            # Step 4: Handle missing values in categorical columns
            for col in categorical_cols:
                if processed_df[col].isnull().any():
                    missing_count = processed_df[col].isnull().sum()
                    if self.categorical_missing_strategy == "unknown":
                        processed_df[col] = processed_df[col].fillna("unknown")
                        self._log_transformation(
                            column=col,
                            operation='missing_value_imputation',
                            missing_count=missing_count,
                            strategy='unknown',
                            details=f"Filled {missing_count} missing values with 'unknown'"
                        )
                    elif self.categorical_missing_strategy == "mode":
                        mode_value = processed_df[col].mode().iloc[0] if not processed_df[col].mode().empty else "unknown"
                        processed_df[col] = processed_df[col].fillna(mode_value)
                        self._log_transformation(
                            column=col,
                            operation='missing_value_imputation',
                            missing_count=missing_count,
                            strategy='mode',
                            mode_value=mode_value,
                            details=f"Filled {missing_count} missing values with mode: {mode_value}"
                        )
                    elif self.categorical_missing_strategy == "forward_fill":
                        processed_df[col] = processed_df[col].fillna(method='ffill').fillna(method='bfill')
                        self._log_transformation(
                            column=col,
                            operation='missing_value_imputation',
                            missing_count=missing_count,
                            strategy='forward_fill',
                            details=f"Filled {missing_count} missing values using forward/backward fill"
                        )
                    else:  # Default to unknown
                        processed_df[col] = processed_df[col].fillna("unknown")
                        self._log_transformation(
                            column=col,
                            operation='missing_value_imputation',
                            missing_count=missing_count,
                            strategy='unknown_default',
                            details=f"Filled {missing_count} missing values with 'unknown' (default strategy)"
                        )

            # Step 5: Encode categorical data
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                original_values = processed_df[col].unique()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))
                
                # Store encoding mapping with enhanced interpretability
                encoding_mapping = dict(enumerate(self.label_encoders[col].classes_))
                reverse_mapping = {v: k for k, v in encoding_mapping.items()}
                
                self.cleaning_log['transformations'][col] = {
                    'type': 'categorical_encoding',
                    'mapping': encoding_mapping,
                    'reverse_mapping': reverse_mapping,
                    'original_unique': len(original_values),
                    'encoded_unique': len(encoding_mapping),
                    'interpretation_guide': {
                        'format': 'integer_code: original_value',
                        'example': f"0: '{list(encoding_mapping.values())[0] if encoding_mapping else 'N/A'}'",
                        'usage': 'Use reverse_mapping to convert back to original values'
                    }
                }
                
                self._log_transformation(
                    column=col,
                    operation='categorical_encoding',
                    original_unique=len(original_values),
                    encoded_unique=len(encoding_mapping),
                    details=f"Encoded categorical values to integers"
                )

            # Step 6: Process numeric columns
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
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
                
                # Log outlier handling
                self._log_transformation(
                    column='NUMERIC_COLUMNS',
                    operation='outlier_handling',
                    outlier_count=stats['outlier_count'],
                    outlier_percentage=stats['outlier_percentage'],
                    strategy=outlier_strategy,
                    details=f"Handled {stats['outlier_count']} outliers using {outlier_strategy} strategy"
                )
                
                # Update dataframe
                processed_df[numeric_cols] = numeric_data_filled
            
            # Step 7: Final statistics and validation
            stats['original_shape'] = df.shape
            stats['processed_shape'] = processed_df.shape
            stats['missing_values_original'] = df.isnull().sum().sum()
            stats['missing_values_processed'] = processed_df.isnull().sum().sum()
            stats['dtypes'] = str(df.dtypes.value_counts().to_dict())
            stats['domain'] = detected_domain
            
            # Store in cleaning log
            self.cleaning_log['statistics'] = stats
            self.cleaning_log['transformations']['dataset'] = {
                'original_shape': df.shape,
                'processed_shape': processed_df.shape,
                'columns_processed': len(processed_df.columns),
                'domain': detected_domain
            }
            
            self.feature_stats = stats
            return processed_df, stats
        except Exception as e:
            error_msg = f"Error in preprocess_data: {e}"
            self.cleaning_log['errors'].append(error_msg)
            print(error_msg)
            return df, {}

    def _validate_schema(self, df):
        """Validate dataset schema and return issues"""
        issues = []
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns found: {empty_cols}")
        
        # Check for columns with only one unique value
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                single_value_cols.append(col)
        if single_value_cols:
            issues.append(f"Columns with single value: {single_value_cols}")
        
        # Check for potential ID columns that should be strings
        potential_id_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'key', 'code', 'number']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    potential_id_cols.append(col)
        if potential_id_cols:
            issues.append(f"Potential ID columns with numeric type: {potential_id_cols}")
        
        return issues

    def _apply_synonym_mapping(self, series, mappings):
        """Apply synonym mapping to a series"""
        def map_value(val):
            if pd.isnull(val):
                return val
            val_str = str(val).strip().lower()
            for standard_value, synonyms in mappings.items():
                if val_str in synonyms:
                    return standard_value
            return val_str
        
        return series.apply(map_value)

    def get_cleaning_log(self):
        """Return the comprehensive cleaning log"""
        return self.cleaning_log

    def export_cleaning_log(self, format='json'):
        """Export cleaning log in specified format"""
        if format == 'json':
            return json.dumps(self.cleaning_log, indent=2, default=str)
        elif format == 'summary':
            summary = {
                'timestamp': self.cleaning_log['timestamp'],
                'total_steps': len(self.cleaning_log['steps']),
                'warnings': len(self.cleaning_log['warnings']),
                'errors': len(self.cleaning_log['errors']),
                'transformations': len(self.cleaning_log['transformations']),
                'statistics': self.cleaning_log['statistics']
            }
            return json.dumps(summary, indent=2, default=str)
        else:
            return str(self.cleaning_log)

    def export_interpretable_mappings(self, format='markdown'):
        """Export categorical mappings in human-readable format"""
        mappings = {}
        for col, info in self.cleaning_log.get('transformations', {}).items():
            if info.get('type') == 'categorical_encoding':
                mappings[col] = {
                    'encoding': info['mapping'],
                    'reverse': info['reverse_mapping'],
                    'counts': info.get('original_unique', 0)
                }
        
        if format == 'markdown':
            md_content = "# Categorical Encoding Mappings\n\n"
            md_content += "This file contains the mapping between original categorical values and their encoded integer representations.\n\n"
            
            for col, mapping_info in mappings.items():
                md_content += f"## Column: {col}\n\n"
                md_content += f"**Total unique values:** {mapping_info['counts']}\n\n"
                md_content += "### Encoding (Integer → Original Value)\n\n"
                md_content += "| Code | Original Value |\n"
                md_content += "|------|----------------|\n"
                
                for code, value in mapping_info['encoding'].items():
                    md_content += f"| {code} | {value} |\n"
                
                md_content += "\n### Reverse Mapping (Original Value → Integer)\n\n"
                md_content += "| Original Value | Code |\n"
                md_content += "|----------------|------|\n"
                
                for value, code in mapping_info['reverse'].items():
                    md_content += f"| {value} | {code} |\n"
                
                md_content += "\n---\n\n"
            
            return md_content
        
        elif format == 'csv':
            csv_rows = []
            for col, mapping_info in mappings.items():
                for code, value in mapping_info['encoding'].items():
                    csv_rows.append({
                        'column': col,
                        'code': code,
                        'original_value': value
                    })
            
            if csv_rows:
                df = pd.DataFrame(csv_rows)
                return df.to_csv(index=False)
            else:
                return "column,code,original_value\n"
        
        else:  # json
            return json.dumps(mappings, indent=2, default=str)
    
    def preprocess_dual_output(self, df: pd.DataFrame, 
                              outlier_strategy: str = "cap") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Complete preprocessing pipeline returning both raw-but-tidy and ML-ready versions"""
        try:
            raw_tidy_df = df.copy()
            stats = {}

            # Step 1: Clean messy columns for raw-but-tidy version (but keep original dtypes where possible)
            for col_name, cleaner_func in self.special_cleaners.items():
                if col_name in raw_tidy_df.columns:
                    raw_tidy_df[col_name] = cleaner_func(raw_tidy_df[col_name])

            # Step 2: Create ML-ready version
            ml_ready_df = raw_tidy_df.copy()
            
            # Separate numeric and categorical columns
            numeric_cols = ml_ready_df.select_dtypes(include=[np.number]).columns
            categorical_cols = ml_ready_df.select_dtypes(include=['object', 'category']).columns

            # Step 3: Handle categorical data (only for ML-ready version)
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                ml_ready_df[col] = self.label_encoders[col].fit_transform(ml_ready_df[col].astype(str))

            # Step 4: Process numeric columns (for ML-ready version)
            if len(numeric_cols) > 0:
                numeric_data = jnp.array(ml_ready_df[numeric_cols].values, dtype=jnp.float32)
                
                # Handle outliers
                numeric_data = self.handle_outliers(numeric_data, outlier_strategy)
                
                # Fill missing values
                numeric_data_filled = self.fill_missing_values_jax(numeric_data)
                
                # Detect outliers after handling
                outlier_mask = self.detect_outliers_jax(numeric_data_filled)
                stats['outlier_count'] = int(jnp.sum(outlier_mask))
                stats['outlier_percentage'] = float(jnp.mean(outlier_mask) * 100)
                
                # Update ML-ready dataframe
                ml_ready_df[numeric_cols] = numeric_data_filled
                
                # Also fill missing values in raw-but-tidy (but don't scale or encode categoricals)
                raw_numeric_data = jnp.array(raw_tidy_df[numeric_cols].values, dtype=jnp.float32)
                raw_numeric_data = self.handle_outliers(raw_numeric_data, outlier_strategy)
                raw_numeric_filled = self.fill_missing_values_jax(raw_numeric_data)
                raw_tidy_df[numeric_cols] = raw_numeric_filled
            
            # Store feature statistics
            stats['original_shape'] = df.shape
            stats['raw_tidy_shape'] = raw_tidy_df.shape
            stats['ml_ready_shape'] = ml_ready_df.shape
            stats['missing_values_original'] = df.isnull().sum().sum()
            stats['missing_values_raw_tidy'] = raw_tidy_df.isnull().sum().sum()
            stats['missing_values_ml_ready'] = ml_ready_df.isnull().sum().sum()
            stats['dtypes'] = str(df.dtypes.value_counts().to_dict())
            
            self.feature_stats = stats
            return raw_tidy_df, ml_ready_df, stats
        except Exception as e:
            print(f"Error in preprocess_dual_output: {e}")
            return df, df, {}

# ======================== ENHANCED VISUALIZATIONS & METRICS ========================
def calculate_advanced_metrics(y_true, y_pred, y_prob=None, problem_type="binary"):
    """Calculate modern ML metrics for both classification and regression"""
    metrics = {}
    
    if problem_type == "regression":
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Additional regression insights
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
    else:
        # Classification metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        if problem_type == "binary":
            # Binary classification metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['precision'] = tp / (tp + fp + 1e-8)
            metrics['recall'] = tp / (tp + fn + 1e-8)
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-8)
            
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                except:
                    metrics['roc_auc'] = np.nan
                try:
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                except:
                    metrics['pr_auc'] = np.nan
                try:
                    metrics['log_loss'] = -np.mean(y_true * np.log(y_prob + 1e-8) + (1 - y_true) * np.log(1 - y_prob + 1e-8))
                except:
                    metrics['log_loss'] = np.nan
        else:
            # Multi-class metrics
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except:
                    metrics['roc_auc'] = np.nan
        
        # Class imbalance analysis
        unique, counts = np.unique(y_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique, counts))
        metrics['imbalance_ratio'] = max(counts) / min(counts) if len(counts) > 1 else 1.0
    
    return metrics

def create_data_exploration_plots(df_clean, target_col, target_type):
    """Create comprehensive data exploration plots after cleaning"""
    st.subheader("📊 Post-Cleaning Data Exploration")
    
    # Separate numerical and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from features for plotting
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Plot distributions of numerical features
    if numeric_cols:
        st.subheader("📈 Numerical Features Distribution")
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        for i in range(0, len(numeric_cols), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                if i + j < len(numeric_cols):
                    feature = numeric_cols[i + j]
                    with col:
                        fig = px.histogram(df_clean, x=feature, title=f"{feature} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    # Plot bar charts of categorical features
    if categorical_cols:
        st.subheader("📊 Categorical Features Distribution")
        for feature in categorical_cols[:6]:  # Limit to 6 to avoid overwhelming
            value_counts = df_clean[feature].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"{feature} Distribution (Top 10)",
                        labels={'x': feature, 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix for numerical features
    if len(numeric_cols) > 1:
        st.subheader("🔗 Feature Correlation Matrix")
        corr_matrix = df_clean[numeric_cols + ([target_col] if target_col in df_clean.columns else [])].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Correlation Matrix of Numerical Features",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top correlations with target
        if target_col in df_clean.columns and target_type == "regression":
            target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
            top_corr = target_corr.head(10)
            
            fig_corr = px.bar(x=top_corr.values, y=top_corr.index,
                             orientation='h',
                             title=f"Top 10 Features Correlated with {target_col}",
                             labels={'x': 'Absolute Correlation', 'y': 'Feature'})
            fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_corr, use_container_width=True)

def create_regression_plots(y_true, y_pred, target_name="Target"):
    """Create regression-specific evaluation plots"""
    # Actual vs Predicted scatter plot
    fig_scatter = px.scatter(x=y_true, y=y_pred, 
                           title=f"Actual vs Predicted {target_name}",
                           labels={'x': f'Actual {target_name}', 'y': f'Predicted {target_name}'})
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red')))
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Residuals plot
    residuals = y_true - y_pred
    fig_residuals = px.scatter(x=y_pred, y=residuals,
                              title="Residuals vs Predicted Values",
                              labels={'x': 'Predicted Values', 'y': 'Residuals'})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Residuals histogram
    fig_hist = px.histogram(residuals, title="Distribution of Residuals",
                           labels={'x': 'Residuals', 'y': 'Frequency'})
    st.plotly_chart(fig_hist, use_container_width=True)

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
try:
    from transformers import pipeline
    # Use smaller, faster models for deployment
    TRANSFORMERS_AVAILABLE = True
    DEFAULT_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Can be changed to smaller model
    DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Smaller than default
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DEFAULT_SUMMARIZATION_MODEL = None
    DEFAULT_SENTIMENT_MODEL = None

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
        return rp.can_fetch("DataAnalyzerBot/1.0", url)
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
    show_raw_html = st.sidebar.checkbox("Show Raw HTML Viewer", value=False)

    url_input = st.text_area("Enter one or more URLs (one per line):")
    
    if st.button("🚀 Fetch & Analyze URLs"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        
        if not urls:
            st.warning("Please enter at least one URL to analyze.")
            return
            
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create containers for better organization
        results_container = st.container()
        error_container = st.container()
        
        total_urls = len(urls)
        processed = 0
        errors = []
        
        # Check robots.txt first
        status_text.text("🔍 Checking robots.txt permissions...")
        allowed_urls = []
        blocked_urls = []
        
        for url in urls:
            if not is_allowed_by_robots(url):
                blocked_urls.append(url)
                with error_container:
                    st.warning(f"🚫 Blocked by robots.txt: {url}")
            else:
                allowed_urls.append(url)
        
        if not allowed_urls:
            st.error("❌ All URLs are blocked by robots.txt. Please try different URLs.")
            return
            
        status_text.text(f"✅ {len(allowed_urls)} URLs allowed, {len(blocked_urls)} blocked")
        
        # Process URLs in batches
        async def fetch_all(urls):
            async with httpx.AsyncClient(timeout=timeout) as client:
                tasks = []
                for url in urls:
                    tasks.append(client.get(url, headers={"User-Agent": "DataAnalyzerBot/1.0"}))
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                return responses
        
        for i in range(0, len(allowed_urls), batch_size):
            batch = allowed_urls[i:i+batch_size]
            status_text.text(f"📡 Fetching batch {i//batch_size + 1}/{(len(allowed_urls) + batch_size - 1)//batch_size}...")
            
            try:
                responses = asyncio.get_event_loop().run_until_complete(fetch_all(batch))
            except Exception as e:
                with error_container:
                    st.error(f"❌ Async fetch error: {e}")
                continue
                
            for url, resp in zip(batch, responses):
                processed += 1
                progress_bar.progress(processed / total_urls)
                status_text.text(f"🔍 Processing {url}...")
                
                if isinstance(resp, Exception):
                    errors.append(f"Failed to fetch {url}: {resp}")
                    with error_container:
                        st.error(f"❌ Failed to fetch {url}: {resp}")
                    continue
                    
                try:
                    html = resp.text if hasattr(resp, 'text') else str(resp)
                    info = parse_and_extract_info_adv(html, url, nlp_model, entity_types, sentiment_enabled)
                    results.append(info)
                    
                    with results_container:
                        st.subheader(f"📄 Results for: {url}")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**📝 Summary:** {info['summary']}")
                            st.write(f"**🏷️ Entities:** {info['entities']}")
                            st.write(f"**😊 Sentiment:** {info.get('sentiment', 'N/A')}")
                            st.write(f"**ℹ️ About Section:** {info['about']}")
                            st.write(f"**📞 Contact Section:** {info['contact']}")
                            st.write(f"**📖 Main Text (first 500 chars):** {info['main_text'][:500]}")
                            
                            if info['tables']:
                                st.write("**📊 Tables Found:**")
                                for i, tbl in enumerate(info['tables']):
                                    with st.expander(f"Table {i+1}"):
                                        st.dataframe(make_arrow_compatible(tbl.head()))
                        
                        with col2:
                            if show_raw_html:
                                with st.expander("🔍 Raw HTML"):
                                    st.code(html[:2000] + "..." if len(html) > 2000 else html, language='html')
                            
                            if info.get('main_text'):
                                with st.expander("📝 Parsed Text"):
                                    st.text(info['main_text'][:1000] + "..." if len(info['main_text']) > 1000 else info['main_text'])
                        
                        st.divider()
                        
                except Exception as e:
                    errors.append(f"Error processing {url}: {str(e)}")
                    with error_container:
                        st.error(f"❌ Error processing {url}: {str(e)}")
        
        # Final status
        progress_bar.progress(1.0)
        status_text.text(f"✅ Processing complete! {len(results)} successful, {len(errors)} errors")
        
        # Download structured results
        if results:
            st.subheader("📥 Download Results")
            
            def format_entities(entities):
                return "; ".join([f"{label}:{text}" for text, label in entities])
            
            def clean_text(text, maxlen=300):
                if not text:
                    return ""
                text = str(text).replace("\n", " ").replace("\r", " ").strip()
                return text[:maxlen] + ("..." if len(text) > maxlen else "")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
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
                st.download_button("📊 Download Extracted Info as CSV", csv, "extracted_info.csv", "text/csv")

            with col_download2:
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
                st.download_button("🤖 Download ML-Ready Data as CSV", csv_ml, "ml_ready_data.csv", "text/csv")
        
        # Show error summary if any
        if errors:
            with st.expander("❌ Error Summary"):
                for error in errors:
                    st.error(error)

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
            # Use optimized model for deployment
            model_name = DEFAULT_SUMMARIZATION_MODEL if DEFAULT_SUMMARIZATION_MODEL else "facebook/bart-large-cnn"
            summarizer = pipeline("summarization", model=model_name)
            summary = summarizer(main_text_clean[:1024])[0]['summary_text']
        except Exception as e:
            summary = main_text_clean[:300]
        if sentiment_enabled:
            try:
                # Use optimized sentiment model for deployment
                model_name = DEFAULT_SENTIMENT_MODEL if DEFAULT_SENTIMENT_MODEL else "cardiffnlp/twitter-roberta-base-sentiment-latest"
                sentiment_pipe = pipeline("sentiment-analysis", model=model_name)
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
    
    # Deployment optimization info
    with st.expander("🚀 Deployment Information", expanded=False):
        st.markdown("""
        **Performance Optimizations:**
        - ✅ Streamlit caching enabled for data loading and model training
        - ✅ JAX-optimized data cleaning pipeline
        - ✅ Smaller transformer models for web scraping
        - ✅ Efficient batch processing for web scraping
        - ✅ Resource-aware model selection
        
        **System Requirements:**
        - Python 3.8+
        - Memory: 2GB+ RAM recommended
        - Storage: 1GB+ for models and dependencies
        
        **Dependencies:** See requirements.txt for full list
        """)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Data Cleaning & Analysis", "Web Scraping & Company Info"])
    if page == "Web Scraping & Company Info":
        web_scraping_ui()
        return

    # --- Automated Data Cleaning Parameters ---
    st.sidebar.subheader("Data Cleaning Parameters")
    outlier_threshold = st.sidebar.slider("Outlier Detection Threshold (Z-score)", 2.0, 5.0, 3.0, 0.1)
    imputation_method = st.sidebar.selectbox("Missing Value Imputation", ["mean", "median", "zero"])
    outlier_strategy = st.sidebar.selectbox("Outlier Handling Strategy", ["cap", "remove", "ignore"])
    categorical_missing_strategy = st.sidebar.selectbox(
        "Categorical Missing Values", 
        ["unknown", "mode", "forward_fill"], 
        help="'unknown': Fill with 'unknown' category, 'mode': Fill with most frequent value, 'forward_fill': Use forward/backward fill"
    )

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = load_data_from_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return
        try:
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            st.subheader("📊 Raw Data Preview")
            st.dataframe(make_arrow_compatible(df.head()), use_container_width=True)

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

            # Target selection and type detection
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 0:
                target_col = cat_cols[-1]
            else:
                target_col = df.columns[-1]
            
            # Detect target type automatically
            target_type = detect_target_type(df[target_col])
            st.info(f"🎯 Automatically selected target column: **{target_col}** ({target_type}). All other columns are treated as features.")
            
            # Analyze data quality issues
            st.subheader("🔍 Data Quality Analysis")
            quality_analysis = analyze_data_quality(df, target_col)
            
            if quality_analysis['total_issues'] > 0:
                st.warning(f"⚠️ Found {quality_analysis['total_issues']} data quality issues")
                
                # Display specific issues
                if 'missing_values' in quality_analysis['issues']:
                    st.subheader("📊 Missing Values")
                    missing_df = pd.DataFrame([
                        {'Column': col, 'Missing Count': count, 'Percentage': f"{(count/len(df)*100):.1f}%"}
                        for col, count in quality_analysis['issues']['missing_values'].items()
                    ])
                    st.dataframe(make_arrow_compatible(missing_df), use_container_width=True)
                
                if 'case_inconsistencies' in quality_analysis['issues']:
                    st.subheader("🔤 Case Inconsistencies")
                    for col, info in quality_analysis['issues']['case_inconsistencies'].items():
                        with st.expander(f"Column: {col} (Original: {info['original_unique']}, Normalized: {info['normalized_unique']})"):
                            st.write("Examples of inconsistent values:")
                            st.write(info['examples'])
                
                if 'duplicate_rows' in quality_analysis['issues']:
                    st.subheader("🔄 Duplicate Rows")
                    st.write(f"Found {quality_analysis['issues']['duplicate_rows']} duplicate rows")
                
                # Show recommendations
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(quality_analysis['recommendations'], 1):
                    st.write(f"{i}. {rec}")
                
                # Show cleaning demos
                col_demo1, col_demo2, col_demo3 = st.columns(3)
                
                with col_demo1:
                    if st.button("🎯 Show STATUS Column Cleaning Demo"):
                        demonstrate_status_cleaning()
                
                with col_demo2:
                    if st.button("🔧 Show Comprehensive Cleaning Demo"):
                        demonstrate_comprehensive_cleaning()
                
                with col_demo3:
                    if st.button("📋 View Cleaning Capabilities"):
                        st.info("""
                        **Enhanced Cleaning Features:**
                        - ✅ Header normalization (snake_case)
                        - ✅ Date parsing (multiple formats → YYYY-MM-DD)
                        - ✅ Email validation & standardization
                        - ✅ Phone number formatting
                        - ✅ Age validation (0-150 range)
                        - ✅ Categorical synonym merging
                        - ✅ Domain detection (healthcare, finance, retail)
                        - ✅ Comprehensive audit logging
                        - ✅ Schema validation
                        - ✅ Outlier detection & handling
                        """)
                
                # Show STATUS cleaning demo if relevant
                if 'case_inconsistencies' in quality_analysis['issues'] and any('status' in col.lower() for col in quality_analysis['issues']['case_inconsistencies'].keys()):
                    if st.button("🔧 Clean STATUS Column Now"):
                        # Apply STATUS cleaning immediately
                        cleaner = JAXDataCleaner()
                        status_cols = [col for col in df.columns if 'status' in col.lower()]
                        if status_cols:
                            for col in status_cols:
                                st.write(f"**Cleaning {col} column...**")
                                cleaned_series = cleaner._clean_status_column(df[col])
                                
                                st.write("**Before cleaning:**")
                                st.write(df[col].value_counts(dropna=False))
                                
                                st.write("**After cleaning:**")
                                st.write(cleaned_series.value_counts(dropna=False))
                                
                                st.success(f"✅ {col} column cleaned! Case inconsistencies resolved.")
            else:
                st.success("✅ No major data quality issues detected!")
            
            st.markdown("---")

            with st.spinner("⏳ Running automated data cleaning pipeline..."):
                try:
                    cleaner = get_cleaner_instance()
                    cleaner.set_parameters(outlier_threshold, imputation_method, categorical_missing_strategy)
                    X_raw = df.drop(columns=[target_col])
                    y_raw = df[target_col]
                    X_clean, stats = cleaner.preprocess_data(X_raw, outlier_strategy)
                    
                    # Handle target encoding based on type
                    if target_type == "classification":
                        le_target = LabelEncoder()
                        y_clean_np = le_target.fit_transform(y_raw)
                    else:  # regression
                        y_clean_np = y_raw.values.astype(float)
                    
                    X_clean_np = cleaner.scaler.fit_transform(X_clean)
                except Exception as e:
                    st.error(f"Error in data cleaning: {e}")
                    return
            
            st.subheader("🧹 Data Cleaning Results")
            col1, col2, col3, col4 = st.columns(4)
            outlier_count = stats.get('outlier_count', 0)
            outlier_percentage = stats.get('outlier_percentage', 0.0)
            col1.metric("Outliers Handled", outlier_count)
            col2.metric("Missing Values Filled", stats.get('missing_values_original', 0))
            col3.metric("Features Scaled", X_clean_np.shape[1])
            
            # Show categorical cleaning results
            categorical_cols_cleaned = 0
            for col in X_raw.select_dtypes(include=['object', 'category']).columns:
                if col in cleaner.label_encoders:
                    categorical_cols_cleaned += 1
            
            col4.metric("Categorical Columns Encoded", categorical_cols_cleaned)
            st.success("✅ Data cleaning complete!")
            
            # Show specific cleaning results for categorical columns
            if categorical_cols_cleaned > 0:
                with st.expander("📋 Categorical Column Cleaning Details"):
                    for col in X_raw.select_dtypes(include=['object', 'category']).columns:
                        if col in cleaner.label_encoders:
                            original_unique = X_raw[col].nunique()
                            cleaned_unique = len(cleaner.label_encoders[col].classes_)
                            st.write(f"**{col}**: {original_unique} → {cleaned_unique} unique values")
                            
                            # Show encoding mapping for small number of categories
                            if cleaned_unique <= 10:
                                mapping = dict(enumerate(cleaner.label_encoders[col].classes_))
                                st.write(f"Encoding: {mapping}")
                    
                    # Special section for STATUS column if it exists
                    status_cols = [col for col in X_raw.columns if 'status' in col.lower()]
                    if status_cols:
                        st.subheader("🎯 STATUS Column Cleaning Results")
                        for col in status_cols:
                            st.write(f"**Original {col} values:**")
                            original_counts = X_raw[col].value_counts(dropna=False)
                            st.write(original_counts)
                            
                            st.write(f"**After cleaning {col} values:**")
                            if col in X_clean.columns:
                                cleaned_counts = X_clean[col].value_counts(dropna=False)
                                st.write(cleaned_counts)
                            else:
                                st.write("Column was encoded for ML processing")
            
            with st.expander("Cleaned Data Preview"):
                try:
                    cleaned_preview = pd.DataFrame(X_clean_np, columns=X_clean.columns).head()
                    st.dataframe(make_arrow_compatible(cleaned_preview), use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying cleaned data: {e}")

            # Add data exploration section
            create_data_exploration_plots(X_clean, target_col, target_type)

            # Enhanced Download Options with Cleaning Log
            st.subheader("💾 Download Options")
            
            col_download1, col_download2, col_download3 = st.columns(3)
            
            try:
                # 1. Raw-but-tidy version (semantic richness preserved)
                raw_tidy_df = X_clean.copy()  # This has original column names but cleaned values
                raw_tidy_df[target_col] = y_raw  # Keep original target values (not encoded)
                
                # Add back original categorical mappings as comments in a separate info column
                mapping_info = []
                for col, encoder in cleaner.label_encoders.items():
                    if col in X_clean.columns:
                        mapping_info.append(f"{col}: {dict(enumerate(encoder.classes_))}")
                
                raw_tidy_csv = raw_tidy_df.to_csv(index=False).encode('utf-8')
                
                with col_download1:
                    st.markdown("**🏷️ Raw-but-Tidy Version**")
                    st.markdown("✅ Semantic richness preserved  \n✅ Original categories/strings  \n✅ Good for feature engineering")
                    st.download_button(
                        label="📥 Download Raw-but-Tidy Dataset",
                        data=raw_tidy_csv,
                        file_name="raw_but_tidy_dataset.csv",
                        mime="text/csv"
                    )
                    
                    if mapping_info:
                        with st.expander("View Categorical Mappings"):
                            for info in mapping_info:
                                st.text(info)
                
                # 2. ML-ready version (fully numeric)
                ml_ready_df = pd.DataFrame(X_clean_np, columns=X_clean.columns)
                ml_ready_df[target_col] = y_clean_np
                ml_ready_csv = ml_ready_df.to_csv(index=False).encode('utf-8')
                
                with col_download2:
                    st.markdown("**🤖 ML-Ready Version**")
                    st.markdown("✅ All numeric columns  \n✅ Scaled features  \n✅ Instant training ready")
                    st.download_button(
                        label="📥 Download ML-Ready Dataset",
                        data=ml_ready_csv,
                        file_name="ml_ready_dataset.csv",
                        mime="text/csv"
                    )
                    
                    # Show encoding mappings for target
                    with st.expander("View Target Encoding"):
                        target_mapping = dict(enumerate(le_target.classes_))
                        st.text(f"Target '{target_col}' encoding: {target_mapping}")
                
                # 3. Comprehensive Cleaning Log & Mappings
                with col_download3:
                    st.markdown("**📋 Cleaning Log & Mappings**")
                    st.markdown("✅ Complete audit trail  \n✅ All transformations  \n✅ Human-readable mappings")
                    
                    # Get comprehensive cleaning log
                    cleaning_log = cleaner.get_cleaning_log()
                    cleaning_log_json = cleaner.export_cleaning_log('json')
                    cleaning_log_summary = cleaner.export_cleaning_log('summary')
                    
                    # Get interpretable mappings
                    mappings_markdown = cleaner.export_interpretable_mappings('markdown')
                    mappings_csv = cleaner.export_interpretable_mappings('csv')
                    mappings_json = cleaner.export_interpretable_mappings('json')
                    
                    st.download_button(
                        label="📥 Download Full Cleaning Log (JSON)",
                        data=cleaning_log_json,
                        file_name="cleaning_log_full.json",
                        mime="application/json"
                    )
                    
                    st.download_button(
                        label="📥 Download Cleaning Summary (JSON)",
                        data=cleaning_log_summary,
                        file_name="cleaning_log_summary.json",
                        mime="application/json"
                    )
                    
                    st.download_button(
                        label="📥 Download Mappings (Markdown)",
                        data=mappings_markdown,
                        file_name="categorical_mappings.md",
                        mime="text/markdown"
                    )
                    
                    st.download_button(
                        label="📥 Download Mappings (CSV)",
                        data=mappings_csv,
                        file_name="categorical_mappings.csv",
                        mime="text/csv"
                    )
                    
                    # Display cleaning log summary in UI
                    with st.expander("View Cleaning Log Summary"):
                        st.json(cleaning_log_summary)
                        
            except Exception as e:
                st.error(f"Error preparing downloads: {e}")
            
            # Display comprehensive cleaning log in UI
            st.subheader("📋 Comprehensive Cleaning Log")
            
            with st.expander("View Complete Cleaning Log", expanded=False):
                if hasattr(cleaner, 'cleaning_log') and cleaner.cleaning_log:
                    # Display key statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Total Steps", len(cleaner.cleaning_log.get('steps', [])))
                    with col_stat2:
                        st.metric("Warnings", len(cleaner.cleaning_log.get('warnings', [])))
                    with col_stat3:
                        st.metric("Errors", len(cleaner.cleaning_log.get('errors', [])))
                    with col_stat4:
                        st.metric("Transformations", len(cleaner.cleaning_log.get('transformations', {})))
                    
                    # Display warnings and errors
                    if cleaner.cleaning_log.get('warnings'):
                        st.subheader("⚠️ Warnings")
                        for warning in cleaner.cleaning_log['warnings']:
                            st.warning(warning)
                    
                    if cleaner.cleaning_log.get('errors'):
                        st.subheader("❌ Errors")
                        for error in cleaner.cleaning_log['errors']:
                            st.error(error)
                    
                    # Display transformation steps
                    st.subheader("🔄 Transformation Steps")
                    for i, step in enumerate(cleaner.cleaning_log.get('steps', []), 1):
                        with st.expander(f"Step {i}: {step.get('operation', 'Unknown')} - {step.get('column', 'Unknown')}"):
                            st.json(step)
                    
                    # Display categorical mappings
                    if cleaner.cleaning_log.get('transformations'):
                        st.subheader("🏷️ Categorical Mappings")
                        for col, mapping_info in cleaner.cleaning_log['transformations'].items():
                            if mapping_info.get('type') == 'categorical_encoding':
                                with st.expander(f"Column: {col}"):
                                    st.write("**Encoding Mapping:**")
                                    st.json(mapping_info['mapping'])
                                    st.write(f"**Original Unique Values:** {mapping_info.get('original_unique', 'N/A')}")
                else:
                    st.info("No detailed cleaning log available. This may be due to using the legacy cleaning method.")
            
            st.markdown("---")

            ### --- START: ENHANCED MODELING SECTION --- ###
            
            st.subheader("🤖 Enhanced Model Training & Evaluation")
            
            # Model Selection UI
            model_configs = get_model_configs()
            available_models = list(model_configs[target_type].keys())
            
            col_model1, col_model2 = st.columns([2, 1])
            
            with col_model1:
                selected_model_name = st.selectbox(
                    f"Select {target_type.title()} Model",
                    available_models,
                    help=f"Choose from {len(available_models)} available {target_type} models"
                )
            
            with col_model2:
                enable_hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            
            # Hyperparameter tuning options
            if enable_hyperparameter_tuning:
                tuning_method = st.radio(
                    "Hyperparameter Tuning Method",
                    ["GridSearchCV", "RandomizedSearchCV"],
                    horizontal=True
                )
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                n_iter = st.slider("RandomizedSearch Iterations", 10, 100, 50) if tuning_method == "RandomizedSearchCV" else None
            
            # Model Training Controls
            st.sidebar.subheader("Training Parameters")
            test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.sidebar.number_input("Random State", value=42)
            
            # Manual hyperparameter controls
            st.sidebar.subheader("Manual Hyperparameters")
            model_config = model_configs[target_type][selected_model_name]
            
            manual_params = {}
            if model_config["params"] and not enable_hyperparameter_tuning:
                for param, values in model_config["params"].items():
                    if isinstance(values[0], (int, float)):
                        if len(values) == 1:
                            manual_params[param] = st.sidebar.number_input(param, value=values[0])
                        else:
                            manual_params[param] = st.sidebar.slider(param, min(values), max(values), values[len(values)//2])
                    else:
                        manual_params[param] = st.sidebar.selectbox(param, values)

            if st.button(f"🚀 Train {selected_model_name}"):
                with st.spinner(f"Training {selected_model_name} model..."):
                    try:
                        # 1. Split Data
                        stratify = y_clean_np if target_type == "classification" else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_clean_np, y_clean_np, test_size=test_size, random_state=random_state, stratify=stratify
                        )
                        
                        st.write(f"📊 Data Split: Training set has {X_train.shape[0]} samples, Test set has {X_test.shape[0]} samples.")

                        # 2. Initialize Model
                        model_class = model_config["model"]
                        
                        if enable_hyperparameter_tuning:
                            # Hyperparameter tuning
                            st.info(f"🔍 Running {tuning_method} with {cv_folds} folds...")
                            
                            search_params = model_config["params"]
                            base_model = model_class(random_state=random_state)
                            
                            if tuning_method == "GridSearchCV":
                                search = GridSearchCV(base_model, search_params, cv=cv_folds, scoring='accuracy' if target_type == "classification" else 'r2')
                            else:
                                search = RandomizedSearchCV(base_model, search_params, cv=cv_folds, n_iter=n_iter, 
                                                           scoring='accuracy' if target_type == "classification" else 'r2')
                            
                            search.fit(X_train, y_train)
                            model = search.best_estimator_
                            
                            st.success(f"✅ Best parameters: {search.best_params_}")
                            st.info(f"Best cross-validation score: {search.best_score_:.4f}")
                            
                        else:
                            # Manual parameters
                            final_params = {**manual_params, 'random_state': random_state}
                            model = model_class(**final_params)
                            model.fit(X_train, y_train)

                        # 3. Get Predictions
                        y_pred = model.predict(X_test)
                        
                        # Handle probabilities for classification
                        y_prob_for_auc = None
                        if target_type == "classification":
                            try:
                                y_prob_full = model.predict_proba(X_test)
                                n_classes = len(np.unique(y_clean_np))
                                is_binary = n_classes == 2
                                y_prob_for_auc = y_prob_full[:, 1] if is_binary else y_prob_full
                            except:
                                pass

                    except Exception as e:
                        st.error(f"❌ Error during model training: {e}")
                        return

                st.success("✅ Model training complete!")
                st.markdown("---")

                # 4. Display Performance Metrics
                st.subheader("📈 Model Performance Metrics")
                
                metrics = calculate_advanced_metrics(y_test, y_pred, y_prob_for_auc, problem_type=target_type)
                
                if target_type == "classification":
                    # Classification metrics
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                    m_col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    m_col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                    m_col4.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
                    
                    if not np.isnan(metrics.get('roc_auc', np.nan)):
                        auc_col1, auc_col2 = st.columns(2)
                        auc_col1.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
                        if 'pr_auc' in metrics:
                           auc_col2.metric("Precision-Recall AUC", f"{metrics.get('pr_auc', 0):.3f}")
                else:
                    # Regression metrics
                    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                    r_col1.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                    r_col2.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
                    r_col3.metric("MAE", f"{metrics.get('mae', 0):.3f}")
                    r_col4.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")

                # Detailed reports
                if target_type == "classification":
                    with st.expander("View Detailed Classification Report"):
                        try:
                            unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
                            if 'le_target' in locals():
                                target_names_filtered = [str(le_target.classes_[i]) for i in unique_test_classes if i < len(le_target.classes_)]
                                report = classification_report(y_test, y_pred, target_names=target_names_filtered, labels=unique_test_classes)
                            else:
                                report = classification_report(y_test, y_pred)
                            st.text(report)
                        except Exception as e:
                            st.error(f"Could not generate classification report: {e}")
                else:
                    with st.expander("View Residual Statistics"):
                        st.write(f"Mean Residual: {metrics.get('mean_residual', 0):.4f}")
                        st.write(f"Std Residual: {metrics.get('std_residual', 0):.4f}")

                # 5. Create and Display Evaluation Plots
                st.subheader("📊 Model Evaluation Plots")
                
                if target_type == "classification" and y_prob_for_auc is not None:
                    n_classes = len(np.unique(y_clean_np))
                    is_binary = n_classes == 2
                    create_evaluation_plots(y_test, y_pred, y_prob_for_auc, target_name=target_col, is_binary=is_binary)
                elif target_type == "regression":
                    create_regression_plots(y_test, y_pred, target_name=target_col)
                
                st.markdown("---")
                
                # 6. Feature Importance
                st.subheader("🌟 Feature Importance Analysis")
                try:
                    if hasattr(model, 'coef_'):
                        # Linear models
                        if model.coef_.shape[0] == 1:  # Binary classification or regression
                            importances = np.abs(model.coef_[0])
                        else:  # Multi-class
                            importances = np.mean(np.abs(model.coef_), axis=0)
                        
                        feature_importance_df = pd.DataFrame({
                            'Feature': X_clean.columns,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)
                        
                        plot_feature_importance(feature_importance_df)
                        
                    elif hasattr(model, 'feature_importances_'):
                        # Tree-based models
                        feature_importance_df = pd.DataFrame({
                            'Feature': X_clean.columns,
                            'Importance': model.feature_importances_
                        }).sort_values(by='Importance', ascending=False)
                        
                        fig = px.bar(feature_importance_df.head(20),
                                   x='Importance', y='Feature',
                                   orientation='h',
                                   title=f"Top 20 Feature Importances ({selected_model_name})")
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.info("This model type does not provide feature importance information.")
                        
                except Exception as e:
                    st.error(f"Could not generate feature importance plot: {e}")
                
                # 7. Model Download
                st.subheader("💾 Download Trained Model")
                try:
                    model_bytes = pickle.dumps(model)
                    st.download_button(
                        label="📥 Download Model (pickle)",
                        data=model_bytes,
                        file_name=f"{selected_model_name.lower().replace(' ', '_')}_model.pkl",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Could not serialize model: {e}")
                
            ### --- END: ENHANCED MODELING SECTION --- ###

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
    st.sidebar.subheader("Source Code")
    st.sidebar.info("This application is available as open source code.")

if __name__ == "__main__":
    # Ensure all helper/webscraping functions are defined or pasted back in before running
    # This is a placeholder for a complete run
    main()
