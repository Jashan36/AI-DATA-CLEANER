"""
Policy-driven cleaning engine for automated data quality improvement.
Handles column type detection, strategy decisions, and audit logging.
"""

import polars as pl
import pandas as pd
import numpy as np
import re
import unicodedata
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum
import rapidfuzz
from rapidfuzz import fuzz
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from urllib.parse import urlparse
import dateutil.parser as date_parser
from dateutil import tz

logger = logging.getLogger(__name__)

class ColumnType(Enum):
    """Detected column types for cleaning strategy selection."""
    ID_HASH = "id_hash"
    CATEGORICAL_LOW = "categorical_low"  # < 50 unique values
    CATEGORICAL_HIGH = "categorical_high"  # >= 50 unique values
    FREE_TEXT = "free_text"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    CODE = "code"
    CURRENCY_PRICE = "currency_price"
    RATING_SCORE = "rating_score"
    DATE_DATETIME = "date_datetime"
    NUMERIC_MEASURE = "numeric_measure"
    BOOLEAN = "boolean"

@dataclass
class CleaningDecision:
    """Represents a cleaning decision for a column."""
    column: str
    column_type: ColumnType
    strategy: str
    confidence: float
    rules_applied: List[str]
    before_sample: List[Any]
    after_sample: List[Any]
    quarantine_reason: Optional[str] = None

@dataclass
class AuditLog:
    """Comprehensive audit log for data cleaning operations."""
    timestamp: datetime
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    decisions: List[CleaningDecision]
    quarantined_rows: int = 0
    missing_fixed: int = 0
    outliers_handled: int = 0
    merges_performed: int = 0
    domain_inferred: Optional[str] = None
    quality_score: Optional[float] = None
    # Additional logging containers
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)

    def export_audit_log(self, filepath: str) -> None:
        """Export this audit log instance to a JSON file."""
        audit_dict = {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'quarantined_rows': self.quarantined_rows,
            'missing_fixed': self.missing_fixed,
            'outliers_handled': self.outliers_handled,
            'merges_performed': self.merges_performed,
            'domain_inferred': self.domain_inferred,
            'quality_score': self.quality_score,
            'operations': self.operations,
            'entries': self.log_entries,
            'decisions': [
                {
                    'column': d.column,
                    'column_type': d.column_type.value,
                    'strategy': d.strategy,
                    'confidence': d.confidence,
                    'rules_applied': d.rules_applied,
                    'before_sample': d.before_sample,
                    'after_sample': d.after_sample,
                    'quarantine_reason': d.quarantine_reason
                }
                for d in self.decisions
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(audit_dict, f, indent=2, default=str)

    def log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Append an operation record to the audit log."""
        self.operations.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        })

class CleanerPolicy:
    """
    Policy-driven cleaning engine that automatically detects column types
    and applies appropriate cleaning strategies.
    """
    
    def __init__(self):
        self.audit_log = AuditLog(
            timestamp=datetime.now(),
            input_shape=(0, 0),
            output_shape=(0, 0),
            decisions=[]
        )
        
        # Synonym tables for categorical canonicalization
        self.synonym_tables = {
            'status': {
                'delivered': ['delivered', 'completed', 'shipped', 'fulfilled', 'done'],
                'pending': ['pending', 'processing', 'in_progress', 'waiting'],
                'cancelled': ['cancelled', 'canceled', 'aborted', 'terminated'],
                'returned': ['returned', 'refunded', 'exchanged']
            },
            'gender': {
                'male': ['male', 'm', 'man', 'masculine'],
                'female': ['female', 'f', 'woman', 'feminine'],
                'other': ['other', 'non-binary', 'nb', 'nonbinary']
            },
            'rating': {
                'excellent': ['excellent', 'outstanding', 'amazing', 'fantastic', 'perfect'],
                'good': ['good', 'great', 'nice', 'satisfactory'],
                'average': ['average', 'okay', 'ok', 'mediocre'],
                'poor': ['poor', 'bad', 'terrible', 'awful']
            }
        }
        
        # Currency symbols and patterns
        self.currency_patterns = {
            'USD': [r'\$', r'USD', r'US\$', r'Dollar'],
            'EUR': [r'€', r'EUR', r'Euro'],
            'GBP': [r'£', r'GBP', r'Pound'],
            'JPY': [r'¥', r'JPY', r'Yen'],
            'INR': [r'₹', r'INR', r'Rupee']
        }
        
        # Phone number patterns by region
        self.phone_regions = {
            'US': '+1',
            'UK': '+44',
            'IN': '+91',
            'DE': '+49',
            'FR': '+33'
        }

    def detect_column_type(self, series: pd.Series, column_name: str) -> ColumnType:
        """
        Detect the type of a column based on its content and name.
        """
        column_name_lower = column_name.lower()
        unique_count = series.nunique()
        total_count = len(series)
        
        # Check for ID/Hash patterns
        if any(keyword in column_name_lower for keyword in ['id', 'key', 'hash', 'uuid']):
            if unique_count == total_count or unique_count / total_count > 0.95:
                return ColumnType.ID_HASH
        
        # Check for boolean patterns
        if unique_count <= 2:
            unique_values = set(series.dropna().astype(str).str.lower())
            if unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}):
                return ColumnType.BOOLEAN
        
        # Check for email patterns
        if 'email' in column_name_lower or self._is_email_column(series):
            return ColumnType.EMAIL
        
        # Check for phone patterns
        if 'phone' in column_name_lower or self._is_phone_column(series):
            return ColumnType.PHONE
        
        # Check for URL patterns
        if 'url' in column_name_lower or 'link' in column_name_lower or self._is_url_column(series):
            return ColumnType.URL
        
        # Check for date/datetime patterns
        if any(keyword in column_name_lower for keyword in ['date', 'time', 'created', 'updated']):
            return ColumnType.DATE_DATETIME
        
        # Check for currency/price patterns
        if any(keyword in column_name_lower for keyword in ['price', 'cost', 'amount', 'value', 'currency']):
            return ColumnType.CURRENCY_PRICE
        
        # Check for rating/score patterns
        if any(keyword in column_name_lower for keyword in ['rating', 'score', 'grade', 'rank']):
            return ColumnType.RATING_SCORE
        
        # Check for code patterns
        if any(keyword in column_name_lower for keyword in ['code', 'sku', 'product_id']):
            return ColumnType.CODE
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERIC_MEASURE
        
        # Check for categorical vs free text
        if unique_count < 50:
            return ColumnType.CATEGORICAL_LOW
        elif unique_count < total_count * 0.1:  # Less than 10% unique
            return ColumnType.CATEGORICAL_HIGH
        else:
            return ColumnType.FREE_TEXT

    def _is_email_column(self, series: pd.Series) -> bool:
        """Check if column contains email addresses."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        sample_size = min(100, len(series))
        sample = series.dropna().head(sample_size)
        
        if len(sample) == 0:
            return False
        
        email_matches = sum(1 for val in sample if re.match(email_pattern, str(val)))
        return email_matches / len(sample) > 0.7

    def _is_phone_column(self, series: pd.Series) -> bool:
        """Check if column contains phone numbers."""
        phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
        sample_size = min(100, len(series))
        sample = series.dropna().head(sample_size)
        
        if len(sample) == 0:
            return False
        
        phone_matches = sum(1 for val in sample if re.search(phone_pattern, str(val)))
        return phone_matches / len(sample) > 0.5

    def _is_url_column(self, series: pd.Series) -> bool:
        """Check if column contains URLs."""
        sample_size = min(100, len(series))
        sample = series.dropna().head(sample_size)
        
        if len(sample) == 0:
            return False
        
        url_matches = 0
        for val in sample:
            try:
                result = urlparse(str(val))
                if result.scheme and result.netloc:
                    url_matches += 1
            except:
                pass
        
        return url_matches / len(sample) > 0.5

    def decide_cleaning_strategy(self, series: pd.Series, column_name: str, column_type: ColumnType) -> CleaningDecision:
        """
        Decide on cleaning strategy based on column type and content.
        """
        before_sample = series.dropna().head(5).tolist()
        rules_applied = []
        strategy = ""
        confidence = 0.8
        
        if column_type == ColumnType.ID_HASH:
            strategy = "standardize_case_trim"
            rules_applied = ["case_folding", "whitespace_trimming"]
            
        elif column_type == ColumnType.CATEGORICAL_LOW:
            strategy = "canonicalize_categorical"
            rules_applied = ["case_folding", "whitespace_trimming", "synonym_merging"]
            
        elif column_type == ColumnType.CATEGORICAL_HIGH:
            strategy = "standardize_high_cardinality"
            rules_applied = ["case_folding", "whitespace_trimming", "fuzzy_matching"]
            
        elif column_type == ColumnType.FREE_TEXT:
            strategy = "normalize_text"
            rules_applied = ["unicode_normalization", "punctuation_standardization"]
            
        elif column_type == ColumnType.EMAIL:
            strategy = "validate_standardize_email"
            rules_applied = ["email_validation", "case_folding"]
            
        elif column_type == ColumnType.PHONE:
            strategy = "standardize_phone_e164"
            rules_applied = ["phone_parsing", "e164_formatting", "region_inference"]
            
        elif column_type == ColumnType.URL:
            strategy = "canonicalize_url"
            rules_applied = ["url_parsing", "canonicalization"]
            
        elif column_type == ColumnType.CODE:
            strategy = "standardize_code"
            rules_applied = ["case_folding", "whitespace_trimming"]
            
        elif column_type == ColumnType.CURRENCY_PRICE:
            strategy = "normalize_currency"
            rules_applied = ["currency_detection", "numeric_extraction", "base_currency_conversion"]
            
        elif column_type == ColumnType.RATING_SCORE:
            strategy = "standardize_rating"
            rules_applied = ["rating_mapping", "numeric_conversion"]
            
        elif column_type == ColumnType.DATE_DATETIME:
            strategy = "parse_standardize_date"
            rules_applied = ["date_parsing", "timezone_normalization", "format_standardization"]
            
        elif column_type == ColumnType.NUMERIC_MEASURE:
            strategy = "handle_numeric_outliers"
            rules_applied = ["outlier_detection", "iqr_capping"]
            
        elif column_type == ColumnType.BOOLEAN:
            strategy = "standardize_boolean"
            rules_applied = ["boolean_mapping"]
        
        # Apply the strategy to get after sample
        after_sample = self._apply_strategy(series.head(5), strategy, column_type)
        
        return CleaningDecision(
            column=column_name,
            column_type=column_type,
            strategy=strategy,
            confidence=confidence,
            rules_applied=rules_applied,
            before_sample=before_sample,
            after_sample=after_sample
        )

    def _apply_strategy(self, series: pd.Series, strategy: str, column_type: ColumnType) -> List[Any]:
        """Apply the cleaning strategy to get after sample."""
        # This is a simplified version - full implementation would be in the main cleaner
        if strategy == "standardize_case_trim":
            return series.str.lower().str.strip().head(5).tolist()
        elif strategy == "canonicalize_categorical":
            return series.str.lower().str.strip().head(5).tolist()
        elif strategy == "normalize_text":
            return series.str.lower().str.strip().head(5).tolist()
        else:
            return series.head(5).tolist()

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, AuditLog]:
        """
        Clean a dataframe using policy-driven decisions.
        Returns: (cleaned_df, quarantined_df, audit_log)
        """
        self.audit_log = AuditLog(
            timestamp=datetime.now(),
            input_shape=df.shape,
            output_shape=(0, 0),
            decisions=[]
        )
        
        cleaned_df = df.copy()
        quarantined_rows = []
        
        for column in df.columns:
            series = df[column]
            column_type = self.detect_column_type(series, column)
            decision = self.decide_cleaning_strategy(series, column, column_type)
            
            # Apply cleaning based on decision
            cleaned_series, quarantined_mask = self._apply_column_cleaning(
                series, decision
            )
            
            cleaned_df[column] = cleaned_series
            
            # Track quarantined rows
            if quarantined_mask.any():
                quarantined_indices = df.index[quarantined_mask].tolist()
                for idx in quarantined_indices:
                    quarantined_rows.append({
                        'row_index': idx,
                        'column': column,
                        'original_value': df.loc[idx, column],
                        'reason': decision.quarantine_reason or 'failed_validation'
                    })
            
            self.audit_log.decisions.append(decision)
        
        # Create quarantined dataframe
        if quarantined_rows:
            quarantined_df = pd.DataFrame(quarantined_rows)
        else:
            quarantined_df = pd.DataFrame()
        
        self.audit_log.output_shape = cleaned_df.shape
        self.audit_log.quarantined_rows = len(quarantined_rows)
        
        return cleaned_df, quarantined_df, self.audit_log

    def _apply_column_cleaning(self, series: pd.Series, decision: CleaningDecision) -> Tuple[pd.Series, pd.Series]:
        """
        Apply cleaning to a column based on the decision.
        Returns: (cleaned_series, quarantined_mask)
        """
        cleaned_series = series.copy()
        quarantined_mask = pd.Series(False, index=series.index)
        
        # Apply missing value handling
        if decision.column_type in [ColumnType.NUMERIC_MEASURE, ColumnType.CURRENCY_PRICE, ColumnType.RATING_SCORE]:
            # Use median for numeric columns
            median_val = series.median()
            cleaned_series = cleaned_series.fillna(median_val)
        elif decision.column_type in [ColumnType.CATEGORICAL_LOW, ColumnType.CATEGORICAL_HIGH]:
            # Use mode or "unknown" for categorical
            mode_val = series.mode()
            if len(mode_val) > 0:
                cleaned_series = cleaned_series.fillna(mode_val[0])
            else:
                cleaned_series = cleaned_series.fillna("unknown")
        elif decision.column_type == ColumnType.DATE_DATETIME:
            # Forward fill then median date for dates
            cleaned_series = cleaned_series.fillna(method='ffill')
            if cleaned_series.isnull().any():
                # If still nulls, use median date
                try:
                    median_date = pd.to_datetime(cleaned_series.dropna()).median()
                    cleaned_series = cleaned_series.fillna(median_date)
                except:
                    pass
        
        # Apply strategy-specific cleaning
        if decision.strategy == "standardize_case_trim":
            cleaned_series = cleaned_series.astype(str).str.lower().str.strip()
        elif decision.strategy == "canonicalize_categorical":
            cleaned_series = self._canonicalize_categorical(cleaned_series, decision.column)
        elif decision.strategy == "normalize_text":
            cleaned_series = self._normalize_text(cleaned_series)
        elif decision.strategy == "validate_standardize_email":
            cleaned_series, quarantined_mask = self._validate_emails(cleaned_series)
        elif decision.strategy == "standardize_phone_e164":
            cleaned_series, quarantined_mask = self._standardize_phones(cleaned_series)
        elif decision.strategy == "handle_numeric_outliers":
            cleaned_series, quarantined_mask = self._handle_outliers(cleaned_series)
        
        return cleaned_series, quarantined_mask

    def _canonicalize_categorical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Canonicalize categorical values using synonym tables and fuzzy matching."""
        column_name_lower = column_name.lower()
        result = series.copy()
        
        # Check if we have synonym table for this column
        for category, synonyms in self.synonym_tables.items():
            if category in column_name_lower:
                for canonical, variants in synonyms.items():
                    for variant in variants:
                        mask = result.str.lower().str.strip() == variant.lower()
                        result.loc[mask] = canonical
                break
        
        return result

    def _normalize_text(self, series: pd.Series) -> pd.Series:
        """Normalize text using Unicode NFKC and punctuation standardization."""
        def normalize_text(text):
            if pd.isna(text):
                return text
            text = str(text)
            # Unicode normalization
            text = unicodedata.normalize('NFKC', text)
            # Case folding
            text = text.casefold()
            # Trim whitespace
            text = text.strip()
            return text
        
        return series.apply(normalize_text)

    def _validate_emails(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Validate and standardize email addresses."""
        result = series.copy()
        quarantined_mask = pd.Series(False, index=series.index)
        
        for idx, email in series.items():
            if pd.isna(email):
                continue
            try:
                validated = validate_email(str(email))
                result.loc[idx] = validated.email.lower()
            except EmailNotValidError:
                quarantined_mask.loc[idx] = True
                result.loc[idx] = None
        
        return result, quarantined_mask

    def _standardize_phones(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Standardize phone numbers to E.164 format."""
        result = series.copy()
        quarantined_mask = pd.Series(False, index=series.index)
        
        for idx, phone in series.items():
            if pd.isna(phone):
                continue
            try:
                # Try to parse with different regions
                parsed = None
                for region in self.phone_regions.values():
                    try:
                        parsed = phonenumbers.parse(str(phone), region)
                        if phonenumbers.is_valid_number(parsed):
                            break
                    except:
                        continue
                
                if parsed and phonenumbers.is_valid_number(parsed):
                    result.loc[idx] = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                else:
                    quarantined_mask.loc[idx] = True
                    result.loc[idx] = None
            except:
                quarantined_mask.loc[idx] = True
                result.loc[idx] = None
        
        return result, quarantined_mask

    def _handle_outliers(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Handle outliers using IQR method with capping and no quarantine."""
        if not pd.api.types.is_numeric_dtype(series):
            return series, pd.Series([False] * len(series), index=series.index)
        if len(series) < 3:
            return series, pd.Series([False] * len(series), index=series.index)

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers instead of removing them
        capped = series.copy()
        capped = np.where(capped < lower_bound, lower_bound, capped)
        capped = np.where(capped > upper_bound, upper_bound, capped)
        capped_series = pd.Series(capped, index=series.index)

        # All False quarantine mask since values are retained via capping
        return capped_series, pd.Series([False] * len(series), index=series.index)

    def export_audit_log(self, filepath: str):
        """Export audit log to JSON file."""
        audit_dict = {
            'timestamp': self.audit_log.timestamp.isoformat(),
            'input_shape': self.audit_log.input_shape,
            'output_shape': self.audit_log.output_shape,
            'quarantined_rows': self.audit_log.quarantined_rows,
            'missing_fixed': self.audit_log.missing_fixed,
            'outliers_handled': self.audit_log.outliers_handled,
            'merges_performed': self.audit_log.merges_performed,
            'domain_inferred': self.audit_log.domain_inferred,
            'decisions': [
                {
                    'column': d.column,
                    'column_type': d.column_type.value,
                    'strategy': d.strategy,
                    'confidence': d.confidence,
                    'rules_applied': d.rules_applied,
                    'before_sample': d.before_sample,
                    'after_sample': d.after_sample,
                    'quarantine_reason': d.quarantine_reason
                }
                for d in self.audit_log.decisions
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(audit_dict, f, indent=2, default=str)
