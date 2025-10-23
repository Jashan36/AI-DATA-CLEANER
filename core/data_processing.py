"""
Advanced data preprocessing utilities and JAX-powered cleaning pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
from dateutil import parser as date_parser
import json
import re

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from sklearn.preprocessing import LabelEncoder, StandardScaler


# -----------------------------
# Pure Data Processing Utilities
# -----------------------------

def analyze_data_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Analyze basic data quality issues (missing values, case inconsistencies, duplicates)."""
    issues: Dict[str, Any] = {}
    recommendations = []

    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        issues["missing_values"] = missing_data[missing_data > 0].to_dict()
        recommendations.append("Handle missing values before training.")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    case_issues: Dict[str, Any] = {}

    for col in categorical_cols:
        if col == target_col:
            continue
        values = df[col].dropna().astype(str)
        unique_lower = set(values.str.lower().unique())
        unique_original = set(values.unique())
        if len(unique_lower) < len(unique_original):
            case_issues[col] = {
                "original_unique": len(unique_original),
                "normalized_unique": len(unique_lower),
                "examples": list(unique_original)[:10],
            }

    if case_issues:
        issues["case_inconsistencies"] = case_issues
        recommendations.append("Normalize case in categorical columns to merge duplicate categories.")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues["duplicate_rows"] = duplicates
        recommendations.append(f"Remove {duplicates} duplicate rows.")

    return {
        "issues": issues,
        "recommendations": recommendations,
        "total_issues": len(issues),
    }


def detect_target_type(y_series: pd.Series, threshold: float = 0.05) -> str:
    """Detect whether target is 'regression' or 'classification' based on discreteness."""
    try:
        if not pd.api.types.is_numeric_dtype(y_series):
            return "classification"

        y_clean = y_series.dropna()
        if len(y_clean) == 0:
            return "classification"

        if all(y_clean == y_clean.astype(int)):
            unique_ratio = len(y_clean.unique()) / len(y_clean)
            if unique_ratio <= threshold:
                return "classification"

        if len(y_clean.unique()) <= 10:
            return "classification"

        return "regression"
    except Exception:
        return "classification"


@dataclass
class JAXDataCleaner:
    """JAX-powered data cleaning pipeline with audit logging (pure processing)."""

    scaler: StandardScaler = field(default_factory=StandardScaler)
    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    feature_stats: Dict[str, Any] = field(default_factory=dict)
    outlier_threshold: float = 3.0
    imputation_method: str = "mean"
    categorical_missing_strategy: str = "unknown"
    cleaning_log: Dict[str, Any] = field(
        default_factory=lambda: {
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "transformations": {},
            "statistics": {},
            "warnings": [],
            "errors": [],
        }
    )

    domain_rules: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "healthcare": {
                "symptoms": ["fever", "cough", "headache", "nausea", "fatigue"],
                "diagnoses": ["hypertension", "diabetes", "asthma", "pneumonia"],
                "medications": ["aspirin", "ibuprofen", "acetaminophen"],
            },
            "finance": {
                "currencies": ["usd", "eur", "gbp", "jpy", "cad", "aud"],
                "account_types": ["checking", "savings", "credit", "investment"],
                "transaction_types": ["deposit", "withdrawal", "transfer", "payment"],
            },
            "retail": {
                "categories": ["electronics", "clothing", "books", "home", "sports"],
                "statuses": ["delivered", "pending", "cancelled", "returned", "shipped"],
            },
        }
    )

    synonym_mappings: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "status": {
                "delivered": ["delivered", "completed", "shipped", "fulfilled"],
                "pending": ["pending", "processing", "in_progress", "awaiting"],
                "cancelled": ["cancelled", "canceled", "failed", "aborted"],
                "returned": ["returned", "refunded", "rejected"],
            },
            "gender": {
                "male": ["male", "m", "man", "masculine"],
                "female": ["female", "f", "woman", "feminine"],
                "other": ["other", "non-binary", "nb", "prefer_not_to_say"],
            },
            "rating": {
                "excellent": ["excellent", "outstanding", "amazing", "perfect", "5"],
                "good": ["good", "great", "satisfactory", "4"],
                "average": ["average", "ok", "okay", "fair", "3"],
                "poor": ["poor", "bad", "terrible", "awful", "2"],
                "very_poor": ["very poor", "worst", "horrible", "1"],
            },
        }
    )

    # --------------------- Internal helpers ---------------------
    def _log(self, column: str, operation: str, **kwargs) -> None:
        self.cleaning_log["steps"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "column": column,
                "operation": operation,
                **kwargs,
            }
        )

    def _detect_domain(self, df: pd.DataFrame) -> str:
        column_names = [col.lower() for col in df.columns]
        domain_scores: Dict[str, int] = defaultdict(int)

        for kw in ["patient", "diagnosis", "symptom", "medication", "treatment", "hospital", "doctor", "age", "gender"]:
            if any(kw in c for c in column_names):
                domain_scores["healthcare"] += 1

        for kw in ["account", "balance", "transaction", "amount", "currency", "payment", "credit", "debit"]:
            if any(kw in c for c in column_names):
                domain_scores["finance"] += 1

        for kw in ["product", "price", "category", "order", "customer", "rating", "review", "inventory"]:
            if any(kw in c for c in column_names):
                domain_scores["retail"] += 1

        return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"

    def set_parameters(self, outlier_threshold: float = 3.0, imputation_method: str = "mean", categorical_missing_strategy: str = "unknown") -> None:
        self.outlier_threshold = float(outlier_threshold)
        self.imputation_method = str(imputation_method)
        self.categorical_missing_strategy = str(categorical_missing_strategy)
    # ------------------ Special column cleaners ------------------
    def _clean_price_column(self, series: pd.Series) -> pd.Series:
        def clean_price(val: Any) -> float:
            try:
                if pd.isnull(val):
                    return np.nan
                s = str(val).lower()
                if "free" in s:
                    return 0.0
                for word in ["rs", "$", "usd", "inr", "₹", "approx.", "approx", "eur", "gbp", "cad", "aud", "n/a", "na", "none"]:
                    s = s.replace(word, "")
                digits = "".join(c for c in s if c.isdigit() or c == ".")
                return float(digits) if digits else np.nan
            except Exception:
                return np.nan

        return series.apply(clean_price)

    def _clean_rating_column(self, series: pd.Series) -> pd.Series:
        rating_map = {"bad": 1, "poor": 2, "ok": 3, "average": 3, "good": 4, "excellent": 5, "n/a": np.nan, "na": np.nan, "none": np.nan}

        def clean_rating(val: Any) -> float:
            try:
                if pd.isnull(val):
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                s = str(val).strip().lower()
                if s in rating_map:
                    return rating_map[s]
                return float(s)
            except Exception:
                return np.nan

        return series.apply(clean_rating)

    def _clean_status_column(self, series: pd.Series) -> pd.Series:
        def clean_status(val: Any) -> Optional[str]:
            if pd.isnull(val):
                return np.nan
            s = str(val).strip().lower()
            for standard, syns in self.synonym_mappings.get("status", {}).items():
                if s in syns:
                    return standard
            mapping = {
                "delivered": "delivered",
                "pending": "pending",
                "returned": "returned",
                "cancelled": "cancelled",
                "canceled": "cancelled",
                "processing": "pending",
                "shipped": "delivered",
                "completed": "delivered",
                "failed": "cancelled",
                "refunded": "returned",
            }
            return mapping.get(s, s)

        cleaned = series.apply(clean_status)
        self._log(
            "STATUS",
            "status_normalization",
            original_unique=int(series.nunique(dropna=True)),
            cleaned_unique=int(cleaned.nunique(dropna=True)),
            missing_count=int(series.isnull().sum()),
        )
        return cleaned
    def _clean_date_column(self, series: pd.Series) -> pd.Series:
        original_missing = int(series.isnull().sum())
        parsing_errors = 0
        successful = 0

        def parse_any(val: Any) -> Optional[str]:
            nonlocal parsing_errors, successful
            try:
                if pd.isnull(val):
                    return np.nan
                s = str(val).strip()
                try:
                    dt = pd.to_datetime(s, infer_datetime_format=True, errors="raise")
                    successful += 1
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                try:
                    dt = date_parser.parse(s, fuzzy=True)
                    successful += 1
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                year_match = re.search(r"(\d{4})", s)
                if year_match:
                    year = year_match.group(1)
                    md = re.search(r"(\d{1,2})[\/\-\.\s]+(\d{1,2})", s)
                    if md:
                        m, d = md.groups()
                        for fmt in ["%Y-%m-%d", "%Y-%d-%m"]:
                            try:
                                from datetime import datetime as _dt

                                dt = _dt.strptime(f"{year}-{m.zfill(2)}-{d.zfill(2)}", fmt)
                                successful += 1
                                return dt.strftime("%Y-%m-%d")
                            except Exception:
                                continue
                parsing_errors += 1
                return None
            except Exception:
                parsing_errors += 1
                return None

        cleaned = series.apply(parse_any)
        if parsing_errors > 0:
            cleaned = cleaned.fillna(method="ffill")
            valid = pd.to_datetime(cleaned.dropna(), errors="coerce")
            if not valid.empty:
                median_date = valid.median()
                cleaned = cleaned.fillna(median_date.strftime("%Y-%m-%d"))
            else:
                from datetime import datetime as _dt

                cleaned = cleaned.fillna(_dt.now().strftime("%Y-%m-%d"))

        self._log(
            "DATE",
            "date_standardization",
            original_missing=original_missing,
            parsing_errors=int(parsing_errors),
            successful_parses=int(successful),
            final_missing=int(cleaned.isnull().sum()),
        )
        return cleaned
    def _clean_age_column(self, series: pd.Series) -> pd.Series:
        invalid = 0

        def f(val: Any) -> Optional[int]:
            nonlocal invalid
            try:
                if pd.isnull(val):
                    return np.nan
                age = pd.to_numeric(val, errors="coerce")
                if pd.isnull(age) or age < 0 or age > 150:
                    invalid += 1
                    return np.nan
                return int(age)
            except Exception:
                invalid += 1
                return np.nan

        cleaned = series.apply(f)
        self._log("AGE", "age_validation", invalid_values=int(invalid))
        return cleaned

    def _clean_email_column(self, series: pd.Series) -> pd.Series:
        pat = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        invalid = 0

        def f(val: Any) -> Optional[str]:
            nonlocal invalid
            try:
                if pd.isnull(val):
                    return np.nan
                s = str(val).strip().lower()
                if pat.match(s):
                    return s
                invalid += 1
                return np.nan
            except Exception:
                invalid += 1
                return np.nan

        cleaned = series.apply(f)
        self._log("EMAIL", "email_validation", invalid_values=int(invalid))
        return cleaned

    def _clean_phone_column(self, series: pd.Series) -> pd.Series:
        invalid = 0

        def f(val: Any) -> Optional[str]:
            nonlocal invalid
            try:
                if pd.isnull(val):
                    return np.nan
                s = re.sub(r"\D", "", str(val).strip())
                if len(s) < 7 or len(s) > 15:
                    invalid += 1
                    return np.nan
                if len(s) == 10:
                    return f"+1{s}"
                if len(s) == 11 and s.startswith("1"):
                    return f"+{s}"
                return f"+{s}"
            except Exception:
                invalid += 1
                return np.nan

        cleaned = series.apply(f)
        self._log("PHONE", "phone_standardization", invalid_values=int(invalid))
        return cleaned

    def _clean_score_column(self, series: pd.Series) -> pd.Series:
        invalid = 0

        def f(val: Any) -> float:
            nonlocal invalid
            try:
                if pd.isnull(val):
                    return np.nan
                s = str(val).strip().lower()
                if "%" in s:
                    s = s.replace("%", "")
                    v = float(s)
                    return v if 0 <= v <= 100 else np.nan
                if "/" in s:
                    a, b = s.split("/")[:2]
                    a, b = float(a), float(b)
                    return (a / b) * 100 if b > 0 else np.nan
                mapping = {
                    "a+": 98,
                    "a": 95,
                    "a-": 92,
                    "b+": 88,
                    "b": 85,
                    "b-": 82,
                    "c+": 78,
                    "c": 75,
                    "c-": 72,
                    "d+": 68,
                    "d": 65,
                    "d-": 62,
                    "f": 0,
                }
                if s in mapping:
                    return mapping[s]
                v = float(s)
                if 0 <= v <= 100:
                    return v
                if 0 <= v <= 1:
                    return v * 100
                if 0 <= v <= 4:
                    return (v / 4) * 100
                invalid += 1
                return np.nan
            except Exception:
                invalid += 1
                return np.nan

        cleaned = series.apply(f)
        if cleaned.isnull().any():
            valid = cleaned.dropna()
            cleaned = cleaned.fillna(valid.median() if not valid.empty else 75.0)
        self._log(
            "SCORE",
            "score_standardization",
            invalid_values=int(invalid),
            final_missing=int(cleaned.isnull().sum()),
        )
        return cleaned
    # ----------------------- JAX kernels -----------------------
    def detect_outliers_jax(self, data: jnp.ndarray) -> jnp.ndarray:
        @jax.jit
        def z_score_outliers(x: jnp.ndarray) -> jnp.ndarray:
            mean = jnp.nanmean(x, axis=0)
            std = jnp.nanstd(x, axis=0)
            z = jnp.abs((x - mean) / (std + 1e-8))
            return jnp.any(z > self.outlier_threshold, axis=1)

        return z_score_outliers(data)

    def fill_missing_values_jax(self, data: jnp.ndarray) -> jnp.ndarray:
        @jax.jit
        def impute(x: jnp.ndarray) -> jnp.ndarray:
            if self.imputation_method == "mean":
                fill = jnp.nanmean(x, axis=0)
            elif self.imputation_method == "median":
                fill = jnp.nanmedian(x, axis=0)
            else:
                fill = jnp.zeros(x.shape[1])
            return jnp.where(jnp.isnan(x), fill, x)

        return impute(data)

    def handle_outliers(self, data: jnp.ndarray, strategy: str = "cap") -> jnp.ndarray:
        @jax.jit
        def process(x: jnp.ndarray) -> jnp.ndarray:
            q1 = jnp.nanpercentile(x, 25, axis=0)
            q3 = jnp.nanpercentile(x, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            if strategy == "remove":
                mask = (x < lower) | (x > upper)
                return jnp.where(mask, jnp.nan, x)
            if strategy == "cap":
                x = jnp.where(x < lower, lower, x)
                x = jnp.where(x > upper, upper, x)
                return x
            return x

        return process(data)
    # ---------------------- Main pipeline ----------------------
    def _validate_schema(self, df: pd.DataFrame) -> list:
        issues = []
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns found: {empty_cols}")
        single_value_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if single_value_cols:
            issues.append(f"Columns with single value: {single_value_cols}")
        potential_id_cols = []
        for col in df.columns:
            if any(k in col.lower() for k in ["id", "key", "code", "number"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    potential_id_cols.append(col)
        if potential_id_cols:
            issues.append(f"Potential ID columns with numeric type: {potential_id_cols}")
        return issues

    def _apply_synonym_mapping(self, series: pd.Series, mappings: Dict[str, Any]) -> pd.Series:
        def map_value(val: Any) -> Any:
            if pd.isnull(val):
                return val
            s = str(val).strip().lower()
            for standard, syns in mappings.items():
                if s in syns:
                    return standard
            return s

        return series.apply(map_value)

    @property
    def special_cleaners(self) -> Dict[str, Any]:
        return {
            "price": self._clean_price_column,
            "rating": self._clean_rating_column,
            "status": self._clean_status_column,
            "date": self._clean_date_column,
            "age": self._clean_age_column,
            "email": self._clean_email_column,
            "phone": self._clean_phone_column,
            "score": self._clean_score_column,
        }
    def preprocess_data(self, df: pd.DataFrame, outlier_strategy: str = "cap") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        try:
            self.cleaning_log = {
                "timestamp": datetime.now().isoformat(),
                "steps": [],
                "transformations": {},
                "statistics": {},
                "warnings": [],
                "errors": [],
            }

            processed_df = df.copy()
            stats: Dict[str, Any] = {}

            detected_domain = self._detect_domain(df)
            self._log("DATASET", "domain_detection", detected_domain=detected_domain)

            schema_issues = self._validate_schema(df)
            if schema_issues:
                self.cleaning_log["warnings"].extend(schema_issues)
                self._log("DATASET", "schema_validation", issues_found=len(schema_issues))

            original_columns = list(df.columns)
            processed_df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if original_columns != list(processed_df.columns):
                self._log(
                    "HEADERS",
                    "header_normalization",
                    original_columns=original_columns,
                    normalized_columns=list(processed_df.columns),
                )

            for col in list(processed_df.columns):
                for key, func in self.special_cleaners.items():
                    if key == col.lower():
                        before = {
                            "unique_count": int(processed_df[col].nunique(dropna=True)),
                            "missing_count": int(processed_df[col].isnull().sum()),
                            "dtype": str(processed_df[col].dtype),
                        }
                        processed_df[col] = func(processed_df[col])
                        after = {
                            "unique_count": int(processed_df[col].nunique(dropna=True)),
                            "missing_count": int(processed_df[col].isnull().sum()),
                            "dtype": str(processed_df[col].dtype),
                        }
                        self._log(col, "special_cleaning", original_stats=before, new_stats=after)
            categorical_cols = processed_df.select_dtypes(include=["object", "category"]).columns
            for col in categorical_cols:
                original_unique = int(processed_df[col].nunique(dropna=True))
                original_missing = int(processed_df[col].isnull().sum())
                col_lower = col.lower()
                for category_type, mappings in self.synonym_mappings.items():
                    if category_type in col_lower:
                        processed_df[col] = self._apply_synonym_mapping(processed_df[col], mappings)
                        break
                processed_df[col] = processed_df[col].astype(str).str.strip().str.lower()
                new_unique = int(processed_df[col].nunique(dropna=True))
                self._log(
                    col,
                    "categorical_standardization",
                    original_unique=original_unique,
                    new_unique=new_unique,
                    missing_count=original_missing,
                )

            for col in categorical_cols:
                if processed_df[col].isnull().any():
                    missing_count = int(processed_df[col].isnull().sum())
                    if self.categorical_missing_strategy == "unknown":
                        processed_df[col] = processed_df[col].fillna("unknown")
                        strat = "unknown"
                    elif self.categorical_missing_strategy == "mode":
                        mode_series = processed_df[col].mode()
                        mode_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
                        processed_df[col] = processed_df[col].fillna(mode_val)
                        strat = f"mode:{mode_val}"
                    elif self.categorical_missing_strategy == "forward_fill":
                        processed_df[col] = processed_df[col].fillna(method="ffill").fillna(method="bfill")
                        strat = "forward_fill"
                    else:
                        processed_df[col] = processed_df[col].fillna("unknown")
                        strat = "unknown_default"
                    self._log(col, "missing_value_imputation", missing_count=missing_count, strategy=strat)

            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                original_values = processed_df[col].unique()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))
                enc = dict(enumerate(self.label_encoders[col].classes_))
                rev = {v: k for k, v in enc.items()}
                self.cleaning_log["transformations"][col] = {
                    "type": "categorical_encoding",
                    "mapping": enc,
                    "reverse_mapping": rev,
                    "original_unique": len(original_values),
                    "encoded_unique": len(enc),
                }
                self._log(
                    col,
                    "categorical_encoding",
                    original_unique=len(original_values),
                    encoded_unique=len(enc),
                )

            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X = jnp.array(processed_df[numeric_cols].values, dtype=jnp.float32)
                X = self.handle_outliers(X, outlier_strategy)
                X = self.fill_missing_values_jax(X)
                outlier_mask = self.detect_outliers_jax(X)
                stats["outlier_count"] = int(jnp.sum(outlier_mask))
                stats["outlier_percentage"] = float(jnp.mean(outlier_mask) * 100.0)
                self._log(
                    "NUMERIC_COLUMNS",
                    "outlier_handling",
                    outlier_count=stats["outlier_count"],
                    outlier_percentage=stats["outlier_percentage"],
                    strategy=outlier_strategy,
                )
                processed_df[numeric_cols] = np.asarray(X)

            stats["original_shape"] = df.shape
            stats["processed_shape"] = processed_df.shape
            stats["missing_values_original"] = int(df.isnull().sum().sum())
            stats["missing_values_processed"] = int(processed_df.isnull().sum().sum())
            stats["dtypes"] = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}
            stats["domain"] = detected_domain

            self.cleaning_log["statistics"] = stats
            self.cleaning_log["transformations"]["dataset"] = {
                "original_shape": df.shape,
                "processed_shape": processed_df.shape,
                "columns_processed": len(processed_df.columns),
                "domain": detected_domain,
            }
            self.feature_stats = stats
            return processed_df, stats
        except Exception as e:
            msg = f"Error in preprocess_data: {e}"
            self.cleaning_log["errors"].append(msg)
            return df, {}

    def preprocess_dual_output(self, df: pd.DataFrame, outlier_strategy: str = "cap"):
        """Return raw-but-tidy and ML-ready versions."""
        try:
            raw_tidy_df = df.copy()
            stats: Dict[str, Any] = {}

            for col in list(raw_tidy_df.columns):
                for key, func in self.special_cleaners.items():
                    if key == col.lower():
                        raw_tidy_df[col] = func(raw_tidy_df[col])

            ml_ready_df = raw_tidy_df.copy()

            cat_cols = ml_ready_df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                ml_ready_df[col] = self.label_encoders[col].fit_transform(ml_ready_df[col].astype(str))

            num_cols = ml_ready_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                X = jnp.array(ml_ready_df[num_cols].values, dtype=jnp.float32)
                X = self.handle_outliers(X, outlier_strategy)
                X = self.fill_missing_values_jax(X)
                ml_ready_df[num_cols] = np.asarray(X)

                Xr = jnp.array(raw_tidy_df[num_cols].values, dtype=jnp.float32)
                Xr = self.handle_outliers(Xr, outlier_strategy)
                Xr = self.fill_missing_values_jax(Xr)
                raw_tidy_df[num_cols] = np.asarray(Xr)

            stats["original_shape"] = df.shape
            stats["raw_tidy_shape"] = raw_tidy_df.shape
            stats["ml_ready_shape"] = ml_ready_df.shape
            stats["missing_values_original"] = int(df.isnull().sum().sum())
            stats["missing_values_raw_tidy"] = int(raw_tidy_df.isnull().sum().sum())
            stats["missing_values_ml_ready"] = int(ml_ready_df.isnull().sum().sum())
            stats["dtypes"] = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}

            self.feature_stats = stats
            return raw_tidy_df, ml_ready_df, stats
        except Exception:
            return df, df, {}

    def get_cleaning_log(self) -> Dict[str, Any]:
        return self.cleaning_log

    def export_cleaning_log(self, format: str = "json") -> str:
        if format == "json":
            return json.dumps(self.cleaning_log, indent=2, default=str)
        if format == "summary":
            summary = {
                "timestamp": self.cleaning_log["timestamp"],
                "total_steps": len(self.cleaning_log["steps"]),
                "warnings": len(self.cleaning_log["warnings"]),
                "errors": len(self.cleaning_log["errors"]),
                "transformations": len(self.cleaning_log["transformations"]),
                "statistics": self.cleaning_log["statistics"],
            }
            return json.dumps(summary, indent=2, default=str)
        return str(self.cleaning_log)

    def export_interpretable_mappings(self, format: str = "markdown") -> str:
        mappings: Dict[str, Any] = {}
        for col, info in self.cleaning_log.get("transformations", {}).items():
            if info.get("type") == "categorical_encoding":
                mappings[col] = {
                    "encoding": info["mapping"],
                    "reverse": info["reverse_mapping"],
                    "counts": info.get("original_unique", 0),
                }

        if format == "markdown":
            md = ["# Categorical Encoding Mappings", ""]
            for col, m in mappings.items():
                md.append(f"## Column: {col}")
                md.append(f"**Total unique values:** {m['counts']}")
                md.append("### Encoding (Integer -> Original)")
                md.append("| Code | Original Value |")
                md.append("|------|----------------|")
                for code, value in m["encoding"].items():
                    md.append(f"| {code} | {value} |")
                md.append("")
                md.append("### Reverse Mapping (Original -> Integer)")
                md.append("| Original Value | Code |")
                md.append("|----------------|------|")
                for value, code in m["reverse"].items():
                    md.append(f"| {value} | {code} |")
                md.append("\n---\n")
            return "\n".join(md)
        if format == "csv":
            rows = []
            for col, m in mappings.items():
                for code, value in m["encoding"].items():
                    rows.append({"column": col, "code": code, "original_value": value})
            if rows:
                df = pd.DataFrame(rows)
                return df.to_csv(index=False)
            return "column,code,original_value\n"
        return json.dumps(mappings, indent=2, default=str)


__all__ = ["JAXDataCleaner", "analyze_data_quality", "detect_target_type"]
