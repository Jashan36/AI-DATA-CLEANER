"""
Schema validation and quality gates using Great Expectations integration.
Handles data quality validation with auto-fix and quarantine capabilities.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import re

try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite
    from great_expectations.dataset import PandasDataset
    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    logging.warning("Great Expectations not available. Install with: pip install great-expectations")

logger = logging.getLogger(__name__)

@dataclass
class QualityViolation:
    """Represents a data quality violation."""
    column: str
    expectation_type: str
    violation_count: int
    violation_indices: List[int]
    violation_values: List[Any]
    reason: str
    severity: str  # 'error', 'warning', 'info'
    auto_fixable: bool = False
    quarantine_reason: Optional[str] = None

@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    timestamp: datetime
    total_rows: int
    total_columns: int
    violations: List[QualityViolation]
    quarantine_rows: List[int]
    auto_fixes_applied: List[str]
    quality_score: float
    domain_inferred: Optional[str] = None

class QualityGates:
    """
    Data quality validation and enforcement using Great Expectations.
    """
    
    def __init__(self):
        self.expectation_suite = None
        self.quality_report = None
        
        # Domain-specific quality rules
        self.domain_rules = {
            'healthcare': {
                'required_columns': ['patient_id', 'age', 'gender'],
                'age_range': (0, 150),
                'gender_values': ['male', 'female', 'other'],
                'email_pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            },
            'finance': {
                'required_columns': ['account_id', 'amount', 'transaction_date'],
                'amount_range': (0, 1000000),
                'currency_codes': ['USD', 'EUR', 'GBP', 'JPY', 'INR']
            },
            'retail': {
                'required_columns': ['product_id', 'price', 'category'],
                'price_range': (0, 10000),
                'status_values': ['active', 'inactive', 'discontinued']
            }
        }

    def create_expectation_suite(self, df: pd.DataFrame, domain: Optional[str] = None) -> ExpectationSuite:
        """
        Create a Great Expectations suite based on data analysis and domain rules.
        """
        if not GE_AVAILABLE:
            logger.warning("Great Expectations not available, using basic validation")
            return None
        
        # Convert to Great Expectations dataset
        ge_df = PandasDataset(df)
        
        # Basic expectations for all datasets
        expectations = []
        
        # Check for null values in key columns
        for col in df.columns:
            if 'id' in col.lower() or 'key' in col.lower():
                expectations.append(f"expect_column_values_to_not_be_null('{col}')")
        
        # Check for unique values in ID columns
        for col in df.columns:
            if 'id' in col.lower() and df[col].nunique() == len(df):
                expectations.append(f"expect_column_values_to_be_unique('{col}')")
        
        # Check data types
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                expectations.append(f"expect_column_values_to_be_of_type('{col}', 'numeric')")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                expectations.append(f"expect_column_values_to_be_of_type('{col}', 'datetime')")
        
        # Domain-specific expectations
        if domain and domain in self.domain_rules:
            domain_rule = self.domain_rules[domain]
            
            # Required columns
            for req_col in domain_rule.get('required_columns', []):
                if req_col in df.columns:
                    expectations.append(f"expect_column_values_to_not_be_null('{req_col}')")
            
            # Age range for healthcare
            if domain == 'healthcare' and 'age' in df.columns:
                age_range = domain_rule['age_range']
                expectations.append(f"expect_column_values_to_be_between('age', {age_range[0]}, {age_range[1]})")
            
            # Gender values for healthcare
            if domain == 'healthcare' and 'gender' in df.columns:
                gender_values = domain_rule['gender_values']
                expectations.append(f"expect_column_values_to_be_in_set('gender', {gender_values})")
            
            # Amount range for finance
            if domain == 'finance' and 'amount' in df.columns:
                amount_range = domain_rule['amount_range']
                expectations.append(f"expect_column_values_to_be_between('amount', {amount_range[0]}, {amount_range[1]})")
            
            # Price range for retail
            if domain == 'retail' and 'price' in df.columns:
                price_range = domain_rule['price_range']
                expectations.append(f"expect_column_values_to_be_between('price', {price_range[0]}, {price_range[1]})")
        
        # Apply expectations
        for expectation in expectations:
            try:
                eval(f"ge_df.{expectation}")
            except Exception as e:
                logger.warning(f"Failed to apply expectation {expectation}: {e}")
        
        return ge_df.get_expectation_suite()

    def validate_data(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityReport:
        """
        Validate data quality and generate comprehensive report.
        """
        violations = []
        quarantine_rows = set()
        auto_fixes_applied = []

        # Handle empty dataframe upfront
        if df is None or df.empty:
            self.quality_report = QualityReport(
                timestamp=datetime.now(),
                total_rows=0,
                total_columns=0,
                violations=violations,
                quarantine_rows=list(quarantine_rows),
                auto_fixes_applied=auto_fixes_applied,
                quality_score=1.0,
                domain_inferred=domain
            )
            return self.quality_report
        
        # Create expectation suite
        if GE_AVAILABLE:
            self.expectation_suite = self.create_expectation_suite(df, domain)
            ge_df = PandasDataset(df)
            
            if self.expectation_suite:
                # Run validation
                validation_result = ge_df.validate(expectation_suite=self.expectation_suite)
                
                # Process validation results
                for result in validation_result.results:
                    if not result.success:
                        violation = self._process_validation_result(result, df)
                        if violation:
                            violations.append(violation)
                            
                            # Add violating rows to quarantine
                            quarantine_rows.update(violation.violation_indices)
                            
                            # Apply auto-fixes if possible
                            if violation.auto_fixable:
                                fix_applied = self._apply_auto_fix(df, violation)
                                if fix_applied:
                                    auto_fixes_applied.append(f"Fixed {violation.expectation_type} in {violation.column}")
        else:
            # Fallback validations when Great Expectations is unavailable
            domain_rule = self.domain_rules.get(domain, {}) if domain else {}

            def register_violation(column: str, expectation_type: str, mask, reason: str, severity: str = 'warning', auto_fix: bool = True):
                indices = df.index[mask].tolist()
                if not indices:
                    return
                violation = QualityViolation(
                    column=column,
                    expectation_type=expectation_type,
                    violation_count=len(indices),
                    violation_indices=indices,
                    violation_values=df.loc[indices, column].tolist(),
                    reason=reason,
                    severity=severity,
                    auto_fixable=auto_fix
                )
                violations.append(violation)
                quarantine_rows.update(indices)
                if auto_fix and self._apply_auto_fix(df, violation):
                    auto_fixes_applied.append(f"Fixed {expectation_type} in {column}")

            # Numeric range checks
            age_range = domain_rule.get('age_range')
            if age_range and 'age' in df.columns:
                age_series = pd.to_numeric(df['age'], errors='coerce')
                invalid_age_mask = (age_series < age_range[0]) | (age_series > age_range[1]) | age_series.isna()
                register_violation('age', 'expect_column_values_to_be_between', invalid_age_mask, f"Values outside {age_range[0]}-{age_range[1]}")

            amount_range = domain_rule.get('amount_range')
            if amount_range and 'amount' in df.columns:
                amount_series = pd.to_numeric(df['amount'], errors='coerce')
                invalid_amount_mask = (amount_series < amount_range[0]) | (amount_series > amount_range[1]) | amount_series.isna()
                register_violation('amount', 'expect_column_values_to_be_between', invalid_amount_mask, f"Values outside {amount_range[0]}-{amount_range[1]}")

            price_range = domain_rule.get('price_range')
            if price_range and 'price' in df.columns:
                price_series = pd.to_numeric(df['price'], errors='coerce')
                invalid_price_mask = (price_series < price_range[0]) | (price_series > price_range[1]) | price_series.isna()
                register_violation('price', 'expect_column_values_to_be_between', invalid_price_mask, f"Values outside {price_range[0]}-{price_range[1]}")

            # Categorical set checks
            gender_values = domain_rule.get('gender_values')
            if gender_values and 'gender' in df.columns:
                normalized_gender = df['gender'].astype(str)
                invalid_gender_mask = ~normalized_gender.isin(gender_values)
                register_violation('gender', 'expect_column_values_to_be_in_set', invalid_gender_mask, f"Values outside {gender_values}")

            status_values = domain_rule.get('status_values')
            if status_values and 'status' in df.columns:
                normalized_status = df['status'].astype(str)
                invalid_status_mask = ~normalized_status.isin(status_values)
                register_violation('status', 'expect_column_values_to_be_in_set', invalid_status_mask, f"Values outside {status_values}")
        
        # Calculate quality score
        total_checks = len(df.columns) * 3  # Basic checks per column
        quality_score = max(0, (total_checks - len(violations)) / total_checks)
        
        self.quality_report = QualityReport(
            timestamp=datetime.now(),
            total_rows=len(df),
            total_columns=len(df.columns),
            violations=violations,
            quarantine_rows=list(quarantine_rows),
            auto_fixes_applied=auto_fixes_applied,
            quality_score=quality_score,
            domain_inferred=domain
        )
        
        return self.quality_report

    def _process_validation_result(self, result, df: pd.DataFrame) -> Optional[QualityViolation]:
        """Process a Great Expectations validation result into a QualityViolation."""
        try:
            expectation_type = result.expectation_config['expectation_type']
            column = result.expectation_config['kwargs'].get('column', 'unknown')
            
            # Extract violation details
            violation_count = result.result.get('unexpected_count', 0)
            violation_indices = result.result.get('unexpected_index_list', [])
            violation_values = result.result.get('unexpected_values', [])
            
            # Determine severity and auto-fixability
            severity = 'error'
            auto_fixable = False
            
            if expectation_type == 'expect_column_values_to_not_be_null':
                severity = 'error'
                auto_fixable = True
            elif expectation_type == 'expect_column_values_to_be_unique':
                severity = 'error'
                auto_fixable = False
            elif expectation_type == 'expect_column_values_to_be_between':
                severity = 'warning'
                auto_fixable = True
            elif expectation_type == 'expect_column_values_to_be_in_set':
                severity = 'warning'
                auto_fixable = True
            
            return QualityViolation(
                column=column,
                expectation_type=expectation_type,
                violation_count=violation_count,
                violation_indices=violation_indices,
                violation_values=violation_values,
                reason=f"Failed {expectation_type}",
                severity=severity,
                auto_fixable=auto_fixable
            )
        except Exception as e:
            logger.warning(f"Error processing validation result: {e}")
            return None

    def _apply_auto_fix(self, df: pd.DataFrame, violation: QualityViolation) -> bool:
        """Apply auto-fixes for common data issues."""
        try:
            # Prefer explicit expectation types
            if hasattr(violation, 'expectation_type'):
                if violation.expectation_type == "expect_column_values_to_be_between":
                    col = violation.column
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        # Apply domain-reasonable bounds for typical test data
                        if col == 'age':
                            df[col] = df[col].clip(lower=0, upper=150)
                            return True
                elif violation.expectation_type == "expect_column_values_to_be_in_set":
                    col = violation.column
                    if col in df.columns:
                        if col == 'gender':
                            valid_values = ['male', 'female', 'other']
                            df[col] = df[col].apply(lambda x: x if x in valid_values else 'other')
                            return True

            # Fallback simple rule-based fixes for common columns
            if hasattr(violation, 'column'):
                col = violation.column
                if col == 'age' and col in df.columns:
                    df[col] = df[col].clip(lower=0, upper=150)
                    return True
                elif col == 'gender' and col in df.columns:
                    valid_genders = ['male', 'female', 'other']
                    df[col] = df[col].apply(lambda x: x if str(x).lower() in valid_genders else 'other')
                    return True
            
            return False
        except Exception:
            return False

    def generate_dqr(self, output_file: str):
        """Generate Data Quality Report in Markdown format."""
        if not self.quality_report:
            logger.error("No quality report available. Run validate_data first.")
            return
        
        report = self.quality_report
        
        with open(output_file, 'w') as f:
            f.write("# Data Quality Report\n\n")
            f.write(f"**Generated:** {report.timestamp.isoformat()}\n\n")
            f.write(f"**Dataset Shape:** {report.total_rows} rows × {report.total_columns} columns\n\n")
            f.write(f"**Quality Score:** {report.quality_score:.2%}\n\n")
            
            if report.domain_inferred:
                f.write(f"**Domain:** {report.domain_inferred}\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- **Total Violations:** {len(report.violations)}\n")
            f.write(f"- **Quarantined Rows:** {len(report.quarantine_rows)}\n")
            f.write(f"- **Auto-fixes Applied:** {len(report.auto_fixes_applied)}\n\n")
            
            # Violations by severity
            error_count = sum(1 for v in report.violations if v.severity == 'error')
            warning_count = sum(1 for v in report.violations if v.severity == 'warning')
            info_count = sum(1 for v in report.violations if v.severity == 'info')
            
            f.write("## Violations by Severity\n\n")
            f.write(f"- **Errors:** {error_count}\n")
            f.write(f"- **Warnings:** {warning_count}\n")
            f.write(f"- **Info:** {info_count}\n\n")
            
            # Detailed violations
            if report.violations:
                f.write("## Detailed Violations\n\n")
                for i, violation in enumerate(report.violations, 1):
                    f.write(f"### {i}. {violation.column} - {violation.expectation_type}\n\n")
                    f.write(f"- **Severity:** {violation.severity}\n")
                    f.write(f"- **Count:** {violation.violation_count}\n")
                    f.write(f"- **Auto-fixable:** {'Yes' if violation.auto_fixable else 'No'}\n")
                    f.write(f"- **Reason:** {violation.reason}\n\n")
                    
                    if violation.violation_values:
                        f.write(f"- **Sample Values:** {violation.violation_values[:5]}\n\n")
            
            # Auto-fixes applied
            if report.auto_fixes_applied:
                f.write("## Auto-fixes Applied\n\n")
                for fix in report.auto_fixes_applied:
                    f.write(f"- {fix}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if error_count > 0:
                f.write("- Review and fix all error-level violations\n")
            if warning_count > 0:
                f.write("- Consider addressing warning-level violations\n")
            if len(report.quarantine_rows) > 0:
                f.write(f"- Review {len(report.quarantine_rows)} quarantined rows\n")
            if report.quality_score < 0.8:
                f.write("- Overall data quality is below 80%, consider data cleaning\n")

    def export_quarantine_data(self, df: pd.DataFrame, output_file: str):
        """Export quarantined rows to a separate file."""
        if not self.quality_report or not self.quality_report.quarantine_rows:
            logger.info("No quarantined rows to export")
            return
        
        quarantine_df = df.iloc[self.quality_report.quarantine_rows].copy()
        quarantine_df.to_parquet(output_file, index=False)
        logger.info(f"Exported {len(quarantine_df)} quarantined rows to {output_file}")

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics as a dictionary."""
        if not self.quality_report:
            return {}
        
        return {
            'quality_score': self.quality_report.quality_score,
            'total_violations': len(self.quality_report.violations),
            'error_count': sum(1 for v in self.quality_report.violations if v.severity == 'error'),
            'warning_count': sum(1 for v in self.quality_report.violations if v.severity == 'warning'),
            'quarantine_count': len(self.quality_report.quarantine_rows),
            'auto_fixes_count': len(self.quality_report.auto_fixes_applied),
            'domain': self.quality_report.domain_inferred
        }
