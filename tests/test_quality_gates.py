"""
Unit tests for QualityGates class.
"""

import unittest
import pandas as pd
import numpy as np
from core.quality_gates import QualityGates, QualityViolation
try:
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration
    GE_AVAILABLE = True
except Exception:
    GE_AVAILABLE = False

class TestQualityGates(unittest.TestCase):
    """Test cases for QualityGates."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quality_gates = QualityGates()
        # Ensure Great Expectations is properly configured for tests
        if GE_AVAILABLE:
            self.quality_gates.expectation_suite = ExpectationSuite(
                expectation_suite_name="test_suite"
            )
            # Age between 0 and 150
            self.quality_gates.expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "age",
                        "min_value": 0,
                        "max_value": 150
                    }
                )
            )
            # Gender in set
            self.quality_gates.expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={
                        "column": "gender",
                        "value_set": ["male", "female", "other"]
                    }
                )
            )
    
    def test_healthcare_domain_validation(self):
        """Test healthcare domain validation."""
        data = {
            'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'age': [25, 30, 45, 60, 35],
            'gender': ['male', 'female', 'male', 'female', 'other'],
            'email': ['john@email.com', 'jane@email.com', 'bob@company.org', 'alice@test.net', 'charlie@example.com']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report.total_rows, 5)
        self.assertEqual(quality_report.total_columns, 4)
        self.assertGreaterEqual(quality_report.quality_score, 0)
        self.assertLessEqual(quality_report.quality_score, 1)
    
    def test_finance_domain_validation(self):
        """Test finance domain validation."""
        data = {
            'account_id': ['A001', 'A002', 'A003', 'A004', 'A005'],
            'amount': [100.50, 250.75, 500.00, 750.25, 1000.00],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='finance')
        
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report.total_rows, 5)
        self.assertEqual(quality_report.total_columns, 3)
    
    def test_retail_domain_validation(self):
        """Test retail domain validation."""
        data = {
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'price': [10.99, 25.50, 50.00, 75.25, 100.00],
            'category': ['electronics', 'clothing', 'home', 'books', 'sports']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='retail')
        
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report.total_rows, 5)
        self.assertEqual(quality_report.total_columns, 3)
    
    def test_violation_detection(self):
        """Test violation detection."""
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'age': [25, 30, 45, 60, 200],  # Invalid age
            'gender': ['male', 'female', 'male', 'female', 'invalid'],  # Invalid gender
            'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        # Should detect violations
        self.assertGreater(len(quality_report.violations), 0)
        
        # Check violation types
        violation_types = [v.expectation_type for v in quality_report.violations]
        self.assertIn('expect_column_values_to_be_between', violation_types)
        self.assertIn('expect_column_values_to_be_in_set', violation_types)
    
    def test_auto_fix_application(self):
        """Test auto-fix application."""
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'age': [25, 30, 45, 60, 200],  # Invalid age
            'gender': ['male', 'female', 'male', 'female', 'invalid'],  # Invalid gender
            'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        # Should have auto-fixes applied
        self.assertGreater(len(quality_report.auto_fixes_applied), 0)
        
        # Check auto-fix types
        fix_types = [fix for fix in quality_report.auto_fixes_applied]
        self.assertTrue(any('age' in fix for fix in fix_types))
        self.assertTrue(any('gender' in fix for fix in fix_types))
    
    def test_quarantine_functionality(self):
        """Test quarantine functionality."""
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'age': [25, 30, 45, 60, 200],  # Invalid age
            'gender': ['male', 'female', 'male', 'female', 'invalid'],  # Invalid gender
            'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        # Should have quarantined rows
        self.assertGreater(len(quality_report.quarantine_rows), 0)
        
        # Check quarantine reasons
        for violation in quality_report.violations:
            if violation.quarantine_reason:
                self.assertIsInstance(violation.quarantine_reason, str)
    
    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'age': [25, 30, 45, 60, 35],
            'gender': ['male', 'female', 'male', 'female', 'other'],
            'email': ['john@email.com', 'jane@email.com', 'bob@company.org', 'alice@test.net', 'charlie@example.com']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        metrics = self.quality_gates.get_quality_metrics()
        
        self.assertIn('quality_score', metrics)
        self.assertIn('total_violations', metrics)
        self.assertIn('error_count', metrics)
        self.assertIn('warning_count', metrics)
        self.assertIn('quarantine_count', metrics)
        self.assertIn('auto_fixes_count', metrics)
        self.assertIn('domain', metrics)
        
        self.assertEqual(metrics['domain'], 'healthcare')
        self.assertGreaterEqual(metrics['quality_score'], 0)
        self.assertLessEqual(metrics['quality_score'], 1)
    
    def test_dqr_generation(self):
        """Test Data Quality Report generation."""
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'age': [25, 30, 45, 60, 35],
            'gender': ['male', 'female', 'male', 'female', 'other'],
            'email': ['john@email.com', 'jane@email.com', 'bob@company.org', 'alice@test.net', 'charlie@example.com']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        # Test DQR generation
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            self.quality_gates.generate_dqr(f.name)
            
            # Check that file was created and contains content
            with open(f.name, 'r') as f:
                content = f.read()
            
            self.assertIn('Data Quality Report', content)
            self.assertIn('Summary', content)
            self.assertIn('Recommendations', content)
    
    def test_quarantine_data_export(self):
        """Test quarantine data export."""
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'age': [25, 30, 45, 60, 200],  # Invalid age
            'gender': ['male', 'female', 'male', 'female', 'invalid'],  # Invalid gender
            'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        # Test quarantine data export
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
            self.quality_gates.export_quarantine_data(df, f.name)
            
            # Check that file was created
            import os
            self.assertTrue(os.path.exists(f.name))
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report.total_rows, 0)
        self.assertEqual(quality_report.total_columns, 0)
        self.assertEqual(len(quality_report.violations), 0)
        self.assertEqual(len(quality_report.quarantine_rows), 0)
    
    def test_single_row_dataframe(self):
        """Test handling of single row dataframe."""
        data = {
            'id': ['ID001'],
            'age': [25],
            'gender': ['male'],
            'email': ['john@email.com']
        }
        df = pd.DataFrame(data)
        
        quality_report = self.quality_gates.validate_data(df, domain='healthcare')
        
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report.total_rows, 1)
        self.assertEqual(quality_report.total_columns, 4)
        self.assertGreaterEqual(quality_report.quality_score, 0)
        self.assertLessEqual(quality_report.quality_score, 1)

if __name__ == '__main__':
    unittest.main()
