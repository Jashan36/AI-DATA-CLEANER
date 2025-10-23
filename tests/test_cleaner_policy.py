"""
Unit tests for CleanerPolicy class.
"""

import unittest
import pandas as pd
import numpy as np
from core.cleaner_policy import CleanerPolicy, ColumnType

class TestCleanerPolicy(unittest.TestCase):
    """Test cases for CleanerPolicy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy = CleanerPolicy()
    
    def test_column_type_detection_id(self):
        """Test ID column detection."""
        series = pd.Series(['ID001', 'ID002', 'ID003', 'ID004', 'ID005'])
        column_type = self.policy.detect_column_type(series, 'patient_id')
        self.assertEqual(column_type, ColumnType.ID_HASH)
    
    def test_column_type_detection_email(self):
        """Test email column detection."""
        series = pd.Series(['john@email.com', 'jane@email.com', 'bob@company.org'])
        column_type = self.policy.detect_column_type(series, 'email')
        self.assertEqual(column_type, ColumnType.EMAIL)
    
    def test_column_type_detection_phone(self):
        """Test phone column detection."""
        series = pd.Series(['+1-555-123-4567', '(555) 987-6543', '555.111.2222'])
        column_type = self.policy.detect_column_type(series, 'phone')
        self.assertEqual(column_type, ColumnType.PHONE)
    
    def test_column_type_detection_categorical_low(self):
        """Test low cardinality categorical detection."""
        series = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
        column_type = self.policy.detect_column_type(series, 'category')
        self.assertEqual(column_type, ColumnType.CATEGORICAL_LOW)
    
    def test_column_type_detection_numeric(self):
        """Test numeric column detection."""
        series = pd.Series([1, 2, 3, 4, 5])
        column_type = self.policy.detect_column_type(series, 'age')
        self.assertEqual(column_type, ColumnType.NUMERIC_MEASURE)
    
    def test_column_type_detection_boolean(self):
        """Test boolean column detection."""
        series = pd.Series([True, False, True, False])
        column_type = self.policy.detect_column_type(series, 'active')
        self.assertEqual(column_type, ColumnType.BOOLEAN)
    
    def test_cleaning_dataframe(self):
        """Test complete dataframe cleaning."""
        data = {
            'id': ['ID001', 'ID002', 'ID003'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'age': [25, 30, 45],
            'email': ['john@email.com', 'jane@email.com', 'invalid-email'],
            'status': ['Delivered', 'DELIVERED', 'delivered']
        }
        df = pd.DataFrame(data)
        
        cleaned_df, quarantined_df, audit_log = self.policy.clean_dataframe(df)
        
        self.assertEqual(len(cleaned_df), 3)
        self.assertEqual(len(audit_log.decisions), 5)  # One decision per column
        self.assertIsInstance(quarantined_df, pd.DataFrame)
    
    def test_synonym_canonicalization(self):
        """Test categorical synonym canonicalization."""
        series = pd.Series(['Delivered', 'DELIVERED', 'delivered', 'Completed', 'Shipped'])
        canonicalized = self.policy._canonicalize_categorical(series, 'status')
        
        # Check that synonyms are merged
        unique_values = canonicalized.unique()
        self.assertIn('delivered', unique_values)
    
    def test_text_normalization(self):
        """Test text normalization."""
        series = pd.Series(['  John Doe  ', 'JANE SMITH', 'Bob Johnson'])
        normalized = self.policy._normalize_text(series)
        
        # Check that text is normalized
        self.assertTrue(all(text.islower() for text in normalized if pd.notna(text)))
        self.assertTrue(all(text.strip() == text for text in normalized if pd.notna(text)))
    
    def test_email_validation(self):
        """Test email validation."""
        series = pd.Series(['john@email.com', 'invalid-email', 'jane@company.org'])
        validated, quarantined = self.policy._validate_emails(series)
        
        # Check that valid emails are preserved
        self.assertEqual(validated.iloc[0], 'john@email.com')
        self.assertEqual(validated.iloc[2], 'jane@company.org')
        
        # Check that invalid emails are quarantined
        self.assertTrue(quarantined.iloc[1])
    
    def test_outlier_handling(self):
        """Test outlier handling."""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        handled, quarantined = self.policy._handle_outliers(series)
        
        # Check that outliers are capped
        self.assertTrue(handled.max() < 100)
        self.assertFalse(quarantined.any())  # No rows quarantined for outliers
    
    def test_audit_log_export(self):
        """Test audit log export."""
        data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
        df = pd.DataFrame(data)
        
        cleaned_df, quarantined_df, audit_log = self.policy.clean_dataframe(df)
        
        # Test export
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            audit_log.export_audit_log(f.name)
            
            # Check that file was created and contains data
            import json
            with open(f.name, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn('timestamp', exported_data)
            self.assertIn('decisions', exported_data)
            self.assertEqual(len(exported_data['decisions']), 2)

if __name__ == '__main__':
    unittest.main()
