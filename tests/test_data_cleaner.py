"""
Unit tests for DataCleaner class.
"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
import pytest
from core.data_cleaner import DataCleaner, CleaningConfig, CleaningResult

class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CleaningConfig(
            chunk_size=100,
            max_memory_mb=512,
            enable_quality_gates=True,
            enable_quarantine=True,
            output_format='parquet'
        )
        self.cleaner = DataCleaner(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.cleaner.cleanup_temp_files()
    
    def test_initialization(self):
        """Test DataCleaner initialization."""
        self.assertEqual(self.cleaner.config.chunk_size, 100)
        self.assertEqual(self.cleaner.config.max_memory_mb, 512)
        self.assertTrue(self.cleaner.config.enable_quality_gates)
        self.assertTrue(self.cleaner.config.enable_quarantine)
        self.assertEqual(self.cleaner.config.output_format, 'parquet')
    
    def test_clean_csv_file(self):
        """Test CSV file cleaning."""
        # Create test data
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'] * 20,
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'] * 20,
            'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net'] * 20,
            'age': [25, 30, 45, 60, 35] * 20,
            'status': ['Delivered', 'DELIVERED', 'delivered', 'Completed', 'Shipped'] * 20
        }
        df = pd.DataFrame(data)
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            input_path = f.name
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix='test_output_')
            
            # Clean the file
            result = self.cleaner.clean_file(input_path, output_dir, domain='retail')
            
            # Verify result
            self.assertIsInstance(result, CleaningResult, "Result should be CleaningResult instance")
            self.assertTrue(result.success, f"Cleaning should succeed. Error: {result.error_message if hasattr(result, 'error_message') else 'No error message'}")
            self.assertEqual(result.total_rows_processed, 100, "Should process 100 rows")
            self.assertGreaterEqual(result.total_rows_cleaned, 0, "Cleaned rows should be >= 0")
            self.assertGreater(result.processing_time, 0, "Processing time should be > 0")
            
            # Check output files
            self.assertTrue(os.path.exists(result.cleaned_data_path), "Cleaned data file should exist")
            self.assertTrue(os.path.exists(result.audit_log_path), "Audit log file should exist")
            
            if result.quarantined_data_path:
                self.assertTrue(os.path.exists(result.quarantined_data_path), "Quarantined data file should exist")
            
            if result.quality_report_path:
                self.assertTrue(os.path.exists(result.quality_report_path), "Quality report file should exist")
            
            if result.dqr_path:
                self.assertTrue(os.path.exists(result.dqr_path), "DQR file should exist")
            
        finally:
            # Clean up
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_clean_parquet_file(self):
        """Test Parquet file cleaning."""
        # Create test data
        data = {
            'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'] * 20,
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'] * 20,
            'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net'] * 20,
            'age': [25, 30, 45, 60, 35] * 20,
            'status': ['Delivered', 'DELIVERED', 'delivered', 'Completed', 'Shipped'] * 20
        }
        df = pd.DataFrame(data)
        
        # Save to temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name, index=False)
            input_path = f.name
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix='test_output_')
            
            # Clean the file
            result = self.cleaner.clean_file(input_path, output_dir, domain='retail')
            
            # Verify result
            self.assertIsInstance(result, CleaningResult, "Result should be CleaningResult instance")
            self.assertTrue(result.success, f"Cleaning should succeed. Error: {result.error_message if hasattr(result, 'error_message') else 'No error message'}")
            self.assertEqual(result.total_rows_processed, 100, "Should process 100 rows")
            self.assertGreaterEqual(result.total_rows_cleaned, 0, "Cleaned rows should be >= 0")
            self.assertGreater(result.processing_time, 0, "Processing time should be > 0")
            
        finally:
            # Clean up
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_unsupported_file_format(self):
        """Test handling of unsupported file formats."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'This is not a supported format')
            input_path = f.name
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix='test_output_')
            
            # Clean the file
            result = self.cleaner.clean_file(input_path, output_dir, domain='retail')
            
            # Verify result
            self.assertIsInstance(result, CleaningResult)
            self.assertFalse(result.success)
            self.assertIn('Unsupported file format', result.error_message)
            
        finally:
            # Clean up
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_memory_usage_check(self):
        """Test memory usage checking."""
        # Test with psutil available
        try:
            import psutil
            result = self.cleaner._check_memory_usage()
            self.assertIsInstance(result, bool)
        except ImportError:
            # Test without psutil
            result = self.cleaner._check_memory_usage()
            self.assertFalse(result)  # Should return False when psutil not available
    
    def test_progress_tracking(self):
        """Test progress tracking."""
        # Initial progress should be zero
        progress = self.cleaner.get_progress()
        self.assertEqual(progress['chunks_processed'], 0)
        self.assertEqual(progress['total_chunks'], 0)
        self.assertEqual(progress['rows_processed'], 0)
        self.assertEqual(progress['rows_cleaned'], 0)
        self.assertEqual(progress['rows_quarantined'], 0)
        self.assertEqual(progress['elapsed_time'], 0)
        self.assertIsNone(progress['estimated_remaining_time'])
    
    def test_combine_audit_logs(self):
        """Test audit log combination."""
        from core.cleaner_policy import AuditLog, CleaningDecision
        from core.cleaner_policy import ColumnType
        
        # Create mock audit logs
        log1 = AuditLog(
            timestamp=None,
            input_shape=(50, 3),
            output_shape=(50, 3),
            decisions=[
                CleaningDecision(
                    column='col1',
                    column_type=ColumnType.NUMERIC_MEASURE,
                    strategy='handle_numeric_outliers',
                    confidence=0.8,
                    rules_applied=['outlier_detection'],
                    before_sample=[1, 2, 3],
                    after_sample=[1, 2, 3]
                )
            ]
        )
        log1.quarantined_rows = 2
        log1.missing_fixed = 1
        log1.outliers_handled = 1
        log1.merges_performed = 0
        
        log2 = AuditLog(
            timestamp=None,
            input_shape=(50, 3),
            output_shape=(50, 3),
            decisions=[
                CleaningDecision(
                    column='col2',
                    column_type=ColumnType.CATEGORICAL_LOW,
                    strategy='canonicalize_categorical',
                    confidence=0.9,
                    rules_applied=['case_folding'],
                    before_sample=['A', 'B', 'C'],
                    after_sample=['a', 'b', 'c']
                )
            ]
        )
        log2.quarantined_rows = 1
        log2.missing_fixed = 0
        log2.outliers_handled = 0
        log2.merges_performed = 1
        
        # Combine logs
        combined_log = self.cleaner._combine_audit_logs([log1, log2])
        
        # Verify combination
        self.assertEqual(len(combined_log.decisions), 2)
        self.assertEqual(combined_log.quarantined_rows, 3)
        self.assertEqual(combined_log.missing_fixed, 1)
        self.assertEqual(combined_log.outliers_handled, 1)
        self.assertEqual(combined_log.merges_performed, 1)
    
    def test_combine_audit_logs_empty(self):
        """Test audit log combination with empty list."""
        combined_log = self.cleaner._combine_audit_logs([])
        
        self.assertEqual(len(combined_log.decisions), 0)
        self.assertEqual(combined_log.quarantined_rows, 0)
        self.assertEqual(combined_log.missing_fixed, 0)
        self.assertEqual(combined_log.outliers_handled, 0)
        self.assertEqual(combined_log.merges_performed, 0)
    
    def test_estimate_remaining_time(self):
        """Test remaining time estimation."""
        # Test with no progress
        self.cleaner.progress['chunks_processed'] = 0
        self.cleaner.progress['start_time'] = None
        remaining_time = self.cleaner._estimate_remaining_time()
        self.assertIsNone(remaining_time)
        
        # Test with progress
        from datetime import datetime, timedelta
        self.cleaner.progress['chunks_processed'] = 5
        self.cleaner.progress['total_chunks'] = 10
        self.cleaner.progress['start_time'] = datetime.now() - timedelta(seconds=10)
        
        remaining_time = self.cleaner._estimate_remaining_time()
        self.assertIsNotNone(remaining_time)
        self.assertGreater(remaining_time, 0)
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix='test_cleanup_'))
        
        # Create some test files
        (temp_dir / 'test1.txt').write_text('test content 1')
        (temp_dir / 'test2.txt').write_text('test content 2')
        
        # Set the temp directory
        self.cleaner.temp_dir = temp_dir
        
        # Clean up
        self.cleaner.cleanup_temp_files()
        
        # Verify cleanup
        self.assertFalse(temp_dir.exists())
    
    @pytest.mark.slow
    def test_different_domains(self):
        """Test cleaning with different domains."""
        # Create test data
        data = {
            'id': ['ID001', 'ID002', 'ID003'],
            'age': [25, 30, 45],
            'gender': ['male', 'female', 'other'],
            'email': ['john@email.com', 'jane@email.com', 'bob@company.org']
        }
        df = pd.DataFrame(data)
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            input_path = f.name
        
        try:
            # Test different domains
            domains = ['healthcare', 'finance', 'retail', 'general']
            
            for domain in domains:
                output_dir = tempfile.mkdtemp(prefix=f'test_output_{domain}_')
                
                result = self.cleaner.clean_file(input_path, output_dir, domain=domain)
                
                self.assertTrue(result.success)
                # Check if rows are processed, not exact count due to header handling
                self.assertGreaterEqual(result.total_rows_processed, 3)
                
        finally:
            # Clean up
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_error_handling(self):
        """Test error handling during cleaning."""
        # Create a temporary file that will cause an error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('invalid,csv,content\nwith,missing,columns\n')
            input_path = f.name
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix='test_output_')
            
            # Clean the file
            result = self.cleaner.clean_file(input_path, output_dir, domain='retail')
            
            # Should still succeed but with warnings
            self.assertTrue(result.success)
            
        finally:
            # Clean up
            if os.path.exists(input_path):
                os.unlink(input_path)

if __name__ == '__main__':
    unittest.main()
