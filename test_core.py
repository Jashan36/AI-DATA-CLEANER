"""
Basic tests for core functionality of AI Data Cleaner & Web Scraper.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import asyncio

from core.cleaner_policy import CleanerPolicy, ColumnType
from core.data_cleaner import DataCleaner, CleaningConfig
from core.quality_gates import QualityGates
from core.web_scraper import WebScraper

def test_cleaner_policy():
    """Test the CleanerPolicy class."""
    print("Testing CleanerPolicy...")
    
    # Create sample data
    data = {
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'age': [25, 30, 45, 60, 35],
        'gender': ['Male', 'FEMALE', 'M', 'Female', 'Other'],
        'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net'],
        'phone': ['+1-555-123-4567', '(555) 987-6543', '555.111.2222', '555-333-4444', 'invalid-phone'],
        'status': ['Delivered', 'DELIVERED', 'delivered', 'Completed', 'Shipped']
    }
    
    df = pd.DataFrame(data)
    
    # Initialize policy
    policy = CleanerPolicy()
    
    # Test column type detection
    for column in df.columns:
        column_type = policy.detect_column_type(df[column], column)
        print(f"Column '{column}' detected as: {column_type}")
    
    # Test cleaning
    cleaned_df, quarantined_df, audit_log = policy.clean_dataframe(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Quarantined shape: {quarantined_df.shape}")
    print(f"Audit log decisions: {len(audit_log.decisions)}")
    
    # Display results
    print("\nCleaned data:")
    print(cleaned_df.head())
    
    if len(quarantined_df) > 0:
        print("\nQuarantined data:")
        print(quarantined_df.head())
    
    print("✅ CleanerPolicy test passed\n")

def test_quality_gates():
    """Test the QualityGates class."""
    print("Testing QualityGates...")
    
    # Create sample data with quality issues
    data = {
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
        'age': [25, 30, 45, 60, 35, 200],  # Invalid age
        'gender': ['male', 'female', 'male', 'female', 'other', 'invalid'],  # Invalid gender
        'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net', 'bad-email'],
        'phone': ['+1-555-123-4567', '(555) 987-6543', '555.111.2222', '555-333-4444', 'invalid-phone', 'bad-phone']
    }
    
    df = pd.DataFrame(data)
    
    # Initialize quality gates
    quality_gates = QualityGates()
    
    # Test validation
    quality_report = quality_gates.validate_data(df, domain='healthcare')
    
    print(f"Quality score: {quality_report.quality_score:.2%}")
    print(f"Total violations: {len(quality_report.violations)}")
    print(f"Quarantined rows: {len(quality_report.quarantine_rows)}")
    print(f"Auto-fixes applied: {len(quality_report.auto_fixes_applied)}")
    
    # Display violations
    for violation in quality_report.violations:
        print(f"  - {violation.column}: {violation.expectation_type} ({violation.severity})")
    
    print("✅ QualityGates test passed\n")

def test_data_cleaner():
    """Test the DataCleaner class."""
    print("Testing DataCleaner...")
    
    # Create sample data
    data = {
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'] * 100,
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'] * 100,
        'email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 'charlie@test.net'] * 100,
        'phone': ['+1-555-123-4567', '(555) 987-6543', '555.111.2222', '555-333-4444', 'invalid-phone'] * 100,
        'age': [25, 30, 45, 60, 35] * 100,
        'status': ['Delivered', 'DELIVERED', 'delivered', 'Completed', 'Shipped'] * 100
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        input_path = f.name
    
    try:
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix='test_output_')
        
        # Initialize data cleaner
        config = CleaningConfig(
            chunk_size=100,
            max_memory_mb=512,
            enable_quality_gates=True,
            enable_quarantine=True,
            output_format='parquet'
        )
        
        cleaner = DataCleaner(config)
        
        # Clean the data
        result = cleaner.clean_file(input_path, output_dir, domain='retail')
        
        print(f"Success: {result.success}")
        print(f"Rows processed: {result.total_rows_processed}")
        print(f"Rows cleaned: {result.total_rows_cleaned}")
        print(f"Rows quarantined: {result.total_rows_quarantined}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.success:
            print(f"Cleaned data: {result.cleaned_data_path}")
            if result.quarantined_data_path:
                print(f"Quarantined data: {result.quarantined_data_path}")
            print(f"Audit log: {result.audit_log_path}")
            if result.quality_report_path:
                print(f"Quality report: {result.quality_report_path}")
            if result.dqr_path:
                print(f"DQR: {result.dqr_path}")
        
        # Clean up
        cleaner.cleanup_temp_files()
        
    finally:
        # Clean up input file
        os.unlink(input_path)
    
    print("✅ DataCleaner test passed\n")

async def test_web_scraper():
    """Test the WebScraper class."""
    print("Testing WebScraper...")
    
    # Test URLs
    urls = [
        'https://httpbin.org/html',
        'https://httpbin.org/json',
        'https://example.com'
    ]
    
    # Initialize web scraper
    scraper = WebScraper(
        max_concurrent=2,
        timeout=10,
        retry_attempts=2,
        respect_robots=False  # Disable for testing
    )
    
    # Scrape URLs
    results = await scraper.scrape_urls(urls)
    
    print(f"Scraped {len(results)} URLs")
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Display results
    for result in results:
        print(f"  - {result.url}: {'✅' if result.success else '❌'}")
        if result.success:
            print(f"    Title: {result.title}")
            print(f"    Response time: {result.response_time:.2f}s")
            print(f"    Cards found: {len(result.structured_cards)}")
        else:
            print(f"    Error: {result.error_message}")
    
    print("✅ WebScraper test passed\n")

def main():
    """Run all tests."""
    print("🧪 Running AI Data Cleaner & Web Scraper Tests\n")
    
    try:
        # Test core components
        test_cleaner_policy()
        test_quality_gates()
        test_data_cleaner()
        
        # Test web scraper
        asyncio.run(test_web_scraper())
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
