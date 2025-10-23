#!/usr/bin/env python3
"""
Example usage of the Enhanced AI Data Cleaner
Demonstrates the Senior Data Scientist capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import the enhanced cleaner
from data3 import JAXDataCleaner

def create_sample_messy_dataset():
    """Create a sample messy dataset with common data quality issues"""
    
    # Create sample data with various issues
    data = {
        'Customer ID': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010'] * 10,
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson', 
                'Diana Prince', 'Eve Adams', 'Frank Miller', 'Grace Lee', 'Henry Davis'] * 10,
        'Email': ['john@email.com', 'jane@email.com', 'invalid-email', 'alice@company.org', 
                 'charlie@test.net', 'diana@invalid', 'eve@email.com', 'frank@company.org', 
                 'grace@test.net', 'henry@email.com'] * 10,
        'Phone': ['+1-555-123-4567', '(555) 987-6543', '555.111.2222', '555-333-4444', 
                 'invalid-phone', '+1-555-555-5555', '(555) 123-4567', '555.999.8888', 
                 '555-777-6666', '+1-555-444-3333'] * 10,
        'Age': [25, 30, 'invalid', 35, 28, 45, 'N/A', 22, 55, 33] * 10,
        'Status': ['Delivered', 'DELIVERED', 'Pending', 'PENDING', 'Cancelled', 
                  'CANCELLED', 'Returned', 'RETURNED', 'Processing', 'PROCESSING'] * 10,
        'Order Date': ['2024-01-15', '15/01/2024', 'Jan 15, 2024', '2024-01-15', '15-01-2024',
                      '2024/01/15', 'January 15, 2024', '15.01.2024', '2024-01-15', '15 01 2024'] * 10,
        'Price': ['$25.99', '30.50', 'invalid', '45.00', 'N/A', '$12.99', '20.00', 
                 'invalid', '$35.50', '15.99'] * 10,
        'Rating': ['Excellent', 'Good', 'Average', 'Poor', 'N/A', 'Excellent', 'Good', 
                  'Average', 'Poor', 'Excellent'] * 10,
        'Category': ['Electronics', 'ELECTRONICS', 'Clothing', 'CLOTHING', 'Books', 
                    'BOOKS', 'Home', 'HOME', 'Sports', 'SPORTS'] * 10
    }
    
    return pd.DataFrame(data)

def demonstrate_senior_data_scientist_workflow():
    """Demonstrate the complete Senior Data Scientist workflow"""
    
    print("🔧 Senior Data Scientist & Data Engineer Workflow Demo")
    print("=" * 60)
    
    # Step 1: Load messy data
    print("\n📊 Step 1: Loading messy dataset...")
    df = create_sample_messy_dataset()
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Step 2: Initialize enhanced cleaner
    print("\n🔧 Step 2: Initializing Enhanced JAX Data Cleaner...")
    cleaner = JAXDataCleaner()
    cleaner.set_parameters(
        outlier_threshold=3.0,
        imputation_method="mean",
        categorical_missing_strategy="unknown"
    )
    
    # Step 3: Apply comprehensive cleaning
    print("\n🧹 Step 3: Applying comprehensive data cleaning pipeline...")
    cleaned_df, stats = cleaner.preprocess_data(df, outlier_strategy="cap")
    
    # Step 4: Display results
    print("\n✅ Step 4: Cleaning Results")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
    print(f"Detected domain: {stats.get('domain', 'Unknown')}")
    print(f"Outliers handled: {stats.get('outlier_count', 0)}")
    
    # Step 5: Show cleaning log
    print("\n📋 Step 5: Comprehensive Cleaning Log")
    cleaning_log = cleaner.get_cleaning_log()
    
    print(f"Total transformation steps: {len(cleaning_log['steps'])}")
    print(f"Warnings: {len(cleaning_log['warnings'])}")
    print(f"Errors: {len(cleaning_log['errors'])}")
    
    # Display key transformations
    print("\n🔄 Key Transformations Applied:")
    for i, step in enumerate(cleaning_log['steps'][:5], 1):  # Show first 5 steps
        print(f"  {i}. {step['operation']} on {step['column']}: {step.get('details', 'N/A')}")
    
    # Step 6: Export results
    print("\n💾 Step 6: Exporting Results")
    
    # Export cleaned dataset
    cleaned_df.to_csv('cleaned_dataset.csv', index=False)
    print("✅ Cleaned dataset exported to: cleaned_dataset.csv")
    
    # Export cleaning log
    with open('cleaning_log.json', 'w') as f:
        json.dump(cleaning_log, f, indent=2, default=str)
    print("✅ Cleaning log exported to: cleaning_log.json")
    
    # Export categorical mappings
    mappings = {}
    for col, info in cleaning_log['transformations'].items():
        if info.get('type') == 'categorical_encoding':
            mappings[col] = info['mapping']
    
    with open('categorical_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    print("✅ Categorical mappings exported to: categorical_mappings.json")
    
    # Step 7: Show before/after comparison
    print("\n📊 Step 7: Before/After Comparison")
    
    # Show original vs cleaned for key columns
    comparison_cols = ['Status', 'Category', 'Email', 'Phone']
    for col in comparison_cols:
        if col in df.columns:
            print(f"\n{col} Column:")
            print(f"  Original unique values: {df[col].nunique()}")
            if col.lower() in cleaned_df.columns:
                print(f"  Cleaned unique values: {cleaned_df[col.lower()].nunique()}")
            
            # Show value counts for original
            print(f"  Original value counts:")
            for val, count in df[col].value_counts().head(3).items():
                print(f"    {val}: {count}")
    
    print("\n🎯 Senior Data Scientist Workflow Complete!")
    print("=" * 60)
    
    return cleaned_df, cleaning_log

def demonstrate_domain_specific_cleaning():
    """Demonstrate domain-specific cleaning capabilities"""
    
    print("\n🏥 Domain-Specific Cleaning Demo")
    print("=" * 40)
    
    # Healthcare dataset example
    healthcare_data = {
        'Patient ID': ['P001', 'P002', 'P003', 'P004', 'P005'] * 20,
        'Age': [25, 30, 45, 60, 35] * 20,
        'Gender': ['Male', 'FEMALE', 'M', 'Female', 'Other'] * 20,
        'Symptoms': ['Fever', 'FEVER', 'Cough', 'COUGH', 'Headache'] * 20,
        'Diagnosis': ['Hypertension', 'HYPERTENSION', 'Diabetes', 'DIABETES', 'Asthma'] * 20,
        'Medication': ['Aspirin', 'ASPIRIN', 'Ibuprofen', 'IBUPROFEN', 'Acetaminophen'] * 20,
        'Visit Date': ['2024-01-15', '15/01/2024', 'Jan 15, 2024', '2024-01-15', '15-01-2024'] * 20
    }
    
    df_healthcare = pd.DataFrame(healthcare_data)
    print(f"Healthcare dataset shape: {df_healthcare.shape}")
    
    # Apply cleaning
    cleaner = JAXDataCleaner()
    cleaned_healthcare, stats = cleaner.preprocess_data(df_healthcare)
    
    print(f"Detected domain: {stats.get('domain', 'Unknown')}")
    print(f"Cleaning steps applied: {len(cleaner.get_cleaning_log()['steps'])}")
    
    return cleaned_healthcare

if __name__ == "__main__":
    # Run the complete demonstration
    cleaned_df, cleaning_log = demonstrate_senior_data_scientist_workflow()
    
    # Run domain-specific demo
    cleaned_healthcare = demonstrate_domain_specific_cleaning()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("\nFiles generated:")
    print("- cleaned_dataset.csv: ML-ready cleaned dataset")
    print("- cleaning_log.json: Complete audit trail")
    print("- categorical_mappings.json: Encoding mappings")
