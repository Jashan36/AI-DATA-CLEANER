# Data Cleaning Policies

This document describes how the AI Data Cleaner makes automatic decisions about data cleaning strategies.

## Column Type Detection

The system automatically detects column types based on content analysis and column names:

### ID/Hash Columns
- **Detection**: Column names containing 'id', 'key', 'hash', 'uuid'
- **Criteria**: High uniqueness (>95% unique values)
- **Strategy**: Case normalization and whitespace trimming only

### Categorical Columns
- **Low Cardinality**: <50 unique values
- **High Cardinality**: 50+ unique values but <10% of total rows
- **Strategy**: Synonym merging, fuzzy matching, case normalization

### Free Text Columns
- **Detection**: High cardinality (>10% unique values)
- **Strategy**: Unicode normalization, punctuation standardization

### Specialized Columns
- **Email**: Regex pattern matching + MX validation
- **Phone**: E.164 formatting with region inference
- **URL**: Canonicalization and validation
- **Currency/Price**: Currency detection and base currency conversion
- **Rating/Score**: Text-to-numeric mapping
- **Date/DateTime**: Locale-aware parsing with timezone normalization

## Missing Value Handling

### Numeric Columns
- **Strategy**: Median imputation
- **Rationale**: Median is robust to outliers

### Categorical Columns
- **Strategy**: Mode or "unknown"
- **Rationale**: Preserves categorical nature

### Date Columns
- **Strategy**: Forward fill, then median date
- **Rationale**: Maintains temporal relationships

### Text Columns
- **Strategy**: Leave blank, record in audit
- **Rationale**: Avoids introducing false information

## Outlier Handling

### Method: IQR (Interquartile Range)
- **Formula**: Q1 - 1.5×IQR to Q3 + 1.5×IQR
- **Action**: Cap outliers instead of removing
- **Exclusions**: ID columns, codes, and hash columns

## Standardization Rules

### Text Standardization
1. Unicode NFKC normalization
2. Case folding (lowercase)
3. Whitespace trimming
4. Punctuation normalization

### Email Standardization
1. Format validation
2. Case normalization
3. Domain validation (optional)

### Phone Standardization
1. Region inference
2. E.164 formatting
3. Validation

### URL Standardization
1. Scheme normalization
2. Domain canonicalization
3. Path normalization

## Categorical Canonicalization

### Synonym Tables
Pre-defined synonym mappings for common categories:
- **Status**: delivered, completed, shipped → delivered
- **Gender**: male, m, man → male
- **Rating**: excellent, outstanding, amazing → excellent

### Fuzzy Matching
- **Threshold**: 85% similarity
- **Algorithm**: RapidFuzz with token_sort_ratio
- **Fallback**: Manual review for low-confidence matches

## Quality Gates

### Great Expectations Integration
- **Uniqueness**: ID columns must be unique
- **Non-null**: Required columns cannot be null
- **Type expectations**: Enforced data types
- **Range checks**: Numeric value ranges
- **Allowed sets**: Categorical value validation
- **Regex patterns**: Format validation

### Auto-fix Rules
- **Safe fixes**: Null imputation, range capping
- **Quarantine**: Invalid values, format violations
- **Manual review**: High-impact changes

## Domain-Specific Rules

### Healthcare
- **Age range**: 0-150 years
- **Gender values**: male, female, other
- **Required columns**: patient_id, age, gender

### Finance
- **Amount range**: 0-1,000,000
- **Currency codes**: USD, EUR, GBP, JPY, INR
- **Required columns**: account_id, amount, transaction_date

### Retail
- **Price range**: 0-10,000
- **Status values**: active, inactive, discontinued
- **Required columns**: product_id, price, category

## Audit and Explainability

### Decision Logging
Every cleaning decision is logged with:
- **Policy decision**: Which rule was applied
- **Before/after samples**: Data transformation evidence
- **Confidence score**: Decision confidence (0-1)
- **Rules applied**: List of specific rules
- **Quarantine reason**: Why data was quarantined

### Quality Metrics
- **Quality score**: Overall data quality (0-1)
- **Violation counts**: Errors, warnings, info
- **Auto-fix counts**: Number of automatic fixes
- **Quarantine counts**: Number of quarantined rows

## Scalability Considerations

### Streaming Processing
- **Chunk size**: 10,000 rows (configurable)
- **Memory limit**: 1GB (configurable)
- **Resume capability**: Idempotent operations

### Performance Optimization
- **Polars**: Fast DataFrame operations
- **Vectorized operations**: Batch processing
- **Memory management**: Garbage collection triggers

## Error Handling

### Graceful Degradation
- **Missing dependencies**: Fallback to basic operations
- **Memory limits**: Automatic chunking
- **Invalid data**: Quarantine with reasons

### Recovery Mechanisms
- **Retry logic**: Exponential backoff
- **Checkpointing**: Resume from failures
- **Partial results**: Return what was processed

## Configuration

### Environment Variables
- `CLEANER_CHUNK_SIZE`: Default chunk size
- `CLEANER_MAX_MEMORY`: Memory limit in MB
- `CLEANER_TEMP_DIR`: Temporary directory

### Runtime Configuration
- **Domain selection**: Healthcare, finance, retail, general
- **Quality gates**: Enable/disable validation
- **Quarantine**: Enable/disable problematic row isolation
- **Output format**: Parquet, CSV, or both
