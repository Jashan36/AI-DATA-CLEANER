# 🔧 AI Data Cleaner & Web Scraper

A fully automated Data Quality & Web-Scrape Summarizer that autonomously cleans tabular data and turns web pages into clear, human-readable summaries with traceable structure. Built with policy-driven cleaning, comprehensive audit logging, and scalable processing.

## 🎯 Core Features

### Policy-Driven Data Cleaning
- ✅ **Automatic Column Type Detection**: ID/Hash, Categorical, Free-text, Email, Phone, URL, Code, Currency, Rating, Date, Numeric, Boolean
- ✅ **Intelligent Strategy Selection**: Each column type gets appropriate cleaning strategy
- ✅ **Missing Value Handling**: Median for numeric, mode/unknown for categorical, forward-fill for dates
- ✅ **Outlier Management**: IQR-based capping with vectorized Polars operations
- ✅ **Standardization**: Unicode NFKC, case-folding, punctuation normalization

### Quality Gates & Validation
- ✅ **Great Expectations Integration**: Uniqueness, non-null, type expectations, range checks
- ✅ **Auto-fix Capabilities**: Safe transformations with quarantine for problematic data
- ✅ **Domain-Specific Rules**: Healthcare, finance, retail with specialized validation
- ✅ **Data Quality Reports**: Comprehensive DQR with before/after metrics

### Scalable Processing
- ✅ **Streaming Chunks**: Polars scan for memory-efficient processing
- ✅ **OOM Safety**: Configurable memory limits with garbage collection
- ✅ **Resume-able Runs**: Idempotent operations for large datasets
- ✅ **Intermediate Storage**: Parquet files for checkpointing

### Web Scraping & Summarization
- ✅ **Async Pipeline**: Concurrent requests with retry/backoff
- ✅ **Robots.txt Respect**: Ethical scraping with rate limiting
- ✅ **Structured Extraction**: About, Products, Team, Contact, Social, Pricing, FAQ, Jobs, Policies
- ✅ **NER + Rule-based**: spaCy entities with TextBlob fallback
- ✅ **Human-readable Summaries**: Concise bullet points (15-22 words) with source anchors

## 🚀 Enhanced Features

### Explainability & Audit
- ✅ **Comprehensive Audit Logs**: Policy decisions, before/after samples, confidence scores
- ✅ **Rule Tracking**: Every transformation logged with reasoning
- ✅ **Quarantine Management**: Problematic rows isolated with detailed reasons
- ✅ **Quality Metrics**: Before/after statistics, violation counts, auto-fix tracking

### Multiple Interfaces
- ✅ **Streamlit UI**: Upload/Connect, Quality Report, Explore, Web Summaries, Explain, Downloads
- ✅ **CLI Tool**: `clean-data` and `scrape` commands with full configuration
- ✅ **FastAPI**: REST endpoints for `/clean` and `/scrape` with async processing
- ✅ **Python API**: Direct integration for custom workflows

### Export Formats
- ✅ **Cleaned Data**: Parquet/CSV with preserved structure
- ✅ **Quarantined Data**: Problematic rows with reasons
- ✅ **Audit Logs**: JSON with complete transformation history
- ✅ **Quality Reports**: Markdown DQR with recommendations
- ✅ **Web Summaries**: Structured JSON + human-readable markdown

## 📊 Output Formats

### 1. Cleaned Dataset
- ✅ **Parquet/CSV**: High-performance format with preserved structure
- ✅ **Quality Assured**: Validated against domain-specific rules
- ✅ **Audit Trail**: Complete transformation history
- ✅ **Human-readable**: Maintains semantic richness

### 2. Comprehensive Audit Log (JSON)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "input_shape": [1000, 9],
  "output_shape": [950, 9],
  "quarantined_rows": 50,
  "quality_score": 0.95,
  "decisions": [
    {
      "column": "status",
      "column_type": "categorical_low",
      "strategy": "canonicalize_categorical",
      "confidence": 0.9,
      "rules_applied": ["case_folding", "synonym_merging"],
      "before_sample": ["Delivered", "DELIVERED", "delivered"],
      "after_sample": ["delivered", "delivered", "delivered"]
    }
  ]
}
```

### 3. Data Quality Report (Markdown)
- ✅ **Quality Score**: Overall data quality percentage
- ✅ **Violation Summary**: Errors, warnings, info counts
- ✅ **Auto-fixes Applied**: List of automatic corrections
- ✅ **Recommendations**: Actionable improvement suggestions

### 4. Web Scraping Results
- ✅ **Structured Cards**: JSON with extracted information
- ✅ **Human Summary**: Markdown with bullet points
- ✅ **Raw Text**: Cleaned HTML content
- ✅ **Entity Extraction**: NER results with confidence scores

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+ (for local installation)
- Docker & Docker Compose (for containerized deployment)
- 2GB+ RAM recommended
- 1GB+ storage for models

### 🐳 Docker Installation (Recommended)

#### Quick Start with Docker Compose
```bash
# Download the project
# Extract the project files to your desired location
cd AI-DATA-CLEANER

# Build and run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:8501
```

#### Manual Docker Build
```bash
# Build the Docker image
docker build -t ai-data-cleaner .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output ai-data-cleaner

# Access the application at http://localhost:8501
```

#### Optional: Jupyter Lab Service
```bash
# Run with Jupyter Lab for data exploration
docker-compose --profile jupyter up --build

# Access Jupyter Lab at http://localhost:8888
```

### 📦 Local Installation

#### Traditional Installation
```bash
# Download the project
# Extract the project files to your desired location
cd AI-DATA-CLEANER

# Install dependencies
pip install -r requirements.txt

# Install optional NLP models
python -m spacy download en_core_web_sm
```

#### Streamlit Web App
```bash
streamlit run streamlit_app.py
```

#### CLI Tool
```bash
# Clean data
python cli.py clean-data --input data.csv --output results --domain healthcare

# Scrape URLs
python cli.py scrape --urls https://example.com https://another-site.com --output results
```

#### FastAPI Server
```bash
python api.py
# Access API at http://localhost:8000
```

#### Python API
```python
from core.data_cleaner import DataCleaner, CleaningConfig

# Initialize cleaner
config = CleaningConfig(
    chunk_size=10000,
    enable_quality_gates=True,
    enable_quarantine=True
)
cleaner = DataCleaner(config)

# Clean data file
result = cleaner.clean_file('messy_data.csv', 'output_dir', domain='healthcare')

if result.success:
    print(f"Cleaned {result.total_rows_cleaned} rows")
    print(f"Quarantined {result.total_rows_quarantined} rows")
    print(f"Processing time: {result.processing_time:.2f}s")
```

#### Example Usage
```bash
python example_usage.py
```

## 📁 Project Structure
```
AI-DATA-CLEANER/
├── core/                   # Core modules
│   ├── cleaner_policy.py   # Policy-driven cleaning engine
│   ├── data_cleaner.py     # Main data cleaner with streaming
│   ├── quality_gates.py    # Great Expectations integration
│   └── web_scraper.py      # Async web scraping pipeline
├── tests/                  # Test suite
│   ├── test_cleaner_policy.py
│   ├── test_quality_gates.py
│   ├── test_data_cleaner.py
│   └── test_web_scraper.py
├── streamlit_app.py        # Streamlit web interface
├── cli.py                  # Command-line interface
├── api.py                  # FastAPI REST endpoints
├── test_core.py           # Basic functionality tests
├── run_tests.py           # Test runner
├── requirements.txt       # Dependencies
├── POLICIES.md            # Cleaning policies documentation
├── API.md                 # API documentation
├── OPERATIONS.md          # Operations guide
├── MAPPINGS.md            # Categorical mappings
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # This file
```

## 🎯 Example Workflow

### Input: Messy Healthcare Dataset
```csv
Patient ID,Age,Gender,Symptoms,Diagnosis,Visit Date
P001,25,Male,Fever,Hypertension,2024-01-15
P002,30,FEMALE,FEVER,HYPERTENSION,15/01/2024
P003,45,M,Headache,Diabetes,Jan 15, 2024
P004,60,Female,COUGH,DIABETES,2024-01-15
P005,35,Other,Nausea,Asthma,15-01-2024
```

### Output: Cleaned Dataset
```csv
patient_id,age,gender,symptoms,diagnosis,visit_date
P001,25,male,fever,hypertension,2024-01-15
P002,30,female,fever,hypertension,2024-01-15
P003,45,male,headache,diabetes,2024-01-15
P004,60,female,cough,diabetes,2024-01-15
P005,35,other,nausea,asthma,2024-01-15
```

### Audit Log Summary
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "input_shape": [5, 6],
  "output_shape": [5, 6],
  "quarantined_rows": 0,
  "quality_score": 0.95,
  "decisions": [
    {
      "column": "gender",
      "strategy": "canonicalize_categorical",
      "confidence": 0.9,
      "rules_applied": ["case_folding", "synonym_merging"]
    }
  ]
}
```

### Web Scraping Example
```bash
python cli.py scrape --urls https://example.com --output results
```

**Output:**
- `scrape_summary.md`: Human-readable summary with bullet points
- `scrape_cards.json`: Structured information cards
- `scrape_raw_text.txt`: Cleaned HTML content

## 🔧 Behavioral Constraints

- ❌ **No Hallucination**: Never guess or invent values
- ✅ **Safe Imputation**: Replace unparseable values with NaN or "unknown"
- ✅ **Policy-Driven**: All decisions based on explicit rules and policies
- ✅ **Audit Trail**: Complete logging of all transformations with reasoning
- ✅ **Quality Gates**: Automatic validation with quarantine for problematic data

## 📈 Performance Features

- ⚡ **Polars Optimization**: Fast DataFrame operations with streaming
- 🚀 **Memory Efficient**: Handles datasets up to 1M+ rows with chunking
- 📊 **Async Processing**: Concurrent web scraping with rate limiting
- 🔄 **Resume-able**: Idempotent operations for large datasets
- 🛡️ **OOM Safe**: Configurable memory limits with garbage collection

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test modules
python -m pytest tests/test_cleaner_policy.py -v
python -m pytest tests/test_quality_gates.py -v
python -m pytest tests/test_data_cleaner.py -v
python -m pytest tests/test_web_scraper.py -v

# Run basic functionality test
python test_core.py
```

## 📚 Documentation

- **[POLICIES.md](POLICIES.md)**: Detailed cleaning policies and decision rules
- **[API.md](API.md)**: REST API documentation with examples
- **[OPERATIONS.md](OPERATIONS.md)**: Production deployment and operations guide
- **[MAPPINGS.md](MAPPINGS.md)**: Categorical value mappings and canonicalization

## 🤝 Contributing

This tool is designed for production use. Contributions should maintain high standards:
- Comprehensive audit logging and explainability
- Policy-driven decision making
- Performance optimization and scalability
- Error handling and validation
- Complete test coverage

## 📄 License
MIT License - See LICENSE file for details

---

**Built for Data Scientists & Engineers who demand production-grade data quality with complete transparency and audit trails.**
