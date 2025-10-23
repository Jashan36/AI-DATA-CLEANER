# API Documentation

This document describes the REST API endpoints for the AI Data Cleaner & Web Scraper.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement proper authentication mechanisms.

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "AI Data Cleaner & Web Scraper API",
  "version": "1.0.0",
  "endpoints": {
    "clean": "/clean - Upload and clean data files",
    "scrape": "/scrape - Scrape and summarize web pages",
    "status": "/status/{job_id} - Check job status",
    "download": "/download/{job_id} - Download results"
  }
}
```

### 2. Data Cleaning

**POST** `/clean`

Upload and clean a data file with policy-driven automation.

**Parameters:**
- `file` (file, required): CSV or Parquet file to clean
- `domain` (string, optional): Data domain (healthcare, finance, retail, general)
- `chunk_size` (integer, optional): Chunk size for streaming processing (default: 10000)
- `max_memory` (integer, optional): Maximum memory usage in MB (default: 1024)
- `output_format` (string, optional): Output format (parquet, csv, both) (default: parquet)
- `enable_quality_gates` (boolean, optional): Enable data quality validation (default: true)
- `enable_quarantine` (boolean, optional): Enable quarantining of problematic rows (default: true)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "message": "Data cleaning started. Use /status/{job_id} to check progress."
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/clean" \
  -F "file=@data.csv" \
  -F "domain=healthcare" \
  -F "chunk_size=5000"
```

### 3. Web Scraping

**POST** `/scrape`

Scrape and summarize multiple URLs.

**Request Body:**
```json
{
  "urls": ["https://example.com", "https://another-site.com"],
  "max_concurrent": 10,
  "timeout": 30,
  "retry_attempts": 3,
  "backoff_factor": 1.5,
  "ignore_robots": false
}
```

**Parameters:**
- `urls` (array, required): List of URLs to scrape
- `max_concurrent` (integer, optional): Maximum concurrent requests (default: 10)
- `timeout` (integer, optional): Request timeout in seconds (default: 30)
- `retry_attempts` (integer, optional): Number of retry attempts (default: 3)
- `backoff_factor` (float, optional): Exponential backoff factor (default: 1.5)
- `ignore_robots` (boolean, optional): Ignore robots.txt restrictions (default: false)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "message": "Web scraping started for 2 URLs. Use /status/{job_id} to check progress."
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/scrape" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com", "https://another-site.com"],
    "max_concurrent": 5,
    "timeout": 20
  }'
```

### 4. Job Status

**GET** `/status/{job_id}`

Check the status of a processing job.

**Response:**
```json
{
  "status": "completed",
  "start_time": "2024-01-15T10:30:00",
  "progress": 100,
  "message": "Data cleaning completed successfully",
  "result": {
    "total_rows_processed": 10000,
    "total_rows_cleaned": 9500,
    "total_rows_quarantined": 500,
    "processing_time": 45.2,
    "files": {
      "cleaned_data": "/path/to/cleaned_data.parquet",
      "quarantined_data": "/path/to/quarantined_data.parquet",
      "audit_log": "/path/to/audit_log.json",
      "quality_report": "/path/to/quality_report.json",
      "dqr": "/path/to/dqr.md"
    }
  }
}
```

**Status Values:**
- `processing`: Job is currently running
- `completed`: Job completed successfully
- `failed`: Job failed with error

**Example:**
```bash
curl "http://localhost:8000/status/123e4567-e89b-12d3-a456-426614174000"
```

### 5. Download Results

**GET** `/download/{job_id}`

Download results from a completed job.

**Parameters:**
- `job_id` (string, required): Job ID
- `file_type` (string, optional): Type of file to download (default: all)

**File Types:**
- `all`: Download all results as ZIP file
- `cleaned`: Download cleaned data (Parquet)
- `quarantined`: Download quarantined data (Parquet)
- `audit`: Download audit log (JSON)
- `quality`: Download quality report (JSON)
- `dqr`: Download Data Quality Report (Markdown)
- `summary`: Download scraping summary (Markdown)
- `cards`: Download structured cards (JSON)
- `raw_text`: Download raw text (TXT)

**Response:**
File download with appropriate MIME type.

**Example:**
```bash
# Download all results
curl "http://localhost:8000/download/123e4567-e89b-12d3-a456-426614174000" -o results.zip

# Download specific file
curl "http://localhost:8000/download/123e4567-e89b-12d3-a456-426614174000?file_type=cleaned" -o cleaned_data.parquet
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Only CSV and Parquet files are supported"
}
```

### 404 Not Found
```json
{
  "detail": "Job not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Unexpected error occurred"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement appropriate rate limiting based on your requirements.

## CORS

CORS is enabled for all origins. In production, configure CORS appropriately for your security requirements.

## WebSocket Support

WebSocket support for real-time progress updates is planned for future releases.

## SDK Examples

### Python

```python
import requests
import time

# Upload and clean data
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/clean',
        files={'file': f},
        data={'domain': 'healthcare'}
    )

job_id = response.json()['job_id']

# Check status
while True:
    status_response = requests.get(f'http://localhost:8000/status/{job_id}')
    status = status_response.json()
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Job failed: {status['message']}")
        break
    
    print(f"Progress: {status['progress']}%")
    time.sleep(5)

# Download results
download_response = requests.get(f'http://localhost:8000/download/{job_id}')
with open('results.zip', 'wb') as f:
    f.write(download_response.content)
```

### JavaScript

```javascript
// Upload and clean data
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('domain', 'healthcare');

const response = await fetch('http://localhost:8000/clean', {
  method: 'POST',
  body: formData
});

const { job_id } = await response.json();

// Check status
const checkStatus = async () => {
  const statusResponse = await fetch(`http://localhost:8000/status/${job_id}`);
  const status = await statusResponse.json();
  
  if (status.status === 'completed') {
    // Download results
    const downloadResponse = await fetch(`http://localhost:8000/download/${job_id}`);
    const blob = await downloadResponse.blob();
    
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.zip';
    a.click();
  } else if (status.status === 'failed') {
    console.error('Job failed:', status.message);
  } else {
    console.log(`Progress: ${status.progress}%`);
    setTimeout(checkStatus, 5000);
  }
};

checkStatus();
```

## CLI Integration

The API can be used with the CLI tool:

```bash
# Start API server
python api.py

# Use CLI with API
python cli.py clean-data --input data.csv --output results --api-url http://localhost:8000
```

## Production Considerations

### Security
- Implement authentication and authorization
- Use HTTPS in production
- Validate and sanitize all inputs
- Implement rate limiting
- Configure CORS appropriately

### Performance
- Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
- Implement caching for frequently accessed data
- Use a reverse proxy (e.g., Nginx)
- Monitor resource usage

### Monitoring
- Implement logging and monitoring
- Use health check endpoints
- Monitor job queue and processing times
- Set up alerts for failures

### Scalability
- Use a job queue system (e.g., Celery with Redis/RabbitMQ)
- Implement horizontal scaling
- Use a distributed file system for large files
- Consider using a database for job state management
