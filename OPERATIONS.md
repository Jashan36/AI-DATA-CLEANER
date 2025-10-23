# Operations Guide

This document provides operational guidance for running and maintaining the AI Data Cleaner & Web Scraper in production environments.

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB free space
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ free space
- **Python**: 3.9+

### Dependencies
- **Polars**: For fast DataFrame operations
- **Great Expectations**: For data quality validation
- **spaCy**: For NLP processing
- **FastAPI**: For API server
- **Streamlit**: For web interface

## Installation

### Local Installation

```bash
# Clone repository
git clone <repository-url>
cd AI-DATA-CLEANER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Docker Installation

```bash
# Build Docker image
docker build -t ai-data-cleaner .

# Run container
docker run -p 8000:8000 -p 8501:8501 ai-data-cleaner
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Configuration

### Environment Variables

```bash
# Data cleaning settings
export CLEANER_CHUNK_SIZE=10000
export CLEANER_MAX_MEMORY=1024
export CLEANER_TEMP_DIR=/tmp/cleaner

# API settings
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Streamlit settings
export STREAMLIT_PORT=8501
export STREAMLIT_HOST=0.0.0.0

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/ai-cleaner.log
```

### Configuration Files

Create `config.yaml`:

```yaml
data_cleaning:
  chunk_size: 10000
  max_memory_mb: 1024
  temp_dir: /tmp/cleaner
  enable_quality_gates: true
  enable_quarantine: true

web_scraping:
  max_concurrent: 10
  timeout: 30
  retry_attempts: 3
  backoff_factor: 1.5
  respect_robots: true

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  max_file_size: 100MB

streamlit:
  port: 8501
  host: 0.0.0.0
  theme: light

logging:
  level: INFO
  file: /var/log/ai-cleaner.log
  max_size: 100MB
  backup_count: 5
```

## Running the Application

### Streamlit Web Interface

```bash
# Start Streamlit app
streamlit run streamlit_app.py

# With custom configuration
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### FastAPI Server

```bash
# Development server
python api.py

# Production server with Gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### CLI Tool

```bash
# Clean data
python cli.py clean-data --input data.csv --output results --domain healthcare

# Scrape URLs
python cli.py scrape --urls https://example.com https://another-site.com --output results
```

## Monitoring and Logging

### Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('ai-cleaner.log', maxBytes=100*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### Health Checks

Create `health_check.py`:

```python
import requests
import sys

def check_api_health():
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_streamlit_health():
    try:
        response = requests.get('http://localhost:8501/', timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == '__main__':
    api_healthy = check_api_health()
    streamlit_healthy = check_streamlit_health()
    
    if api_healthy and streamlit_healthy:
        print("All services healthy")
        sys.exit(0)
    else:
        print("Some services unhealthy")
        sys.exit(1)
```

### Monitoring Metrics

Key metrics to monitor:

- **Processing time**: Time to complete data cleaning jobs
- **Memory usage**: RAM consumption during processing
- **Disk usage**: Temporary file storage usage
- **Error rates**: Failed jobs and error types
- **Queue length**: Number of pending jobs
- **Throughput**: Rows processed per minute

## Performance Tuning

### Memory Optimization

```python
# Adjust chunk size based on available memory
chunk_size = min(10000, available_memory_mb * 1000 // estimated_row_size)

# Enable garbage collection
import gc
gc.collect()
```

### CPU Optimization

```python
# Use multiprocessing for large datasets
from multiprocessing import Pool

def process_chunk(chunk):
    # Process chunk
    pass

with Pool(processes=4) as pool:
    results = pool.map(process_chunk, chunks)
```

### I/O Optimization

```python
# Use Polars for fast I/O
import polars as pl

# Read with streaming
df = pl.scan_csv('large_file.csv')

# Write with compression
df.write_parquet('output.parquet', compression='snappy')
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Disk Space Issues
```bash
# Check disk usage
df -h
du -sh /tmp/cleaner/*

# Clean temporary files
find /tmp/cleaner -type f -mtime +1 -delete
```

#### Network Issues
```bash
# Check network connectivity
ping google.com
curl -I https://example.com

# Check DNS resolution
nslookup example.com
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 1001 | Memory limit exceeded | Increase max_memory or reduce chunk_size |
| 1002 | Disk space insufficient | Free up disk space or change temp_dir |
| 1003 | Invalid file format | Ensure file is CSV or Parquet |
| 1004 | Network timeout | Check network connectivity and increase timeout |
| 1005 | Permission denied | Check file permissions and user access |

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python cli.py clean-data --input data.csv --output results --verbose

# Enable API debug mode
python api.py --debug
```

## Backup and Recovery

### Data Backup

```bash
# Backup configuration
cp config.yaml config.yaml.backup

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz /var/log/ai-cleaner.log*

# Backup temporary data
rsync -av /tmp/cleaner/ /backup/cleaner/
```

### Recovery Procedures

```bash
# Restore configuration
cp config.yaml.backup config.yaml

# Restart services
systemctl restart ai-cleaner-api
systemctl restart ai-cleaner-streamlit

# Check service status
systemctl status ai-cleaner-api
systemctl status ai-cleaner-streamlit
```

## Security Considerations

### File Permissions

```bash
# Set proper permissions
chmod 600 config.yaml
chmod 755 /var/log/ai-cleaner.log
chmod 700 /tmp/cleaner
```

### Network Security

```bash
# Firewall rules
ufw allow 8000/tcp  # API
ufw allow 8501/tcp  # Streamlit
ufw deny 22/tcp     # SSH (if not needed)
```

### Input Validation

```python
# Validate file types
ALLOWED_EXTENSIONS = {'.csv', '.parquet'}
ALLOWED_MIME_TYPES = {'text/csv', 'application/octet-stream'}

def validate_file(file):
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise ValueError("Invalid file type")
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise ValueError("Invalid MIME type")
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    image: ai-data-cleaner
    ports:
      - "8000:8000"
    deploy:
      replicas: 3
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Load Balancing

```nginx
# nginx.conf
upstream api_backend {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Job Queue

```python
# Use Celery for job queuing
from celery import Celery

app = Celery('ai-cleaner')

@app.task
def clean_data_task(file_path, config):
    # Process data cleaning
    pass

@app.task
def scrape_urls_task(urls, config):
    # Process web scraping
    pass
```

## Maintenance

### Regular Tasks

```bash
# Daily cleanup
find /tmp/cleaner -type f -mtime +1 -delete

# Weekly log rotation
logrotate /etc/logrotate.d/ai-cleaner

# Monthly backup
tar -czf backup_$(date +%Y%m).tar.gz /var/log/ai-cleaner.log*
```

### Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update Docker images
docker-compose pull
docker-compose up -d

# Restart services
systemctl restart ai-cleaner-api
systemctl restart ai-cleaner-streamlit
```

### Performance Monitoring

```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor application logs
tail -f /var/log/ai-cleaner.log

# Monitor API endpoints
curl -s http://localhost:8000/status/health | jq
```
