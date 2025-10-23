"""
FastAPI interface for AI Data Cleaner & Web Scraper.
Provides REST endpoints for data cleaning and web scraping operations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
import uuid

from core.data_cleaner import DataCleaner, CleaningConfig
from core.web_scraper import WebScraper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Data Cleaner & Web Scraper API",
    description="Automated data quality improvement and web content summarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for job status (in production, use Redis or database)
job_status = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Data Cleaner & Web Scraper API",
        "version": "1.0.0",
        "endpoints": {
            "clean": "/clean - Upload and clean data files",
            "scrape": "/scrape - Scrape and summarize web pages",
            "status": "/status/{job_id} - Check job status",
            "download": "/download/{job_id} - Download results"
        }
    }

@app.post("/clean")
async def clean_data(
    file: UploadFile = File(...),
    domain: Optional[str] = "general",
    chunk_size: int = 10000,
    max_memory: int = 1024,
    output_format: str = "parquet",
    enable_quality_gates: bool = True,
    enable_quarantine: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Clean uploaded data file with policy-driven automation.
    
    Args:
        file: CSV or Parquet file to clean
        domain: Data domain (healthcare, finance, retail, general)
        chunk_size: Chunk size for streaming processing
        max_memory: Maximum memory usage in MB
        output_format: Output format (parquet, csv, both)
        enable_quality_gates: Enable data quality validation
        enable_quarantine: Enable quarantining of problematic rows
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.parquet')):
            raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")
        
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp(prefix=f'clean_{job_id}_'))
        
        # Save uploaded file
        input_path = temp_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize job status
        job_status[job_id] = {
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "message": "Starting data cleaning...",
            "temp_dir": str(temp_dir)
        }
        
        # Start background processing
        background_tasks.add_task(
            process_clean_job,
            job_id,
            str(input_path),
            str(temp_dir),
            domain,
            chunk_size,
            max_memory,
            output_format,
            enable_quality_gates,
            enable_quarantine
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Data cleaning started. Use /status/{job_id} to check progress."
        }
        
    except Exception as e:
        logger.error(f"Error starting clean job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_urls(
    urls: List[str],
    max_concurrent: int = 10,
    timeout: int = 30,
    retry_attempts: int = 3,
    backoff_factor: float = 1.5,
    ignore_robots: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Scrape and summarize multiple URLs.
    
    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts
        backoff_factor: Exponential backoff factor
        ignore_robots: Ignore robots.txt restrictions
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Validate URLs
        if not urls:
            raise HTTPException(status_code=400, detail="At least one URL is required")
        
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp(prefix=f'scrape_{job_id}_'))
        
        # Initialize job status
        job_status[job_id] = {
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "message": "Starting web scraping...",
            "temp_dir": str(temp_dir)
        }
        
        # Start background processing
        background_tasks.add_task(
            process_scrape_job,
            job_id,
            urls,
            str(temp_dir),
            max_concurrent,
            timeout,
            retry_attempts,
            backoff_factor,
            ignore_robots
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Web scraping started for {len(urls)} URLs. Use /status/{job_id} to check progress."
        }
        
    except Exception as e:
        logger.error(f"Error starting scrape job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/download/{job_id}")
async def download_results(job_id: str, file_type: str = "all"):
    """
    Download results from a completed job.
    
    Args:
        job_id: Job ID
        file_type: Type of file to download (all, cleaned, quarantined, audit, quality, dqr, summary, cards)
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    temp_dir = Path(job["temp_dir"])
    
    # Map file types to actual files
    file_mapping = {
        "cleaned": "cleaned_data.parquet",
        "quarantined": "quarantined_data.parquet",
        "audit": "audit_log.json",
        "quality": "quality_report.json",
        "dqr": "dqr.md",
        "summary": "scrape_summary.md",
        "cards": "scrape_cards.json",
        "raw_text": "scrape_raw_text.txt"
    }
    
    if file_type == "all":
        # Create a zip file with all results
        import zipfile
        zip_path = temp_dir / f"{job_id}_results.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in temp_dir.glob("*"):
                if file_path.is_file() and file_path.name != zip_path.name:
                    zipf.write(file_path, file_path.name)
        
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=f"{job_id}_results.zip"
        )
    
    elif file_type in file_mapping:
        file_path = temp_dir / file_mapping[file_type]
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {file_type} not found")
        
        return FileResponse(
            file_path,
            filename=file_path.name
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")

async def process_clean_job(
    job_id: str,
    input_path: str,
    output_dir: str,
    domain: str,
    chunk_size: int,
    max_memory: int,
    output_format: str,
    enable_quality_gates: bool,
    enable_quarantine: bool
):
    """Background task for processing data cleaning jobs."""
    try:
        # Update job status
        job_status[job_id]["message"] = "Initializing data cleaner..."
        
        # Create cleaning configuration
        config = CleaningConfig(
            chunk_size=chunk_size,
            max_memory_mb=max_memory,
            enable_quality_gates=enable_quality_gates,
            enable_quarantine=enable_quarantine,
            output_format=output_format
        )
        
        # Initialize data cleaner
        cleaner = DataCleaner(config)
        
        # Update job status
        job_status[job_id]["message"] = "Processing data..."
        
        # Clean the data
        result = cleaner.clean_file(input_path, output_dir, domain)
        
        if result.success:
            job_status[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Data cleaning completed successfully",
                "result": {
                    "total_rows_processed": result.total_rows_processed,
                    "total_rows_cleaned": result.total_rows_cleaned,
                    "total_rows_quarantined": result.total_rows_quarantined,
                    "processing_time": result.processing_time,
                    "files": {
                        "cleaned_data": result.cleaned_data_path,
                        "quarantined_data": result.quarantined_data_path,
                        "audit_log": result.audit_log_path,
                        "quality_report": result.quality_report_path,
                        "dqr": result.dqr_path
                    }
                }
            })
        else:
            job_status[job_id].update({
                "status": "failed",
                "message": f"Data cleaning failed: {result.error_message}"
            })
    
    except Exception as e:
        logger.error(f"Error processing clean job {job_id}: {e}")
        job_status[job_id].update({
            "status": "failed",
            "message": f"Unexpected error: {str(e)}"
        })

async def process_scrape_job(
    job_id: str,
    urls: List[str],
    output_dir: str,
    max_concurrent: int,
    timeout: int,
    retry_attempts: int,
    backoff_factor: float,
    ignore_robots: bool
):
    """Background task for processing web scraping jobs."""
    try:
        # Update job status
        job_status[job_id]["message"] = "Initializing web scraper..."
        
        # Initialize web scraper
        scraper = WebScraper(
            max_concurrent=max_concurrent,
            timeout=timeout,
            retry_attempts=retry_attempts,
            backoff_factor=backoff_factor,
            respect_robots=not ignore_robots
        )
        
        # Update job status
        job_status[job_id]["message"] = f"Scraping {len(urls)} URLs..."
        
        # Scrape URLs
        results = await scraper.scrape_urls(urls)
        
        # Update job status
        job_status[job_id]["message"] = "Exporting results..."
        
        # Export results
        scraper.export_results(results, output_dir)
        
        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        job_status[job_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"Web scraping completed: {successful} successful, {failed} failed",
            "result": {
                "total_urls": len(urls),
                "successful": successful,
                "failed": failed,
                "results": [
                    {
                        "url": r.url,
                        "success": r.success,
                        "title": r.title,
                        "error": r.error_message
                    }
                    for r in results
                ]
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing scrape job {job_id}: {e}")
        job_status[job_id].update({
            "status": "failed",
            "message": f"Unexpected error: {str(e)}"
        })

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Clean up temporary directories
    for job in job_status.values():
        temp_dir = job.get("temp_dir")
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir {temp_dir}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
