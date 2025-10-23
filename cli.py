"""
Command-line interface for AI Data Cleaner & Web Scraper.
Provides clean-data and scrape commands with comprehensive options.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
import asyncio

from core.data_cleaner import DataCleaner, CleaningConfig
from core.web_scraper import WebScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_clean_data_parser() -> argparse.ArgumentParser:
    """Setup argument parser for clean-data command."""
    parser = argparse.ArgumentParser(
        description='Clean tabular data with policy-driven automation',
        prog='clean-data'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input data file (CSV or Parquet)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for cleaned data and reports'
    )
    
    parser.add_argument(
        '--domain', '-d',
        choices=['healthcare', 'finance', 'retail', 'general'],
        default='general',
        help='Data domain for specialized cleaning rules'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for streaming processing (default: 10000)'
    )
    
    parser.add_argument(
        '--max-memory',
        type=int,
        default=1024,
        help='Maximum memory usage in MB (default: 1024)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv', 'both'],
        default='parquet',
        help='Output format for cleaned data (default: parquet)'
    )
    
    parser.add_argument(
        '--disable-quality-gates',
        action='store_true',
        help='Disable data quality validation'
    )
    
    parser.add_argument(
        '--disable-quarantine',
        action='store_true',
        help='Disable quarantining of problematic rows'
    )
    
    parser.add_argument(
        '--temp-dir',
        help='Temporary directory for processing (default: system temp)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser

def setup_scrape_parser() -> argparse.ArgumentParser:
    """Setup argument parser for scrape command."""
    parser = argparse.ArgumentParser(
        description='Scrape and summarize web pages',
        prog='scrape'
    )
    
    parser.add_argument(
        '--urls', '-u',
        nargs='+',
        required=True,
        help='URLs to scrape'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for scraping results'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent requests (default: 10)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=3,
        help='Number of retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--backoff-factor',
        type=float,
        default=1.5,
        help='Exponential backoff factor (default: 1.5)'
    )
    
    parser.add_argument(
        '--ignore-robots',
        action='store_true',
        help='Ignore robots.txt restrictions'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser

def clean_data_command(args) -> int:
    """Execute clean-data command."""
    try:
        # Setup logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create cleaning configuration
        config = CleaningConfig(
            chunk_size=args.chunk_size,
            max_memory_mb=args.max_memory,
            enable_quality_gates=not args.disable_quality_gates,
            enable_quarantine=not args.disable_quarantine,
            output_format=args.output_format,
            temp_dir=args.temp_dir
        )
        
        # Initialize data cleaner
        cleaner = DataCleaner(config)
        
        logger.info(f"Starting data cleaning: {input_path} -> {output_path}")
        logger.info(f"Domain: {args.domain}")
        logger.info(f"Chunk size: {args.chunk_size}")
        logger.info(f"Max memory: {args.max_memory}MB")
        
        # Clean the data
        result = cleaner.clean_file(str(input_path), str(output_path), args.domain)
        
        if result.success:
            logger.info("Data cleaning completed successfully!")
            logger.info(f"Rows processed: {result.total_rows_processed}")
            logger.info(f"Rows cleaned: {result.total_rows_cleaned}")
            logger.info(f"Rows quarantined: {result.total_rows_quarantined}")
            logger.info(f"Processing time: {result.processing_time:.2f} seconds")
            logger.info(f"Cleaned data: {result.cleaned_data_path}")
            
            if result.quarantined_data_path:
                logger.info(f"Quarantined data: {result.quarantined_data_path}")
            
            logger.info(f"Audit log: {result.audit_log_path}")
            
            if result.quality_report_path:
                logger.info(f"Quality report: {result.quality_report_path}")
            
            if result.dqr_path:
                logger.info(f"Data Quality Report: {result.dqr_path}")
            
            return 0
        else:
            logger.error(f"Data cleaning failed: {result.error_message}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

async def scrape_command(args) -> int:
    """Execute scrape command."""
    try:
        # Setup logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize web scraper
        scraper = WebScraper(
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            retry_attempts=args.retry_attempts,
            backoff_factor=args.backoff_factor,
            respect_robots=not args.ignore_robots
        )
        
        logger.info(f"Starting web scraping: {len(args.urls)} URLs")
        logger.info(f"Max concurrent: {args.max_concurrent}")
        logger.info(f"Timeout: {args.timeout}s")
        logger.info(f"Retry attempts: {args.retry_attempts}")
        
        # Scrape URLs
        results = await scraper.scrape_urls(args.urls)
        
        # Export results
        scraper.export_results(results, str(output_path))
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        logger.info(f"Scraping completed: {successful} successful, {failed} failed")
        logger.info(f"Results exported to: {output_path}")
        
        if failed > 0:
            logger.warning("Some URLs failed to scrape:")
            for result in results:
                if not result.success:
                    logger.warning(f"  {result.url}: {result.error_message}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI Data Cleaner & Web Scraper',
        prog='ai-data-cleaner'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add subcommands
    clean_parser = setup_clean_data_parser()
    scrape_parser = setup_scrape_parser()
    
    subparsers.add_parser('clean-data', parents=[clean_parser], add_help=False)
    subparsers.add_parser('scrape', parents=[scrape_parser], add_help=False)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'clean-data':
        return clean_data_command(args)
    elif args.command == 'scrape':
        return asyncio.run(scrape_command(args))
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
