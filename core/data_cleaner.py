"""
Main data cleaner that integrates policy-driven cleaning, quality gates, and scalability features.
Handles streaming chunks, OOM safety, and resume-able runs.
Now also bundles the fast JAXDataCleaner pipeline for in-memory optimization.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import tempfile
from pathlib import Path
import gc

from .cleaner_policy import CleanerPolicy, AuditLog, ColumnType
from .quality_gates import QualityGates, QualityReport
from .data_processing import (
    JAXDataCleaner,
    analyze_data_quality,
    detect_target_type,
)

logger = logging.getLogger(__name__)

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    chunk_size: int = 10000
    max_memory_mb: int = 1024
    enable_quality_gates: bool = True
    enable_quarantine: bool = True
    output_format: str = 'parquet'  # 'parquet', 'csv', 'both'
    resume_enabled: bool = True
    temp_dir: Optional[str] = None
    use_fast_pipeline: bool = True
    fast_mode_threshold_rows: int = 150000
    save_ml_ready_output: bool = True

@dataclass
class CleaningResult:
    """Result of data cleaning operation."""
    cleaned_data_path: str
    quarantined_data_path: Optional[str]
    audit_log_path: str
    quality_report_path: Optional[str]
    dqr_path: Optional[str]
    total_rows_processed: int
    total_rows_cleaned: int
    total_rows_quarantined: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    ml_ready_data_path: Optional[str] = None
    data_profile_path: Optional[str] = None
    pipeline_log_path: Optional[str] = None

class DataCleaner:
    """
    Main data cleaner with streaming, scalability, and quality assurance.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.policy = CleanerPolicy()
        self.quality_gates = QualityGates()
        self.pipeline = JAXDataCleaner()
        
        # Setup temporary directory
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='data_cleaner_'))
        
        self.temp_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        self._reset_progress()

    def _reset_progress(self) -> None:
        """Reset internal progress counters."""
        self.progress = {
            'chunks_processed': 0,
            'total_chunks': 0,
            'rows_processed': 0,
            'rows_cleaned': 0,
            'rows_quarantined': 0,
            'start_time': None
        }

    def _should_use_fast_mode(self, total_rows: Optional[int]) -> bool:
        """Determine whether to run in-memory fast pipeline."""
        if not self.config.use_fast_pipeline:
            return False
        if total_rows is None:
            return False
        return total_rows <= self.config.fast_mode_threshold_rows

    def clean_file(self, input_path: str, output_dir: str, domain: Optional[str] = None) -> CleaningResult:
        """
        Clean a data file with streaming and quality assurance.
        """
        self._reset_progress()
        start_time = datetime.now()
        self.progress['start_time'] = start_time
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Determine file format and setup
            input_path = Path(input_path)
            file_format = input_path.suffix.lower()
            
            if file_format == '.csv':
                return self._clean_csv_file(input_path, output_path, domain)
            elif file_format == '.parquet':
                return self._clean_parquet_file(input_path, output_path, domain)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error cleaning file: {e}")
            return CleaningResult(
                cleaned_data_path="",
                quarantined_data_path=None,
                audit_log_path="",
                quality_report_path=None,
                dqr_path=None,
                total_rows_processed=0,
                total_rows_cleaned=0,
                total_rows_quarantined=0,
                processing_time=0,
                success=False,
                error_message=str(e)
            )

    def _clean_csv_file(self, input_path: Path, output_path: Path, domain: Optional[str]) -> CleaningResult:
        """Clean a CSV file with streaming."""
        # Use Polars for efficient streaming
        df_stream = pl.scan_csv(str(input_path))
        
        # Get total rows for progress tracking (use pl.len for newer Polars)
        total_rows = df_stream.select(pl.len()).collect().item()
        if self._should_use_fast_mode(total_rows):
            self.progress['total_chunks'] = 1
            return self._clean_in_memory(
                input_path,
                output_path,
                domain,
                loader=lambda: pd.read_csv(input_path),
                format_hint="csv"
            )
        self.progress['total_chunks'] = (total_rows + self.config.chunk_size - 1) // self.config.chunk_size
        
        # Process in chunks
        cleaned_chunks = []
        quarantined_chunks = []
        all_audit_logs = []
        
        for chunk_df in self._stream_dataframe_chunks(df_stream):
            cleaned_chunk, quarantined_chunk, audit_log = self._process_chunk(chunk_df, domain)
            
            if cleaned_chunk is not None:
                cleaned_chunks.append(cleaned_chunk)
            if quarantined_chunk is not None and len(quarantined_chunk) > 0:
                quarantined_chunks.append(quarantined_chunk)
            if audit_log:
                all_audit_logs.append(audit_log)
            
            # Memory management
            if self._check_memory_usage():
                gc.collect()
        
        # Combine results
        return self._combine_results(cleaned_chunks, quarantined_chunks, all_audit_logs, output_path, domain)

    def _clean_parquet_file(self, input_path: Path, output_path: Path, domain: Optional[str]) -> CleaningResult:
        """Clean a Parquet file with streaming."""
        # Use Polars for efficient streaming
        df_stream = pl.scan_parquet(str(input_path))
        
        # Get total rows for progress tracking (use pl.len for newer Polars)
        total_rows = df_stream.select(pl.len()).collect().item()
        if self._should_use_fast_mode(total_rows):
            self.progress['total_chunks'] = 1
            return self._clean_in_memory(
                input_path,
                output_path,
                domain,
                loader=lambda: pd.read_parquet(input_path),
                format_hint="parquet"
            )
        self.progress['total_chunks'] = (total_rows + self.config.chunk_size - 1) // self.config.chunk_size
        
        # Process in chunks
        cleaned_chunks = []
        quarantined_chunks = []
        all_audit_logs = []
        
        for chunk_df in self._stream_dataframe_chunks(df_stream):
            cleaned_chunk, quarantined_chunk, audit_log = self._process_chunk(chunk_df, domain)
            
            if cleaned_chunk is not None:
                cleaned_chunks.append(cleaned_chunk)
            if quarantined_chunk is not None and len(quarantined_chunk) > 0:
                quarantined_chunks.append(quarantined_chunk)
            if audit_log:
                all_audit_logs.append(audit_log)
            
            # Memory management
            if self._check_memory_usage():
                gc.collect()
        
        # Combine results
        return self._combine_results(cleaned_chunks, quarantined_chunks, all_audit_logs, output_path, domain)

    def _clean_in_memory(
        self,
        input_path: Path,
        output_path: Path,
        domain: Optional[str],
        loader: Callable[[], pd.DataFrame],
        format_hint: str,
    ) -> CleaningResult:
        """Fast-path cleaning for datasets that fit comfortably in memory."""
        try:
            df = loader()
        except Exception as exc:
            logger.error(f"Failed to load {format_hint} file in fast mode: {exc}")
            return CleaningResult(
                cleaned_data_path="",
                quarantined_data_path=None,
                audit_log_path="",
                quality_report_path=None,
                dqr_path=None,
                total_rows_processed=0,
                total_rows_cleaned=0,
                total_rows_quarantined=0,
                processing_time=0,
                success=False,
                error_message=str(exc)
            )

        self.progress['total_chunks'] = 1
        self.progress['chunks_processed'] = 1
        self.progress['rows_processed'] = len(df)

        pipeline_log: Optional[Dict[str, Any]] = None
        pipeline_stats: Optional[Dict[str, Any]] = None
        ml_ready_df: Optional[pd.DataFrame] = None
        working_df = df

        if self.config.use_fast_pipeline:
            raw_tidy_df, ml_ready_df, pipeline_stats = self.pipeline.preprocess_dual_output(df)
            pipeline_log = self.pipeline.get_cleaning_log()
            working_df = raw_tidy_df

        cleaned_df, quarantined_df, audit_log = self.policy.clean_dataframe(working_df)

        if self.config.enable_quality_gates:
            quality_report = self.quality_gates.validate_data(cleaned_df, domain)
            if audit_log:
                audit_log.quality_score = getattr(quality_report, "quality_score", None)
                audit_log.quarantined_rows += len(getattr(quality_report, "quarantine_rows", []))
            quarantine_rows = getattr(quality_report, "quarantine_rows", [])
            if quarantine_rows:
                additional_quarantine = cleaned_df.iloc[quarantine_rows]
                if quarantined_df is None:
                    quarantined_df = additional_quarantine.reset_index(drop=True)
                else:
                    quarantined_df = pd.concat([quarantined_df, additional_quarantine], ignore_index=True)
                cleaned_df = cleaned_df.drop(index=quarantine_rows).reset_index(drop=True)

        self.progress['rows_cleaned'] = len(cleaned_df)
        if quarantined_df is not None:
            self.progress['rows_quarantined'] = len(quarantined_df)

        profile_summary = analyze_data_quality(cleaned_df)

        return self._combine_results(
            cleaned_chunks=[cleaned_df],
            quarantined_chunks=[quarantined_df] if quarantined_df is not None and len(quarantined_df) > 0 else [],
            audit_logs=[audit_log] if audit_log else [],
            output_path=output_path,
            domain=domain,
            ml_ready_df=ml_ready_df if self.config.save_ml_ready_output else None,
            pipeline_log=pipeline_log,
            pipeline_stats=pipeline_stats,
            raw_profile=profile_summary
        )

    def _stream_dataframe_chunks(self, df_stream: pl.LazyFrame) -> Iterator[pd.DataFrame]:
        """Stream dataframe in chunks for memory efficiency."""
        offset = 0
        
        while True:
            chunk = df_stream.slice(offset, self.config.chunk_size).collect()
            
            if len(chunk) == 0:
                break
            
            # Convert to pandas for processing
            chunk_pandas = chunk.to_pandas()
            yield chunk_pandas
            
            offset += self.config.chunk_size
            self.progress['chunks_processed'] += 1
            self.progress['rows_processed'] += len(chunk_pandas)

    def _process_chunk(self, chunk_df: pd.DataFrame, domain: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[AuditLog]]:
        """Process a single chunk of data."""
        try:
            # Apply policy-driven cleaning
            cleaned_df, quarantined_df, audit_log = self.policy.clean_dataframe(chunk_df)
            
            # Apply quality gates if enabled
            if self.config.enable_quality_gates:
                quality_report = self.quality_gates.validate_data(cleaned_df, domain)
                
                # Update audit log with quality information
                if audit_log:
                    audit_log.quality_score = quality_report.quality_score
                    audit_log.quarantined_rows += len(quality_report.quarantine_rows)
            
            # Update progress
            self.progress['rows_cleaned'] += len(cleaned_df)
            if quarantined_df is not None:
                self.progress['rows_quarantined'] += len(quarantined_df)
            
            return cleaned_df, quarantined_df, audit_log
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return None, None, None

    def _combine_results(
        self,
        cleaned_chunks: List[pd.DataFrame],
        quarantined_chunks: List[pd.DataFrame],
        audit_logs: List[AuditLog],
        output_path: Path,
        domain: Optional[str],
        ml_ready_df: Optional[pd.DataFrame] = None,
        pipeline_log: Optional[Dict[str, Any]] = None,
        pipeline_stats: Optional[Dict[str, Any]] = None,
        raw_profile: Optional[Dict[str, Any]] = None,
    ) -> CleaningResult:
        """Combine chunk results into final outputs."""
        try:
            # Combine cleaned data
            if cleaned_chunks:
                final_cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
            else:
                final_cleaned_df = pd.DataFrame()
            
            # Combine quarantined data
            if quarantined_chunks:
                final_quarantined_df = pd.concat(quarantined_chunks, ignore_index=True)
            else:
                final_quarantined_df = pd.DataFrame()

            ml_ready_data_path: Optional[Path] = None
            data_profile_path: Optional[Path] = None
            pipeline_log_path: Optional[Path] = None
            
            # Save cleaned data with robust fallbacks
            cleaned_data_path = output_path / 'cleaned_data.parquet'
            csv_path = output_path / 'cleaned_data.csv'
            wrote_any = False
            # Try Parquet if requested
            if self.config.output_format in ['parquet', 'both']:
                try:
                    final_cleaned_df.to_parquet(cleaned_data_path, index=False)
                    wrote_any = True
                except Exception as e:
                    logger.warning(f"Failed to write Parquet, falling back to CSV: {e}")
            # Try CSV if requested or Parquet failed
            if (self.config.output_format in ['csv', 'both']) or (not wrote_any):
                try:
                    final_cleaned_df.to_csv(csv_path, index=False)
                    cleaned_data_path = csv_path
                    wrote_any = True
                except Exception as e:
                    logger.error(f"Failed to write cleaned data to CSV: {e}")
            
            # Save quarantined data
            quarantined_data_path = None
            if len(final_quarantined_df) > 0:
                # Prefer Parquet, fallback to CSV
                q_parquet = output_path / 'quarantined_data.parquet'
                q_csv = output_path / 'quarantined_data.csv'
                try:
                    final_quarantined_df.to_parquet(q_parquet, index=False)
                    quarantined_data_path = q_parquet
                except Exception:
                    final_quarantined_df.to_csv(q_csv, index=False)
                    quarantined_data_path = q_csv

            # Optional profile and ML-ready export
            if raw_profile is None and len(final_cleaned_df) > 0:
                try:
                    raw_profile = analyze_data_quality(final_cleaned_df)
                except Exception as profile_exc:
                    logger.warning(f"Failed to analyze data quality: {profile_exc}")
                    raw_profile = None

            if raw_profile:
                try:
                    profile_path = output_path / 'data_profile.json'
                    with open(profile_path, 'w') as f:
                        json.dump(raw_profile, f, indent=2, default=str)
                    data_profile_path = profile_path
                except Exception as profile_write_exc:
                    logger.warning(f"Failed to write data profile: {profile_write_exc}")
                    data_profile_path = None

            if self.config.save_ml_ready_output and len(final_cleaned_df) > 0:
                ml_ready_candidate = ml_ready_df
                if ml_ready_candidate is None and self.config.use_fast_pipeline and len(final_cleaned_df) <= self.config.fast_mode_threshold_rows:
                    try:
                        _, ml_ready_candidate, pipeline_stats = self.pipeline.preprocess_dual_output(final_cleaned_df.copy())
                        if pipeline_log is None:
                            pipeline_log = self.pipeline.get_cleaning_log()
                    except Exception as ml_exc:
                        logger.warning(f"Failed to generate ML-ready dataset: {ml_exc}")
                        ml_ready_candidate = None
                if ml_ready_candidate is not None and len(ml_ready_candidate) > 0:
                    ml_ready_path = output_path / 'ml_ready_data.parquet'
                    try:
                        ml_ready_candidate.to_parquet(ml_ready_path, index=False)
                    except Exception:
                        ml_ready_path = output_path / 'ml_ready_data.csv'
                        try:
                            ml_ready_candidate.to_csv(ml_ready_path, index=False)
                        except Exception as ml_write_exc:
                            logger.warning(f"Failed to write ML-ready dataset: {ml_write_exc}")
                            ml_ready_path = None
                    if ml_ready_path:
                        ml_ready_data_path = ml_ready_path

            if pipeline_log is not None or pipeline_stats is not None:
                pipeline_payload: Dict[str, Any] = {}
                if pipeline_log:
                    pipeline_payload.update(pipeline_log)
                if pipeline_stats:
                    existing_stats = pipeline_payload.get("statistics")
                    if not existing_stats:
                        pipeline_payload["statistics"] = pipeline_stats
                pipeline_log_path = output_path / 'pipeline_log.json'
                try:
                    with open(pipeline_log_path, 'w') as f:
                        json.dump(pipeline_payload, f, indent=2, default=str)
                except Exception as pipeline_exc:
                    logger.warning(f"Failed to write pipeline log: {pipeline_exc}")
                    pipeline_log_path = None
            
            # Combine audit logs
            combined_audit_log = self._combine_audit_logs(audit_logs)
            # Update shapes from final outputs for accurate reporting
            combined_audit_log.input_shape = (self.progress['rows_processed'], len(final_cleaned_df.columns) if len(final_cleaned_df.columns) > 0 else 0)
            combined_audit_log.output_shape = final_cleaned_df.shape
            combined_audit_log.domain_inferred = domain

            # Generate quality report (before exporting audit so we can include quality_score)
            quality_report_path = None
            dqr_path = None
            if self.config.enable_quality_gates and len(final_cleaned_df) > 0:
                quality_report = self.quality_gates.validate_data(final_cleaned_df, domain)
                # Attach quality score to audit
                try:
                    combined_audit_log.quality_score = getattr(quality_report, 'quality_score', None)
                except Exception:
                    combined_audit_log.quality_score = None

                quality_report_path = output_path / 'quality_report.json'
                try:
                    with open(quality_report_path, 'w') as f:
                        json.dump(self.quality_gates.get_quality_metrics(), f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"Failed to write quality report JSON: {e}")
                    quality_report_path = None
                
                dqr_path = output_path / 'dqr.md'
                try:
                    self.quality_gates.generate_dqr(str(dqr_path))
                except Exception as e:
                    logger.warning(f"Failed to write DQR markdown: {e}")
                    dqr_path = None
            elif self.config.enable_quality_gates:
                # Dataset empty – emit minimal quality artifacts so UI has something to display
                try:
                    from .quality_gates import QualityReport
                    empty_report = QualityReport(
                        timestamp=datetime.now(),
                        total_rows=0,
                        total_columns=0,
                        violations=[],
                        quarantine_rows=[],
                        auto_fixes_applied=[],
                        quality_score=0.0,
                        domain_inferred=domain
                    )
                    self.quality_gates.quality_report = empty_report
                    quality_report_path = output_path / 'quality_report.json'
                    with open(quality_report_path, 'w') as f:
                        json.dump(self.quality_gates.get_quality_metrics(), f, indent=2, default=str)
                    dqr_path = output_path / 'dqr.md'
                    self.quality_gates.generate_dqr(str(dqr_path))
                    # Attach score to audit
                    combined_audit_log.quality_score = 0.0
                except Exception as e:
                    logger.warning(f"Failed to create minimal quality report: {e}")
                    quality_report_path = None
                    dqr_path = None

            # Export audit log after enriching it
            audit_log_path = output_path / 'audit_log.json'
            combined_audit_log.export_audit_log(str(audit_log_path))
            
            # Calculate processing time
            processing_time = (datetime.now() - self.progress['start_time']).total_seconds()
            
            return CleaningResult(
                cleaned_data_path=str(cleaned_data_path),
                quarantined_data_path=str(quarantined_data_path) if quarantined_data_path else None,
                audit_log_path=str(audit_log_path),
                quality_report_path=str(quality_report_path) if quality_report_path else None,
                dqr_path=str(dqr_path) if dqr_path else None,
                total_rows_processed=self.progress['rows_processed'],
                total_rows_cleaned=self.progress['rows_cleaned'],
                total_rows_quarantined=self.progress['rows_quarantined'],
                processing_time=processing_time,
                success=True,
                ml_ready_data_path=str(ml_ready_data_path) if ml_ready_data_path else None,
                data_profile_path=str(data_profile_path) if data_profile_path else None,
                pipeline_log_path=str(pipeline_log_path) if pipeline_log_path else None
            )
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return CleaningResult(
                cleaned_data_path="",
                quarantined_data_path=None,
                audit_log_path="",
                quality_report_path=None,
                dqr_path=None,
                total_rows_processed=self.progress['rows_processed'],
                total_rows_cleaned=0,
                total_rows_quarantined=0,
                processing_time=0,
                success=False,
                error_message=str(e)
            )

    def _combine_audit_logs(self, audit_logs: List[AuditLog]) -> AuditLog:
        """Combine multiple audit logs into a single comprehensive log."""
        if not audit_logs:
            return AuditLog(
                timestamp=datetime.now(),
                input_shape=(0, 0),
                output_shape=(0, 0),
                decisions=[]
            )
        
        # Use the first log as base
        combined_log = audit_logs[0]
        
        # Combine all decisions
        all_decisions = []
        for log in audit_logs:
            all_decisions.extend(log.decisions)
        
        combined_log.decisions = all_decisions
        combined_log.quarantined_rows = sum(log.quarantined_rows for log in audit_logs)
        combined_log.missing_fixed = sum(log.missing_fixed for log in audit_logs)
        combined_log.outliers_handled = sum(log.outliers_handled for log in audit_logs)
        combined_log.merges_performed = sum(log.merges_performed for log in audit_logs)
        
        return combined_log

    def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb > self.config.max_memory_mb
        except ImportError:
            # If psutil not available, assume memory is fine
            return False

    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        if self.progress['start_time']:
            elapsed_time = (datetime.now() - self.progress['start_time']).total_seconds()
        else:
            elapsed_time = 0
        
        progress_pct = 0
        if self.progress['total_chunks'] > 0:
            progress_pct = (self.progress['chunks_processed'] / self.progress['total_chunks']) * 100
        
        return {
            'chunks_processed': self.progress['chunks_processed'],
            'total_chunks': self.progress['total_chunks'],
            'progress_percentage': progress_pct,
            'rows_processed': self.progress['rows_processed'],
            'rows_cleaned': self.progress['rows_cleaned'],
            'rows_quarantined': self.progress['rows_quarantined'],
            'elapsed_time': elapsed_time,
            'estimated_remaining_time': self._estimate_remaining_time()
        }

    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining processing time."""
        if self.progress['chunks_processed'] == 0 or not self.progress['start_time']:
            return None
        
        elapsed_time = (datetime.now() - self.progress['start_time']).total_seconds()
        chunks_per_second = self.progress['chunks_processed'] / elapsed_time
        remaining_chunks = self.progress['total_chunks'] - self.progress['chunks_processed']
        
        return remaining_chunks / chunks_per_second if chunks_per_second > 0 else None

    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_temp_files()
