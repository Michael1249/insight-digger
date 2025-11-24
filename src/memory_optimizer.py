# -*- coding: utf-8 -*-
"""
Memory Usage Optimization Module for Insight Digger EFA Toolkit

Provides comprehensive memory management and optimization capabilities.

Author: Insight Digger Project
Created: November 24, 2025
"""

import gc
import os
import sys
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available - memory monitoring will be limited")


class MemoryMonitor:
    """Monitor and manage memory usage during EFA operations."""
    
    def __init__(self):
        """Initialize memory monitor."""
        self.peak_memory_mb = 0.0
        self.baseline_memory_mb = self._get_memory_usage()
        self.memory_history = []
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 ** 2)
            except Exception:
                pass
        
        # Fallback method
        try:
            return sum(sys.getsizeof(obj) for obj in gc.get_objects()) / (1024 ** 2)
        except Exception:
            return 0.0
    
    def record_memory_usage(self, operation: str = "operation") -> float:
        """Record current memory usage with operation label."""
        current_memory = self._get_memory_usage()
        self.memory_history.append({
            'operation': operation,
            'memory_mb': current_memory,
            'delta_from_baseline': current_memory - self.baseline_memory_mb
        })
        
        if current_memory > self.peak_memory_mb:
            self.peak_memory_mb = current_memory
            
        return current_memory
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        current_memory = self._get_memory_usage()
        
        return {
            'current_memory_mb': round(current_memory, 2),
            'baseline_memory_mb': round(self.baseline_memory_mb, 2),
            'peak_memory_mb': round(self.peak_memory_mb, 2),
            'memory_increase_mb': round(current_memory - self.baseline_memory_mb, 2),
            'peak_increase_mb': round(self.peak_memory_mb - self.baseline_memory_mb, 2),
            'n_measurements': len(self.memory_history)
        }
    
    def optimize_memory(self) -> float:
        """Perform memory optimization and garbage collection."""
        gc.collect()
        memory_after = self.record_memory_usage("memory_optimization")
        return memory_after


class ChunkedProcessor:
    """Process large datasets in memory-efficient chunks."""
    
    def __init__(self, memory_limit_mb: float = 500.0):
        """Initialize chunked processor."""
        self.memory_limit_mb = memory_limit_mb
        self.monitor = MemoryMonitor()
        
    def calculate_optimal_chunk_size(self, data_shape: Tuple[int, int]) -> int:
        """Calculate optimal chunk size based on data shape and memory constraints."""
        n_obs, n_vars = data_shape
        
        # Estimate memory per observation (in MB)
        memory_per_obs = (n_vars * 8) / (1024 ** 2)  # 8 bytes per float64
        
        # Calculate chunk size to stay within memory limit
        target_memory_per_chunk = self.memory_limit_mb * 0.8  # Use 80% of limit
        optimal_chunk_size = int(target_memory_per_chunk / memory_per_obs)
        
        # Ensure reasonable bounds
        optimal_chunk_size = max(100, min(optimal_chunk_size, n_obs))
        
        return optimal_chunk_size


class MemoryEfficientEFA:
    """Memory-efficient implementations of EFA algorithms."""
    
    def __init__(self, memory_limit_mb: float = 500.0):
        """Initialize memory-efficient EFA processor."""
        self.memory_limit_mb = memory_limit_mb
        self.processor = ChunkedProcessor(memory_limit_mb=memory_limit_mb)
        self.monitor = MemoryMonitor()
        
    def estimate_memory_requirements(self, data_shape: Tuple[int, int], 
                                   n_factors: int) -> Dict[str, float]:
        """Estimate memory requirements for EFA analysis."""
        n_obs, n_vars = data_shape
        
        estimates = {
            'data_storage_mb': (n_obs * n_vars * 8) / (1024 ** 2),
            'correlation_matrix_mb': (n_vars * n_vars * 8) / (1024 ** 2),
            'factor_loadings_mb': (n_vars * n_factors * 8) / (1024 ** 2),
            'working_memory_mb': (n_vars * n_vars * 8 * 2) / (1024 ** 2),
        }
        
        estimates['total_estimated_mb'] = sum(estimates.values())
        estimates['recommended_available_mb'] = estimates['total_estimated_mb'] * 1.5
        
        return estimates
    
    def memory_efficient_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with memory optimization."""
        n_obs, n_vars = data.shape
        
        # For small datasets, use standard method
        if n_obs * n_vars * 8 < self.memory_limit_mb * 1024 ** 2:
            return data.corr()
        
        # Use chunked processing for large datasets
        warnings.warn("Large dataset detected. Using chunked correlation calculation.")
        
        # Simplified chunked correlation (basic implementation)
        chunk_size = self.processor.calculate_optimal_chunk_size(data.shape)
        
        # Process data in chunks and accumulate correlations
        correlation_sum = np.zeros((n_vars, n_vars))
        n_chunks = (n_obs + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_obs)
            
            chunk = data.iloc[start_idx:end_idx]
            chunk_corr = chunk.corr()
            
            # Simple averaging (can be improved with proper statistical accumulation)
            correlation_sum += chunk_corr.fillna(0).values / n_chunks
        
        # Ensure diagonal is 1.0
        np.fill_diagonal(correlation_sum, 1.0)
        
        return pd.DataFrame(correlation_sum, index=data.columns, columns=data.columns)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    monitor = MemoryMonitor()
    return monitor.get_memory_summary()


def optimize_pandas_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize pandas DataFrame memory usage."""
    optimized_df = df.copy()
    
    # Optimize numeric columns
    for col in optimized_df.select_dtypes(include=[np.number]).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        
        # Float optimization
        if optimized_df[col].dtype == 'float64':
            if (optimized_df[col] == optimized_df[col].astype(np.float32)).all():
                optimized_df[col] = optimized_df[col].astype(np.float32)
    
    return optimized_df