# -*- coding: utf-8 -*-
"""
EFA Error Handling and Warnings System

Centralized error handling, warning management, and logging utilities
for the EFA analysis module. Provides consistent error reporting and
user-friendly warning messages.

Author: Insight Digger Project
Created: November 24, 2025
"""

import warnings
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class WarningLevel(Enum):
    """Warning severity levels for EFA operations."""
    INFO = "INFO"
    WARNING = "WARNING" 
    CRITICAL = "CRITICAL"


class EFAWarningManager:
    """
    Centralized warning management for EFA operations.
    
    Handles warning collection, filtering, and reporting with different
    severity levels and user-friendly messaging.
    """
    
    def __init__(self, show_warnings: bool = True, warning_level: str = "WARNING"):
        self.show_warnings = show_warnings
        self.warning_level = WarningLevel(warning_level)
        self.warnings_collected = []
        
        # Setup logging
        self.logger = logging.getLogger("EFA")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def add_warning(self, message: str, level: WarningLevel = WarningLevel.WARNING,
                   category: str = "General", details: Optional[Dict] = None):
        """
        Add warning to collection with metadata.
        
        Args:
            message: Warning message
            level: Severity level
            category: Warning category (e.g., "Data", "Statistical", "Computational")
            details: Additional details dictionary
        """
        warning_entry = {
            'message': message,
            'level': level,
            'category': category,
            'details': details or {}
        }
        
        self.warnings_collected.append(warning_entry)
        
        # Emit warning if appropriate
        if self.show_warnings and self._should_show_warning(level):
            if level == WarningLevel.CRITICAL:
                self.logger.error(f"[{category}] {message}")
            elif level == WarningLevel.WARNING:
                self.logger.warning(f"[{category}] {message}")
            else:
                self.logger.info(f"[{category}] {message}")
    
    def _should_show_warning(self, level: WarningLevel) -> bool:
        """Check if warning should be displayed based on current settings."""
        level_hierarchy = {
            WarningLevel.INFO: 0,
            WarningLevel.WARNING: 1, 
            WarningLevel.CRITICAL: 2
        }
        return level_hierarchy[level] >= level_hierarchy[self.warning_level]
    
    def get_warnings(self, level: Optional[WarningLevel] = None, 
                    category: Optional[str] = None) -> List[Dict]:
        """
        Retrieve warnings with optional filtering.
        
        Args:
            level: Filter by warning level
            category: Filter by category
            
        Returns:
            List of warning dictionaries
        """
        filtered = self.warnings_collected
        
        if level:
            filtered = [w for w in filtered if w['level'] == level]
        if category:
            filtered = [w for w in filtered if w['category'] == category]
            
        return filtered
    
    def clear_warnings(self):
        """Clear all collected warnings."""
        self.warnings_collected = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of warnings by level and category."""
        summary = {
            'total': len(self.warnings_collected),
            'by_level': {},
            'by_category': {}
        }
        
        for warning in self.warnings_collected:
            level = warning['level'].value
            category = warning['category']
            
            summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
        return summary


# Global warning manager instance
_warning_manager = EFAWarningManager()

def get_warning_manager() -> EFAWarningManager:
    """Get the global warning manager instance."""
    return _warning_manager

def configure_warnings(show_warnings: bool = True, level: str = "WARNING"):
    """Configure global warning settings."""
    global _warning_manager
    _warning_manager.show_warnings = show_warnings
    _warning_manager.warning_level = WarningLevel(level)

def add_efa_warning(message: str, level: str = "WARNING", category: str = "General",
                   details: Optional[Dict] = None):
    """Convenience function to add EFA warning."""
    _warning_manager.add_warning(message, WarningLevel(level), category, details)


# Custom exception hierarchy for EFA
class EFAError(Exception):
    """Base exception for EFA-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}
        
        # Log the error
        logger = logging.getLogger("EFA")
        logger.error(f"EFAError: {message}")
        if self.details:
            logger.error(f"Error details: {self.details}")

class DataValidationError(EFAError):
    """Exception for data validation failures."""
    pass

class ConvergenceError(EFAError):
    """Exception for algorithm convergence failures."""
    pass

class FactorabilityError(EFAError):
    """Exception for factorability assessment failures."""
    pass

class ComputationError(EFAError):
    """Exception for computational errors in factor analysis."""
    pass


# Error handling decorators
def handle_efa_errors(func):
    """Decorator to handle common EFA errors with user-friendly messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except np.linalg.LinAlgError as e:
            raise ComputationError(
                "Matrix computation failed - data may be singular or ill-conditioned",
                details={'original_error': str(e), 'function': func.__name__}
            )
        except ValueError as e:
            if "singular matrix" in str(e).lower():
                raise ComputationError(
                    "Singular correlation matrix - check for perfect multicollinearity",
                    details={'original_error': str(e), 'function': func.__name__}
                )
            else:
                raise DataValidationError(
                    f"Data validation error: {str(e)}",
                    details={'original_error': str(e), 'function': func.__name__}
                )
        except Exception as e:
            raise EFAError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={'original_error': str(e), 'function': func.__name__}
            )
    return wrapper


# Utility functions for error reporting
def format_error_message(error: Exception, include_details: bool = False) -> str:
    """Format error message for user display."""
    message = str(error)
    
    if isinstance(error, EFAError) and include_details and error.details:
        details_str = ", ".join([f"{k}: {v}" for k, v in error.details.items()])
        message += f" (Details: {details_str})"
        
    return message

def check_data_warnings(data, warning_manager: Optional[EFAWarningManager] = None) -> bool:
    """
    Perform common data checks and add warnings.
    
    Returns:
        bool: True if data passes basic checks
    """
    if warning_manager is None:
        warning_manager = get_warning_manager()
        
    passed = True
    
    # Check for missing data
    missing_pct = (data.isnull().sum().sum() / data.size) * 100
    if missing_pct > 0:
        level = WarningLevel.CRITICAL if missing_pct > 50 else WarningLevel.WARNING
        warning_manager.add_warning(
            f"Missing data detected: {missing_pct:.1f}% of values",
            level=level,
            category="Data"
        )
        if missing_pct > 50:
            passed = False
    
    # Check for zero variance
    zero_var = data.var() == 0
    if zero_var.any():
        warning_manager.add_warning(
            f"Zero variance variables detected: {zero_var.sum()} variables",
            level=WarningLevel.WARNING,
            category="Data"
        )
    
    return passed

# Import numpy for LinAlgError
import numpy as np