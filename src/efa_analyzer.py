# -*- coding: utf-8 -*-
"""
Exploratory Factor Analysis (EFA) Analyzer Module

This module provides comprehensive EFA capabilities for discovering hidden psychological 
dimensions in survey data. Implements principal axis factoring with rotation options,
factor score computation, and statistical validation.

Author: Insight Digger Project
Created: November 24, 2025
"""

import warnings
from typing import Optional, Union, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    warnings.warn("SciPy not available - some statistical functions will be limited")

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False
    FactorAnalyzer = None
    calculate_bartlett_sphericity = None
    calculate_kmo = None
    warnings.warn("factor_analyzer not available - installing dependencies required")


class ValidationResults:
    """Container for data validation results."""
    
    def __init__(self, is_valid: bool = True, warnings: List[str] = None, 
                 errors: List[str] = None, kmo_score: float = None,
                 bartlett_p: float = None):
        self.is_valid = is_valid
        self.warnings = warnings or []
        self.errors = errors or []
        self.kmo_score = kmo_score
        self.bartlett_p = bartlett_p


class FactorSolution:
    """Container for complete EFA results."""
    
    def __init__(self, loadings: pd.DataFrame = None, communalities: pd.Series = None,
                 eigenvalues: np.ndarray = None, variance_explained: Dict = None,
                 factor_scores: pd.DataFrame = None, rotation_method: str = None,
                 extraction_method: str = None, n_factors: int = None):
        self.loadings = loadings
        self.communalities = communalities  
        self.eigenvalues = eigenvalues
        self.variance_explained = variance_explained
        self.factor_scores = factor_scores
        self.rotation_method = rotation_method
        self.extraction_method = extraction_method
        self.n_factors = n_factors


class EFAAnalyzer:
    """
    Exploratory Factor Analysis Analyzer
    
    Provides comprehensive EFA capabilities including correlation matrix calculation,
    factor extraction using multiple methods, rotation options, and factor score computation.
    
    Attributes:
        n_factors (int): Number of factors to extract
        extraction_method (str): Factor extraction method ('principal', 'ml', 'minres')
        rotation_method (str): Rotation method ('oblimin', 'varimax', 'quartimax')
        convergence_tolerance (float): Convergence criterion for iterative algorithms
        max_iterations (int): Maximum iterations for convergence
    """
    
    def __init__(self, 
                 n_factors: Optional[int] = None,
                 extraction_method: str = 'principal',
                 rotation_method: str = 'oblimin', 
                 convergence_tolerance: float = 1e-4,
                 max_iterations: int = 100):
        """
        Initialize EFA Analyzer.
        
        Args:
            n_factors: Number of factors to extract. If None, determined automatically.
            extraction_method: Factor extraction method ('principal', 'ml', 'minres')
            rotation_method: Factor rotation method ('oblimin', 'varimax', 'quartimax')
            convergence_tolerance: Convergence criterion for iterative algorithms
            max_iterations: Maximum iterations for convergence
            
        Raises:
            ValueError: Invalid method names or parameter ranges
            TypeError: Invalid parameter types
        """
        self.n_factors = n_factors
        self.extraction_method = extraction_method
        self.rotation_method = rotation_method
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize internal state
        self.fit_completed = False
        self.correlation_matrix = None
        self.factor_analyzer = None
        self.validation_results = None
        
    def _validate_parameters(self):
        """Enhanced parameter validation with specific error messages."""
        valid_extraction = ['principal', 'ml', 'minres']
        valid_rotation = ['oblimin', 'varimax', 'quartimax', 'promax']
        
        # Enhanced extraction method validation
        if self.extraction_method not in valid_extraction:
            available = "', '".join(valid_extraction)
            raise ValueError(f"extraction_method '{self.extraction_method}' not supported. "
                           f"Available: ['{available}']")
            
        # Enhanced rotation method validation  
        if self.rotation_method not in valid_rotation:
            available = "', '".join(valid_rotation)
            raise ValueError(f"rotation_method '{self.rotation_method}' not supported. "
                           f"Available: ['{available}']")
            
        # Enhanced numerical parameter validation
        if not isinstance(self.convergence_tolerance, (int, float)):
            raise TypeError(f"convergence_tolerance must be numeric, got {type(self.convergence_tolerance).__name__}")
        if self.convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance must be positive, got {self.convergence_tolerance}")
        if self.convergence_tolerance > 0.1:
            warnings.warn(f"convergence_tolerance very large ({self.convergence_tolerance}), "
                         f"may cause premature convergence")
            
        if not isinstance(self.max_iterations, int):
            raise TypeError(f"max_iterations must be int, got {type(self.max_iterations).__name__}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.max_iterations > 10000:
            warnings.warn(f"max_iterations very large ({self.max_iterations}), "
                         f"may cause performance issues")
    
    def validate_data(self, data: pd.DataFrame) -> ValidationResults:
        """
        Validate input data for factor analysis suitability.
        
        Args:
            data: Input data matrix (observations × variables)
            
        Returns:
            ValidationResults: Comprehensive validation results
            
        Raises:
            TypeError: Invalid data type
            ValueError: Insufficient data for analysis
        """
        warnings_list = []
        errors_list = []
        is_valid = True
        
        # Type validation with detailed error message
        if not isinstance(data, pd.DataFrame):
            actual_type = type(data).__name__
            raise TypeError(f"Data must be pandas DataFrame, got {actual_type}. "
                          f"Convert your data using pd.DataFrame(your_data).")
            
        # Empty data check
        if data.empty:
            raise ValueError("Data is empty. Provide a dataset with observations and variables.")
            
        # Shape validation with detailed recommendations
        n_obs, n_vars = data.shape
        if n_obs < 3:
            raise ValueError(f"Insufficient observations: got {n_obs}, need ≥3. "
                           f"Provide more data points for meaningful analysis.")
        if n_vars < 3:
            raise ValueError(f"Insufficient variables: got {n_vars}, need ≥3. "
                           f"Factor analysis requires multiple variables to identify factors.")
        
        # Advanced shape validation
        if n_obs < n_vars:
            warnings_list.append(f"More variables ({n_vars}) than observations ({n_obs}). "
                                f"Results may be unstable - consider dimensionality reduction.")
            
        # Sample size adequacy (general rule: N ≥ 5*variables, minimum 50)
        min_recommended = max(50, 5 * n_vars)
        if n_obs < min_recommended:
            warnings_list.append(f"Small sample: {n_obs} observations for {n_vars} variables. Recommended: ≥{min_recommended}")
            
        # Factor count validation if specified
        max_factors = min(n_vars // 3, n_obs - 1)
        if self.n_factors is not None:
            if self.n_factors > max_factors:
                errors_list.append(f"Too many factors: {self.n_factors} requested, maximum {max_factors}")
                is_valid = False
                
        # Data type validation
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] != n_vars:
            errors_list.append("All variables must be numeric")
            is_valid = False
            
        # Missing data assessment
        missing_pct = (data.isnull().sum().sum() / (n_obs * n_vars)) * 100
        if missing_pct > 50:
            errors_list.append(f"Excessive missing data: {missing_pct:.1f}%")
            is_valid = False
        elif missing_pct > 20:
            warnings_list.append(f"High missing data: {missing_pct:.1f}%")
            
        # Variance check (zero variance variables)
        zero_var_cols = data.columns[data.var() == 0].tolist()
        if zero_var_cols:
            warnings_list.append(f"Zero variance variables detected: {zero_var_cols}")
            
        # Basic factorability check using available data
        kmo_score = None
        bartlett_p = None
        
        if FACTOR_ANALYZER_AVAILABLE:
            try:
                # Quick KMO and Bartlett's test
                numeric_clean = numeric_data.dropna()
                if numeric_clean.shape[0] > 10 and numeric_clean.shape[1] > 2:
                    kmo_overall, kmo_individual = calculate_kmo(numeric_clean)
                    kmo_score = kmo_overall
                    
                    bartlett_stat, bartlett_p = calculate_bartlett_sphericity(numeric_clean)
                    
                    if kmo_score < 0.5:
                        warnings_list.append(f"Poor KMO measure: {kmo_score:.3f} (recommend > 0.6)")
                    if bartlett_p > 0.05:
                        warnings_list.append(f"Bartlett's test non-significant: p={bartlett_p:.3f}")
                        
            except Exception as e:
                warnings_list.append(f"Could not perform initial factorability tests: {str(e)}")
        else:
            warnings_list.append("factor_analyzer not available - install dependencies for full validation")
            
        return ValidationResults(
            is_valid=is_valid,
            warnings=warnings_list, 
            errors=errors_list,
            kmo_score=kmo_score,
            bartlett_p=bartlett_p
        )
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                   method: str = 'pearson',
                                   chunk_size: int = None) -> pd.DataFrame:
        """
        Calculate correlation matrix using specified method with memory optimization.
        
        Args:
            data: Input data matrix (observations × variables)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            chunk_size: Process data in chunks for large datasets (None for auto)
            
        Returns:
            Correlation matrix (variables × variables)
            
        Raises:
            ValueError: Invalid correlation method or data issues
        """
        valid_methods = ['pearson', 'spearman', 'kendall']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        n_obs, n_vars = data.shape
        
        # Memory optimization: use chunking for large datasets
        memory_threshold_mb = 500  # Start chunking above 500MB
        estimated_memory_mb = (n_obs * n_vars * 8) / (1024 ** 2)  # 8 bytes per float64
        
        if chunk_size is None and estimated_memory_mb > memory_threshold_mb:
            # Calculate optimal chunk size
            target_chunk_mb = 100
            chunk_size = max(1000, int((target_chunk_mb / estimated_memory_mb) * n_obs))
            warnings.warn(f"Large dataset detected ({estimated_memory_mb:.1f}MB). "
                         f"Using chunk size {chunk_size} for memory efficiency.")
            
        # Handle missing data with pairwise deletion (FR-014)
        try:
            if chunk_size is None or n_obs < 5000:
                # Standard calculation for small to medium datasets
                if method == 'pearson':
                    corr_matrix = data.corr(method='pearson', min_periods=1)
                else:
                    corr_matrix = data.corr(method=method, min_periods=1)
            else:
                # Chunked calculation for large datasets
                warnings.warn(f"Using chunked correlation calculation with chunk size {chunk_size}")
                
                # Initialize correlation matrix
                corr_matrix = pd.DataFrame(
                    np.zeros((n_vars, n_vars)), 
                    index=data.columns, 
                    columns=data.columns
                )
                pair_counts = np.zeros((n_vars, n_vars))
                
                # Process in chunks
                n_chunks = (n_obs + chunk_size - 1) // chunk_size
                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, n_obs)
                    
                    chunk_data = data.iloc[start_idx:end_idx]
                    
                    if method == 'pearson':
                        chunk_corr = chunk_data.corr(method='pearson', min_periods=1)
                    else:
                        chunk_corr = chunk_data.corr(method=method, min_periods=1)
                    
                    # Accumulate correlations (weighted by valid pairs)
                    valid_pairs = (~chunk_data.isnull()).sum().values[:, np.newaxis] * (~chunk_data.isnull()).sum().values
                    
                    # Update running correlation (simplified approach)
                    chunk_weight = len(chunk_data) / n_obs
                    corr_matrix += chunk_corr.fillna(0) * chunk_weight
                    
                    if chunk_idx % max(1, n_chunks // 10) == 0 and n_chunks > 10:
                        print(f"Correlation calculation progress: {chunk_idx + 1}/{n_chunks} chunks")
                        
                # Ensure diagonal is 1.0
                np.fill_diagonal(corr_matrix.values, 1.0)
        except Exception as e:
            raise ValueError(f"Failed to compute {method} correlation matrix: {str(e)}. "
                           f"Check for non-numeric data or extreme outliers.") from e
            
        # Comprehensive correlation matrix validation
        if corr_matrix.isnull().any().any():
            null_vars = corr_matrix.columns[corr_matrix.isnull().any()].tolist()
            raise ValueError(f"Correlation matrix contains NaN values for variables: {null_vars}. "
                           f"Check for constant variables or insufficient valid observations.")
            
        # Check matrix properties
        try:
            # Determinant check for singularity
            det = np.linalg.det(corr_matrix.values)
            if abs(det) < 1e-8:
                raise ValueError(f"Correlation matrix is singular (determinant={det:.2e}). "
                               f"Remove perfectly correlated or constant variables.")
                
            # Condition number check
            cond_num = np.linalg.cond(corr_matrix.values)
            if cond_num > 1e12:
                warnings.warn(f"Correlation matrix is ill-conditioned (condition number={cond_num:.2e}). "
                             f"Results may be numerically unstable.")
                             
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Correlation matrix has linear algebra issues: {str(e)}. "
                           f"Check for duplicate or linearly dependent variables.") from e
            
        # Check for perfect correlations (excluding diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        perfect_corrs = np.abs(corr_matrix.values[mask]) >= 0.999
        if perfect_corrs.any():
            # Find which variables have perfect correlations
            row_idx, col_idx = np.where(np.abs(corr_matrix.values) >= 0.999)
            perfect_pairs = [(corr_matrix.index[r], corr_matrix.columns[c]) 
                           for r, c in zip(row_idx, col_idx) if r != c]
            warnings.warn(f"Perfect correlations detected between: {perfect_pairs[:5]}. "
                         f"Consider removing redundant variables to avoid singularity.")
            
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def fit(self, data: pd.DataFrame) -> FactorSolution:
        """
        Perform complete factor analysis on input data.
        
        Args:
            data: Input data matrix (observations × variables)
            
        Returns:
            FactorSolution: Complete factor analysis results
            
        Raises:
            ValueError: Data validation failures or convergence issues
            RuntimeError: Factor analysis computation errors
        """
        # Step 1: Comprehensive data validation
        try:
            validation_results = self.validate_data(data)
        except (TypeError, ValueError) as e:
            # Re-raise validation errors with context
            raise ValueError(f"Data validation failed: {str(e)}") from e
            
        # Check validation results
        if not validation_results.is_valid:
            error_summary = "; ".join(validation_results.errors)
            raise ValueError(f"Data unsuitable for factor analysis: {error_summary}")
            
        # Issue warnings for data quality concerns
        for warning_msg in validation_results.warnings:
            warnings.warn(f"Data quality warning: {warning_msg}", UserWarning)
            
        self.validation_results = validation_results
        
        # Step 2: Calculate correlation matrix (T015)
        corr_matrix = self.calculate_correlation_matrix(data, method='pearson')
        
        # Step 3: Determine number of factors if not specified (T017)
        if self.n_factors is None:
            self.n_factors = self._determine_n_factors(corr_matrix)
            
        # Step 4: Principal axis factoring and rotation (T016, T020)
        if FACTOR_ANALYZER_AVAILABLE:
            solution = self._perform_factor_analysis(data, corr_matrix)
        else:
            solution = self._perform_basic_factor_analysis(data, corr_matrix)
            
        # Step 5: Calculate factor scores (T022) 
        solution.factor_scores = self._calculate_factor_scores(data, solution)
        
        self.fit_completed = True
        return solution
    
    def get_factor_interpretation(self, solution: FactorSolution, 
                                loading_threshold: float = 0.4) -> Dict[str, Any]:
        """
        Generate factor interpretation guidelines based on loading patterns.
        
        Args:
            solution: Factor analysis solution
            loading_threshold: Minimum loading for practical significance
            
        Returns:
            Dictionary containing interpretation guidelines
        """
        if solution.loadings is None:
            raise ValueError("No factor loadings available for interpretation")
            
        interpretation = {
            'loading_threshold': loading_threshold,
            'factor_interpretations': {},
            'variable_assignments': {},
            'structure_summary': {}
        }
        
        # Analyze each factor
        for factor_col in solution.loadings.columns:
            factor_loadings = solution.loadings[factor_col]
            
            # Find significant loadings
            significant_vars = factor_loadings[np.abs(factor_loadings) >= loading_threshold]
            
            # Sort by absolute loading strength
            significant_vars = significant_vars.reindex(
                significant_vars.abs().sort_values(ascending=False).index
            )
            
            interpretation['factor_interpretations'][factor_col] = {
                'high_loading_variables': significant_vars.to_dict(),
                'n_significant': len(significant_vars),
                'max_loading': factor_loadings.abs().max(),
                'interpretation_quality': 'Strong' if len(significant_vars) >= 3 else 'Weak'
            }
            
            # Assign variables to factors (highest absolute loading)
            for var in significant_vars.index:
                if var not in interpretation['variable_assignments']:
                    interpretation['variable_assignments'][var] = factor_col
                    
        # Overall structure assessment
        n_unassigned = len(solution.loadings.index) - len(interpretation['variable_assignments'])
        interpretation['structure_summary'] = {
            'total_variables': len(solution.loadings.index),
            'assigned_variables': len(interpretation['variable_assignments']),
            'unassigned_variables': n_unassigned,
            'simple_structure_quality': 'Good' if n_unassigned <= len(solution.loadings.index) * 0.3 else 'Poor'
        }
        
        return interpretation
    
    def _create_significance_matrix(self, loadings: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """
        Create significance highlighting matrix for factor loadings.
        
        Args:
            loadings: Factor loading matrix
            threshold: Significance threshold
            
        Returns:
            Dictionary with significance levels and highlighting information
        """
        significance = {
            'threshold': threshold,
            'significance_levels': {},
            'highlights': {}
        }
        
        # Define significance levels
        levels = {
            'very_high': 0.8,
            'high': 0.6,
            'moderate': threshold,
            'low': 0.2
        }
        
        significance['significance_levels'] = levels
        
        # Create highlighting matrix
        highlights = {}
        for factor in loadings.columns:
            factor_highlights = {}
            for var in loadings.index:
                loading = loadings.loc[var, factor]
                abs_loading = abs(loading)
                
                if abs_loading >= levels['very_high']:
                    level = 'very_high'
                elif abs_loading >= levels['high']:
                    level = 'high'
                elif abs_loading >= levels['moderate']:
                    level = 'moderate'
                elif abs_loading >= levels['low']:
                    level = 'low'
                else:
                    level = 'negligible'
                
                factor_highlights[var] = {
                    'loading': loading,
                    'abs_loading': abs_loading,
                    'significance_level': level,
                    'is_significant': abs_loading >= threshold
                }
            
            highlights[factor] = factor_highlights
        
        significance['highlights'] = highlights
        return significance
    
    def _assess_interpretation_quality(self, significant_loadings: pd.Series) -> str:
        """Assess the interpretability quality of a factor."""
        n_significant = len(significant_loadings)
        max_loading = significant_loadings.abs().max() if len(significant_loadings) > 0 else 0
        
        if n_significant >= 4 and max_loading >= 0.7:
            return 'Excellent'
        elif n_significant >= 3 and max_loading >= 0.6:
            return 'Good'
        elif n_significant >= 2 and max_loading >= 0.5:
            return 'Adequate'
        elif n_significant >= 1:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _get_dominant_variables(self, factor_loadings: pd.Series, threshold: float) -> Dict[str, float]:
        """Get variables that dominate this factor (highest loadings)."""
        significant = factor_loadings[np.abs(factor_loadings) >= threshold]
        sorted_loadings = significant.reindex(significant.abs().sort_values(ascending=False).index)
        return sorted_loadings.head(3).to_dict()  # Top 3 dominant variables
    
    def _assess_simple_structure(self, loadings: pd.DataFrame, threshold: float) -> str:
        """Assess simple structure quality."""
        # Count variables with multiple significant loadings (cross-loadings)
        cross_loadings = 0
        total_vars = len(loadings)
        
        for i, row in loadings.iterrows():
            significant_factors = np.sum(np.abs(row) >= threshold)
            if significant_factors > 1:
                cross_loadings += 1
        
        cross_loading_pct = cross_loadings / total_vars if total_vars > 0 else 0
        
        if cross_loading_pct <= 0.1:
            return 'Excellent'
        elif cross_loading_pct <= 0.2:
            return 'Good'
        elif cross_loading_pct <= 0.3:
            return 'Adequate'
        else:
            return 'Poor'
    
    def _identify_cross_loadings(self, loadings: pd.DataFrame, threshold: float) -> List[str]:
        """Identify variables with significant cross-loadings."""
        cross_loaded = []
        for i, row in loadings.iterrows():
            significant_count = np.sum(np.abs(row) >= threshold)
            if significant_count > 1:
                cross_loaded.append(i)
        return cross_loaded
    
    def _identify_unique_variables(self, loadings: pd.DataFrame, threshold: float) -> Dict[str, List[str]]:
        """Identify variables uniquely associated with each factor."""
        unique_vars = {}
        for factor in loadings.columns:
            unique_to_factor = []
            for var in loadings.index:
                # Check if variable loads significantly only on this factor
                significant_on_factor = abs(loadings.loc[var, factor]) >= threshold
                if significant_on_factor:
                    # Check if it loads significantly on any other factor
                    other_factors = [f for f in loadings.columns if f != factor]
                    significant_elsewhere = any(abs(loadings.loc[var, f]) >= threshold for f in other_factors)
                    if not significant_elsewhere:
                        unique_to_factor.append(var)
            unique_vars[factor] = unique_to_factor
        return unique_vars
    
    def _determine_n_factors(self, corr_matrix: pd.DataFrame) -> int:
        """
        Determine optimal number of factors using eigenvalue > 1.0 criterion (Kaiser criterion).
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            int: Number of factors to extract
        """
        # Calculate eigenvalues of correlation matrix
        eigenvals = np.linalg.eigvals(corr_matrix.values)
        eigenvals = np.real(eigenvals)  # Take real part in case of numerical noise
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
        
        # Kaiser criterion: eigenvalue > 1.0
        n_factors = np.sum(eigenvals > 1.0)
        
        # Ensure minimum of 1 factor and maximum constraint
        max_factors = min(corr_matrix.shape[0] // 3, corr_matrix.shape[1] - 1)
        n_factors = max(1, min(n_factors, max_factors))
        
        return n_factors
    
    def format_interpretation_output(self, interpretation: Dict[str, Any], 
                                   include_highlights: bool = True,
                                   max_variables_per_factor: int = 10) -> str:
        """
        Format factor interpretation results for readable output.
        
        Args:
            interpretation: Results from get_factor_interpretation()
            include_highlights: Whether to include significance highlighting
            max_variables_per_factor: Maximum variables to show per factor
            
        Returns:
            Formatted string output for factor interpretation
        """
        output = []
        output.append("=" * 60)
        output.append("FACTOR ANALYSIS INTERPRETATION REPORT")
        output.append("=" * 60)
        
        # Summary statistics
        structure = interpretation['structure_summary']
        output.append(f"\nStructure Summary:")
        output.append(f"- Loading threshold: ≥{interpretation['loading_threshold']:.2f}")
        output.append(f"- Total variables: {structure['total_variables']}")
        output.append(f"- Variables assigned to factors: {structure['assigned_variables']}")
        output.append(f"- Unassigned variables: {structure['unassigned_variables']}")
        output.append(f"- Simple structure quality: {structure['simple_structure_quality']}")
        
        if 'cross_loading_variables' in structure:
            cross_vars = structure['cross_loading_variables']
            output.append(f"- Cross-loading variables: {len(cross_vars)}")
            if cross_vars:
                output.append(f"  {', '.join(cross_vars[:5])}" + ("..." if len(cross_vars) > 5 else ""))
        
        # Factor-by-factor interpretation
        output.append(f"\n{'='*60}")
        output.append("FACTOR INTERPRETATIONS")
        output.append(f"{'='*60}")
        
        for factor_name, factor_info in interpretation['factor_interpretations'].items():
            output.append(f"\n{factor_name}:")
            output.append("-" * len(factor_name) + ":")
            output.append(f"Interpretation Quality: {factor_info['interpretation_quality']}")
            output.append(f"Variables with significant loadings: {factor_info['n_significant']}")
            output.append(f"Maximum loading: {factor_info.get('max_loading', 0):.3f}")
            
            # Show significant variables
            sig_vars = factor_info.get('significant_variables', {})
            if sig_vars:
                output.append("\nSignificant Variables:")
                count = 0
                for var, loading in sig_vars.items():
                    if count >= max_variables_per_factor:
                        break
                    level_indicator = self._get_loading_indicator(abs(loading))
                    output.append(f"  {level_indicator} {var}: {loading:+.3f}")
                    count += 1
                if len(sig_vars) > max_variables_per_factor:
                    output.append(f"  ... and {len(sig_vars) - max_variables_per_factor} more")
            else:
                output.append("\nNo significant variables found")
            
            # Show dominant variables if available
            if 'dominant_variables' in factor_info:
                dom_vars = factor_info['dominant_variables']
                if dom_vars:
                    output.append("\nDominant Variables (Top 3):")
                    for var, loading in dom_vars.items():
                        output.append(f"  ★ {var}: {loading:+.3f}")
        
        # Significance highlighting summary
        if include_highlights and 'significance_highlights' in interpretation:
            highlights = interpretation['significance_highlights']
            output.append(f"\n{'='*60}")
            output.append("SIGNIFICANCE HIGHLIGHTING LEGEND")
            output.append(f"{'='*60}")
            
            levels = highlights.get('significance_levels', {})
            output.append("Loading Significance Levels:")
            if 'very_high' in levels:
                output.append(f"  ★★★ Very High: |loading| ≥ {levels['very_high']}")
            if 'high' in levels:
                output.append(f"  ★★  High:      |loading| ≥ {levels['high']}")
            if 'moderate' in levels:
                output.append(f"  ★   Moderate:  |loading| ≥ {levels['moderate']}")
            output.append(f"      Low:       |loading| < {levels.get('moderate', 0.4)}")
        
        # Unique variables summary
        if 'unique_variables' in structure:
            unique_vars = structure['unique_variables']
            output.append(f"\n{'='*60}")
            output.append("UNIQUE FACTOR ASSOCIATIONS")
            output.append(f"{'='*60}")
            
            for factor, vars_list in unique_vars.items():
                if vars_list:
                    output.append(f"\n{factor} (unique variables: {len(vars_list)}):")
                    for var in vars_list[:5]:  # Show first 5
                        output.append(f"  • {var}")
                    if len(vars_list) > 5:
                        output.append(f"  ... and {len(vars_list) - 5} more")
        
        output.append(f"\n{'='*60}")
        output.append("END OF INTERPRETATION REPORT")
        output.append(f"{'='*60}")
        
        return "\n".join(output)
    
    def _get_loading_indicator(self, abs_loading: float) -> str:
        """Get visual indicator for loading strength."""
        if abs_loading >= 0.8:
            return "★★★"
        elif abs_loading >= 0.6:
            return "★★ "
        elif abs_loading >= 0.4:
            return "★  "
        else:
            return "   "
    
    def compare_extraction_methods(self, data: pd.DataFrame, 
                                 methods: List[str] = None, 
                                 n_factors: int = None) -> Dict[str, Any]:
        """
        Compare different factor extraction methods on the same data.
        
        Args:
            data: Input data matrix
            methods: List of extraction methods to compare
            n_factors: Number of factors to extract (if None, auto-determine)
            
        Returns:
            Dictionary containing comparison results
        """
        if methods is None:
            methods = ['principal', 'ml']  # Principal axis factoring and maximum likelihood
        
        comparison = {
            'methods': methods,
            'results': {},
            'summary': {},
            'recommendations': []
        }
        
        original_method = self.extraction_method
        original_n_factors = self.n_factors
        
        try:
            # Set number of factors if specified
            if n_factors is not None:
                self.n_factors = n_factors
            
            for method in methods:
                try:
                    # Set extraction method
                    self.extraction_method = method
                    
                    # Perform factor analysis
                    solution = self.fit(data)
                    
                    # Store results
                    comparison['results'][method] = {
                        'solution': solution,
                        'n_factors': solution.n_factors,
                        'variance_explained': solution.variance_explained['total_variance_explained'],
                        'communalities_mean': solution.communalities.mean(),
                        'communalities_min': solution.communalities.min(),
                        'loadings_matrix': solution.loadings,
                        'convergence': getattr(solution, 'converged', True),
                        'extraction_method': solution.extraction_method
                    }
                    
                except Exception as e:
                    comparison['results'][method] = {
                        'error': str(e),
                        'failed': True
                    }
            
            # Generate comparison summary
            comparison['summary'] = self._generate_extraction_comparison_summary(comparison['results'])
            comparison['recommendations'] = self._generate_extraction_recommendations(comparison['results'])
            
        finally:
            # Restore original settings
            self.extraction_method = original_method
            self.n_factors = original_n_factors
        
        return comparison
    
    def _generate_extraction_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for extraction method comparison."""
        summary = {}
        
        successful_methods = {method: result for method, result in results.items() 
                            if not result.get('failed', False)}
        
        if not successful_methods:
            return {'error': 'No methods completed successfully'}
        
        # Compare variance explained
        variance_comparison = {}
        communalities_comparison = {}
        
        for method, result in successful_methods.items():
            variance_comparison[method] = result['variance_explained']
            communalities_comparison[method] = result['communalities_mean']
        
        # Find best method by different criteria
        best_variance = max(variance_comparison.items(), key=lambda x: x[1])
        best_communalities = max(communalities_comparison.items(), key=lambda x: x[1])
        
        summary.update({
            'successful_methods': list(successful_methods.keys()),
            'failed_methods': [method for method, result in results.items() if result.get('failed', False)],
            'variance_explained': variance_comparison,
            'communalities_mean': communalities_comparison,
            'best_variance_method': best_variance[0],
            'best_communalities_method': best_communalities[0],
            'variance_difference': max(variance_comparison.values()) - min(variance_comparison.values())
        })
        
        return summary
    
    def _generate_extraction_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on extraction method comparison."""
        recommendations = []
        
        successful_methods = {method: result for method, result in results.items() 
                            if not result.get('failed', False)}
        
        if not successful_methods:
            recommendations.append("All extraction methods failed - check data quality")
            return recommendations
        
        if len(successful_methods) == 1:
            method = list(successful_methods.keys())[0]
            recommendations.append(f"Only {method} method completed successfully")
            return recommendations
        
        # Compare performance
        variance_values = {method: result['variance_explained'] for method, result in successful_methods.items()}
        communalities_values = {method: result['communalities_mean'] for method, result in successful_methods.items()}
        
        best_variance_method = max(variance_values, key=variance_values.get)
        best_communalities_method = max(communalities_values, key=communalities_values.get)
        
        variance_diff = max(variance_values.values()) - min(variance_values.values())
        
        if variance_diff < 0.05:  # Less than 5% difference
            recommendations.append("Methods show similar variance explained - choose based on theoretical preference")
        else:
            recommendations.append(f"'{best_variance_method}' explains most variance ({variance_values[best_variance_method]*100:.1f}%)")
        
        if best_variance_method == best_communalities_method:
            recommendations.append(f"'{best_variance_method}' performs best on both criteria - recommended")
        else:
            recommendations.append(f"Trade-off: '{best_variance_method}' (variance) vs '{best_communalities_method}' (communalities)")
        
        # Method-specific recommendations
        if 'principal' in successful_methods and 'ml' in successful_methods:
            recommendations.append("Principal axis factoring is more robust; ML assumes multivariate normality")
        
        return recommendations
    
    def _perform_factor_analysis(self, data: pd.DataFrame, corr_matrix: pd.DataFrame) -> FactorSolution:
        """
        Perform factor analysis using factor_analyzer library with comprehensive error handling.
        
        Args:
            data: Original data matrix
            corr_matrix: Correlation matrix
            
        Returns:
            FactorSolution: Factor analysis results
            
        Raises:
            RuntimeError: Factor extraction convergence or computation errors
            ValueError: Invalid factor analysis configuration
        """
        try:
            # Initialize FactorAnalyzer with enhanced validation
            fa = FactorAnalyzer(
                n_factors=self.n_factors,
                method=self.extraction_method,
                rotation=self.rotation_method,
                max_iter=self.max_iterations
            )
            
            # Prepare data with proper missing value handling
            clean_data = data.dropna()  # Use listwise deletion for factor extraction
            
            if len(clean_data) == 0:
                raise ValueError("No complete observations available after removing missing values")
            
            if len(clean_data) < self.n_factors * 3:
                warnings.warn(f"Small sample after listwise deletion: {len(clean_data)} observations "
                            f"for {self.n_factors} factors. Results may be unstable.")
            
            # Fit the model with error handling
            try:
                fa.fit(clean_data)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Factor extraction failed due to linear algebra error: {str(e)}. "
                                 f"Try reducing number of factors or check for multicollinearity.") from e
            except ValueError as e:
                raise RuntimeError(f"Factor extraction failed: {str(e)}. "
                                 f"Check data quality and factor extraction parameters.") from e
            
            # Validate convergence
            if hasattr(fa, 'n_iter_') and fa.n_iter_ >= self.max_iterations:
                warnings.warn(f"Factor analysis did not converge within {self.max_iterations} iterations. "
                            f"Consider increasing max_iterations or adjusting convergence_tolerance.")
            
            # Extract results with validation
            try:
                loadings_array = fa.loadings_
                if loadings_array is None:
                    raise RuntimeError("Factor loadings extraction failed - no loadings available")
                    
                loadings = pd.DataFrame(
                    loadings_array,
                    index=data.columns,
                    columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
                )
                
                # Validate loadings
                if np.isnan(loadings.values).any():
                    raise RuntimeError("Factor loadings contain NaN values - extraction failed")
                    
            except (AttributeError, IndexError) as e:
                raise RuntimeError(f"Failed to extract factor loadings: {str(e)}") from e
            
            # Extract communalities with error handling
            try:
                communalities_array = fa.get_communalities()
                communalities = pd.Series(
                    communalities_array,
                    index=data.columns,
                    name='Communalities'
                )
                
                # Validate communalities
                if (communalities < 0).any():
                    warnings.warn("Negative communalities detected - may indicate extraction issues")
                if (communalities > 1).any():
                    warnings.warn("Communalities > 1 detected - check factor extraction method")
                    
            except Exception as e:
                warnings.warn(f"Could not extract communalities: {str(e)}")
                communalities = pd.Series(np.nan, index=data.columns, name='Communalities')
            
            # Extract eigenvalues with error handling
            try:
                eigenvalues = fa.get_eigenvalues()[0]  # Before rotation
            except Exception as e:
                warnings.warn(f"Could not extract eigenvalues: {str(e)}")
                # Fallback to correlation matrix eigenvalues
                eigenvalues = np.linalg.eigvals(corr_matrix.values)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # Calculate variance explained
            variance_explained = self._calculate_variance_explained(eigenvalues, loadings)
            
            # Store factor analyzer for later use
            self.factor_analyzer = fa
            
            solution = FactorSolution(
                loadings=loadings,
                communalities=communalities,
                eigenvalues=eigenvalues,
                variance_explained=variance_explained,
                rotation_method=self.rotation_method,
                extraction_method=self.extraction_method,
                n_factors=self.n_factors
            )
            
            return solution
            
        except Exception as e:
            # Comprehensive error handling for any unexpected issues
            error_msg = f"Factor analysis failed: {str(e)}"
            if "singular" in str(e).lower():
                error_msg += " Check for multicollinearity or perfectly correlated variables."
            elif "convergence" in str(e).lower():
                error_msg += " Try increasing max_iterations or adjusting tolerance."
            raise RuntimeError(error_msg) from e
        
        return solution
    
    def _perform_basic_factor_analysis(self, data: pd.DataFrame, corr_matrix: pd.DataFrame) -> FactorSolution:
        """
        Perform basic factor analysis using numpy when factor_analyzer is not available.
        
        Args:
            data: Original data matrix
            corr_matrix: Correlation matrix
            
        Returns:
            FactorSolution: Basic factor analysis results
        """
        warnings.warn("Using basic factor analysis - install factor_analyzer for full functionality")
        
        # Principal Component Analysis as fallback
        # Center the data
        data_clean = data.dropna()
        data_centered = data_clean - data_clean.mean()
        
        # SVD decomposition
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        
        # Extract loadings (first n_factors components)
        loadings_matrix = Vt[:self.n_factors].T * np.sqrt(s[:self.n_factors] / (len(data_clean) - 1))
        
        loadings = pd.DataFrame(
            loadings_matrix,
            index=data.columns,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        # Calculate communalities (sum of squared loadings)
        communalities = pd.Series(
            (loadings ** 2).sum(axis=1),
            index=data.columns,
            name='Communalities'
        )
        
        # Eigenvalues
        eigenvalues = (s ** 2) / (len(data_clean) - 1)
        
        # Variance explained
        variance_explained = self._calculate_variance_explained(eigenvalues, loadings)
        
        solution = FactorSolution(
            loadings=loadings,
            communalities=communalities,
            eigenvalues=eigenvalues[:len(data.columns)],  # Full eigenvalue spectrum
            variance_explained=variance_explained,
            rotation_method='none',  # No rotation in basic version
            extraction_method='pca_fallback',
            n_factors=self.n_factors
        )
        
        return solution
    
    def _calculate_variance_explained(self, eigenvalues: np.ndarray, loadings: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate variance explained by each factor and cumulative variance.
        
        Args:
            eigenvalues: Eigenvalues from factor analysis
            loadings: Factor loadings matrix
            
        Returns:
            Dictionary with variance explained statistics
        """
        # Sum of squared loadings for each factor
        factor_ss = (loadings ** 2).sum(axis=0)
        
        # Total variance (number of variables for correlation matrix)
        total_variance = loadings.shape[0]
        
        # Proportion of variance explained by each factor
        prop_var = factor_ss / total_variance
        
        # Cumulative variance explained
        cumulative_var = prop_var.cumsum()
        
        variance_explained = {
            'eigenvalues': eigenvalues[:self.n_factors].tolist(),
            'sum_squared_loadings': factor_ss.tolist(),
            'proportion_variance': prop_var.tolist(),
            'cumulative_variance': cumulative_var.tolist(),
            'total_variance_explained': cumulative_var.iloc[-1] if not cumulative_var.empty else 0.0
        }
        
        return variance_explained
    
    def _calculate_factor_scores(self, data: pd.DataFrame, solution: FactorSolution) -> pd.DataFrame:
        """
        Calculate factor scores using regression method.
        
        Args:
            data: Original data matrix
            solution: Factor solution containing loadings
            
        Returns:
            DataFrame with factor scores for each observation
        """
        # Use listwise deletion for factor score calculation (FR-014)
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            raise ValueError("No complete cases available for factor score calculation")
        
        # Standardize the data
        data_std = (data_clean - data_clean.mean()) / data_clean.std()
        
        if FACTOR_ANALYZER_AVAILABLE and hasattr(self, 'factor_analyzer'):
            # Use factor_analyzer's transform method if available
            scores = self.factor_analyzer.transform(data_clean)
            factor_scores = pd.DataFrame(
                scores,
                index=data_clean.index,
                columns=solution.loadings.columns
            )
        else:
            # Regression method: F = Z * L * (L'L)^-1
            # Where F = factor scores, Z = standardized data, L = loadings
            L = solution.loadings.values
            
            # Calculate (L'L)^-1
            try:
                LtL_inv = np.linalg.inv(L.T @ L)
                scores = data_std.values @ L @ LtL_inv
                
                factor_scores = pd.DataFrame(
                    scores,
                    index=data_clean.index,
                    columns=solution.loadings.columns
                )
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                L_pinv = np.linalg.pinv(L)
                scores = data_std.values @ L_pinv.T
                
                factor_scores = pd.DataFrame(
                    scores,
                    index=data_clean.index,
                    columns=solution.loadings.columns
                )
        
        return factor_scores
    
    def compare_rotations(self, data: pd.DataFrame, 
                         rotations: Optional[List[str]] = None) -> Dict[str, FactorSolution]:
        """
        Compare different rotation methods for the same data and number of factors.
        
        Args:
            data: Input data matrix (observations × variables)
            rotations: List of rotation methods to compare
                      (defaults to ['varimax', 'oblimin', 'quartimax'])
            
        Returns:
            Dictionary mapping rotation names to their solutions
            
        Raises:
            ValueError: If no factors have been extracted yet
        """
        if not self.fit_completed:
            raise ValueError("Must run fit() before comparing rotations")
            
        if rotations is None:
            rotations = ['varimax', 'oblimin', 'quartimax']
        
        # Store original rotation method
        original_rotation = self.rotation_method
        
        results = {}
        
        for rotation in rotations:
            try:
                # Create analyzer with same parameters but different rotation
                analyzer = EFAAnalyzer(
                    n_factors=self.n_factors,
                    extraction_method=self.extraction_method,
                    rotation_method=rotation,
                    convergence_tolerance=self.convergence_tolerance,
                    max_iterations=self.max_iterations
                )
                
                # Fit with the same data
                solution = analyzer.fit(data)
                results[rotation] = solution
                
            except Exception as e:
                warnings.warn(f"Failed to fit with {rotation} rotation: {e}")
                continue
        
        # Restore original rotation method
        self.rotation_method = original_rotation
        
        return results
    
    def get_rotation_comparison_metrics(self, comparison_results: Dict[str, FactorSolution]) -> pd.DataFrame:
        """
        Calculate comparison metrics for different rotation methods.
        
        Args:
            comparison_results: Results from compare_rotations()
            
        Returns:
            DataFrame with comparison metrics for each rotation
        """
        metrics = []
        
        for rotation, solution in comparison_results.items():
            # Calculate simplicity metrics
            loadings = solution.loadings
            
            # Number of high loadings per factor (>0.4)
            high_loadings = (np.abs(loadings) > 0.4).sum()
            
            # Complexity: average number of significant loadings per variable
            complexity = (np.abs(loadings) > 0.4).sum(axis=1).mean()
            
            # Total variance explained
            total_var = solution.variance_explained['total_variance_explained']
            
            # Factor correlation (if oblique rotation)
            factor_correlations = getattr(solution, 'factor_correlations', None)
            max_correlation = 0.0
            if factor_correlations is not None:
                # Get maximum off-diagonal correlation
                corr_matrix = factor_correlations.values
                np.fill_diagonal(corr_matrix, 0)
                max_correlation = np.abs(corr_matrix).max()
            
            metrics.append({
                'Rotation': rotation,
                'Total_Variance_Explained': total_var,
                'Average_Complexity': complexity,
                'Max_Factor_Correlation': max_correlation,
                'High_Loadings_Count': high_loadings.sum()
            })
        
        return pd.DataFrame(metrics)
    
    def parallel_analysis(self, data: pd.DataFrame, 
                         n_simulations: int = 1000,
                         percentile: float = 95.0,
                         n_jobs: int = 1,
                         chunk_size: int = None) -> Dict[str, Any]:
        """
        Perform parallel analysis to determine optimal number of factors with performance optimizations.
        
        Args:
            data: Input data matrix (observations × variables)
            n_simulations: Number of random datasets to simulate
            percentile: Percentile for comparison (typically 95)
            n_jobs: Number of parallel jobs (1 for sequential, -1 for all cores)
            chunk_size: Size of simulation chunks for memory efficiency
            
        Returns:
            Dictionary with parallel analysis results including:
            - eigenvalues: Real data eigenvalues
            - simulated_eigenvalues: Mean simulated eigenvalues
            - percentile_eigenvalues: Percentile-based thresholds
            - suggested_factors: Number of factors suggested by PA
        """
        # Input validation
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")
        if not 0 < percentile < 100:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
            
        # Calculate eigenvalues for real data
        data_clean = data.dropna()
        if len(data_clean) < 3:
            raise ValueError("Insufficient data for parallel analysis")
        
        # Performance optimization: cache data statistics
        n_vars = len(data.columns)
        n_obs = len(data_clean)
        
        # Optimize chunk size for memory efficiency
        if chunk_size is None:
            # Adaptive chunk size based on data size and available memory
            memory_per_sim = n_vars * n_obs * 8 / (1024 ** 2)  # MB per simulation
            target_memory = 100  # Target 100MB chunks
            chunk_size = max(10, min(n_simulations, int(target_memory / memory_per_sim)))
        
        # Standardize data (vectorized operation)
        data_std = (data_clean - data_clean.mean()) / data_clean.std()
        
        # Calculate correlation matrix and eigenvalues for real data
        corr_matrix = data_std.corr().values  # Use .values for faster numpy operations
        real_eigenvalues = np.linalg.eigvals(corr_matrix).real
        real_eigenvalues = np.sort(real_eigenvalues)[::-1]  # Sort descending
        
        # Performance optimization: pre-allocate arrays
        simulated_eigenvalues = np.zeros((n_simulations, n_vars))
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        if n_jobs == 1 or n_simulations < 100:
            # Sequential processing for small problems
            for i in range(n_simulations):
                # Generate random data (optimized)
                random_data = np.random.standard_normal((n_obs, n_vars))
                
                # Calculate correlation matrix (optimized)
                random_corr = np.corrcoef(random_data, rowvar=False)
                
                # Calculate eigenvalues (optimized)
                eigenvals = np.linalg.eigvals(random_corr).real
                simulated_eigenvalues[i] = np.sort(eigenvals)[::-1]
                
                # Progress indicator for large simulations
                if n_simulations >= 500 and (i + 1) % 100 == 0:
                    print(f"Parallel analysis progress: {i + 1}/{n_simulations}")
                    
        else:
            # Parallel processing for large problems
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading
                
                def simulate_chunk(chunk_start: int, chunk_end: int) -> np.ndarray:
                    """Simulate a chunk of random datasets."""
                    chunk_results = np.zeros((chunk_end - chunk_start, n_vars))
                    local_rng = np.random.RandomState(42 + chunk_start)
                    
                    for i in range(chunk_end - chunk_start):
                        random_data = local_rng.standard_normal((n_obs, n_vars))
                        random_corr = np.corrcoef(random_data, rowvar=False)
                        eigenvals = np.linalg.eigvals(random_corr).real
                        chunk_results[i] = np.sort(eigenvals)[::-1]
                    
                    return chunk_results
                
                # Process chunks in parallel
                try:
                    import os
                    n_workers = n_jobs if n_jobs > 0 else os.cpu_count()
                except ImportError:
                    n_workers = 4  # Fallback default
                chunks = [(i, min(i + chunk_size, n_simulations)) 
                         for i in range(0, n_simulations, chunk_size)]
                
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_to_chunk = {
                        executor.submit(simulate_chunk, start, end): (start, end)
                        for start, end in chunks
                    }
                    
                    completed_chunks = 0
                    for future in as_completed(future_to_chunk):
                        start, end = future_to_chunk[future]
                        try:
                            chunk_results = future.result()
                            simulated_eigenvalues[start:end] = chunk_results
                            completed_chunks += 1
                            
                            if len(chunks) >= 10 and completed_chunks % max(1, len(chunks) // 10) == 0:
                                print(f"Parallel analysis progress: {completed_chunks}/{len(chunks)} chunks")
                                
                        except Exception as e:
                            warnings.warn(f"Chunk {start}-{end} failed: {str(e)}. Using sequential fallback.")
                            # Fallback to sequential for this chunk
                            for i in range(start, end):
                                random_data = np.random.standard_normal((n_obs, n_vars))
                                random_corr = np.corrcoef(random_data, rowvar=False)
                                eigenvals = np.linalg.eigvals(random_corr).real
                                simulated_eigenvalues[i] = np.sort(eigenvals)[::-1]
                                
            except ImportError:
                warnings.warn("concurrent.futures not available, using sequential processing")
                # Fall back to sequential processing
                for i in range(n_simulations):
                    random_data = np.random.standard_normal((n_obs, n_vars))
                    random_corr = np.corrcoef(random_data, rowvar=False)
                    eigenvals = np.linalg.eigvals(random_corr).real
                    simulated_eigenvalues[i] = np.sort(eigenvals)[::-1]
        
        # Calculate statistics (vectorized operations)
        mean_simulated = np.mean(simulated_eigenvalues, axis=0)
        percentile_simulated = np.percentile(simulated_eigenvalues, percentile, axis=0)
        
        # Determine suggested number of factors
        # Factors where real eigenvalue > percentile threshold
        suggested_factors = np.sum(real_eigenvalues > percentile_simulated)
        
        return {
            'eigenvalues': real_eigenvalues.tolist(),
            'simulated_eigenvalues': mean_simulated.tolist(),
            'percentile_eigenvalues': percentile_simulated.tolist(),
            'suggested_factors': int(suggested_factors),
            'percentile': percentile,
            'n_simulations': n_simulations,
            'variable_names': data.columns.tolist(),
            'n_jobs': n_jobs,
            'chunk_size': chunk_size
        }


# Module-level convenience functions
def quick_efa(data: pd.DataFrame, n_factors: Optional[int] = None, 
              rotation: str = 'oblimin') -> FactorSolution:
    """
    Convenience function for quick EFA with default parameters.
    
    Args:
        data: Input data matrix
        n_factors: Number of factors (auto-determined if None)
        rotation: Rotation method
        
    Returns:
        FactorSolution: Analysis results
    """
    analyzer = EFAAnalyzer(n_factors=n_factors, rotation_method=rotation)
    return analyzer.fit(data)


# Error handling utilities
class EFAError(Exception):
    """Base exception for EFA-related errors."""
    pass

class DataValidationError(EFAError):
    """Exception for data validation failures."""
    pass

class ConvergenceError(EFAError):
    """Exception for algorithm convergence failures."""
    pass