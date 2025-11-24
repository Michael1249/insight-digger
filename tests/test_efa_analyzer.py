# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for EFA Analyzer Module

Tests cover all functionality of the EFAAnalyzer class including:
- Parameter validation and initialization
- Data validation and preprocessing  
- Factor extraction methods (principal, ML)
- Rotation algorithms (varimax, oblimin, quartimax)
- Parallel analysis and factor retention
- Factor score computation and validation
- Error handling and edge cases
- Performance and memory optimization

Author: Insight Digger Project
Enhanced: November 24, 2025
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from efa_analyzer import (EFAAnalyzer, ValidationResults, FactorSolution, 
                         quick_efa, EFAError, DataValidationError, ConvergenceError)


class TestEFAAnalyzer:
    """Comprehensive test suite for EFAAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        
        # Create realistic test data with known factor structure
        self.n_obs = 150
        self.n_vars = 8
        
        # Generate correlated latent factors
        factor1 = np.random.normal(0, 1, self.n_obs)  # Academic factor
        factor2 = np.random.normal(0, 1, self.n_obs)  # Social factor
        
        # Create observed variables with clear loadings pattern
        self.valid_data = pd.DataFrame({
            'academic1': 0.8 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.3, self.n_obs),
            'academic2': 0.7 * factor1 + 0.2 * factor2 + np.random.normal(0, 0.4, self.n_obs),
            'academic3': 0.6 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.5, self.n_obs),
            'academic4': 0.5 * factor1 + 0.2 * factor2 + np.random.normal(0, 0.6, self.n_obs),
            'social1': 0.1 * factor1 + 0.8 * factor2 + np.random.normal(0, 0.3, self.n_obs),
            'social2': 0.2 * factor1 + 0.7 * factor2 + np.random.normal(0, 0.4, self.n_obs),
            'social3': 0.0 * factor1 + 0.6 * factor2 + np.random.normal(0, 0.5, self.n_obs),
            'social4': 0.1 * factor1 + 0.5 * factor2 + np.random.normal(0, 0.6, self.n_obs)
        })
        
        # Create problematic datasets for edge case testing
        self.small_data = self.valid_data.iloc[:8]  # Insufficient observations
        self.few_vars_data = self.valid_data[['academic1', 'social1']]  # Too few variables
        
        # Data with missing values
        self.missing_data = self.valid_data.copy()
        self.missing_data.iloc[0:10, 0] = np.nan
        
        # Data with constant variable
        self.constant_data = self.valid_data.copy()
        self.constant_data['constant'] = 5.0
        
        # Data with perfect multicollinearity
        self.singular_data = self.valid_data.copy()
        self.singular_data['academic1_copy'] = self.singular_data['academic1']
    
    # ===== INITIALIZATION AND PARAMETER VALIDATION TESTS =====
    
    def test_initialization_valid_parameters(self):
        """Test EFAAnalyzer initialization with valid parameters."""
        analyzer = EFAAnalyzer(
            n_factors=2,
            extraction_method='principal',
            rotation_method='varimax',
            convergence_tolerance=1e-5,
            max_iterations=200
        )
        
        assert analyzer.n_factors == 2
        assert analyzer.extraction_method == 'principal'
        assert analyzer.rotation_method == 'varimax'
        assert analyzer.convergence_tolerance == 1e-5
        assert analyzer.max_iterations == 200
        assert not analyzer.fit_completed
        
    def test_initialization_defaults(self):
        """Test EFAAnalyzer initialization with default parameters."""
        analyzer = EFAAnalyzer()
        
        assert analyzer.n_factors is None
        assert analyzer.extraction_method == 'principal'
        assert analyzer.rotation_method == 'oblimin'
        assert analyzer.convergence_tolerance == 1e-4
        assert analyzer.max_iterations == 100
        
    def test_invalid_extraction_method(self):
        """Test initialization with invalid extraction method."""
        with pytest.raises(ValueError, match="extraction_method must be one of"):
            EFAAnalyzer(extraction_method='invalid_method')
            
    def test_invalid_rotation_method(self):
        """Test initialization with invalid rotation method."""
        with pytest.raises(ValueError, match="rotation_method must be one of"):
            EFAAnalyzer(rotation_method='invalid_rotation')
            
    def test_invalid_convergence_tolerance(self):
        """Test initialization with invalid convergence tolerance."""
        with pytest.raises(ValueError, match="convergence_tolerance must be positive"):
            EFAAnalyzer(convergence_tolerance=-0.1)
            
    def test_invalid_max_iterations(self):
        """Test initialization with invalid max iterations."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            EFAAnalyzer(max_iterations=0)
    
    # ===== DATA VALIDATION TESTS =====
    
    def test_data_validation_valid_data(self):
        """Test data validation with valid dataset."""
        analyzer = EFAAnalyzer()
        result = analyzer.validate_data(self.valid_data)
        
        assert isinstance(result, ValidationResults)
        assert result.is_valid
        assert len(result.errors) == 0
        
    def test_data_validation_wrong_type(self):
        """Test data validation with wrong data type."""
        analyzer = EFAAnalyzer()
        
        with pytest.raises(TypeError):
            analyzer.validate_data("not_a_dataframe")
            
    def test_data_validation_insufficient_observations(self):
        """Test data validation with too few observations."""
        analyzer = EFAAnalyzer()
        result = analyzer.validate_data(self.small_data)
        
        assert not result.is_valid
        assert any('observations' in error for error in result.errors)
        
    def test_data_validation_insufficient_variables(self):
        """Test data validation with too few variables."""
        analyzer = EFAAnalyzer()
        result = analyzer.validate_data(self.few_vars_data)
        
        assert not result.is_valid
        assert any('variables' in error for error in result.errors)
        
    def test_data_validation_constant_variable(self):
        """Test data validation with constant variables."""
        analyzer = EFAAnalyzer()
        result = analyzer.validate_data(self.constant_data)
        
        # Should have warnings about constant variables
        assert any('constant' in warning.lower() for warning in result.warnings)
        
    def test_data_validation_missing_values(self):
        """Test data validation with missing values."""
        analyzer = EFAAnalyzer()
        result = analyzer.validate_data(self.missing_data)
        
        # Should have warnings about missing values
        assert any('missing' in warning.lower() for warning in result.warnings)
    
    # ===== FACTOR ANALYSIS CORE TESTS =====
    
    def test_fit_valid_data_specified_factors(self):
        """Test EFA fitting with specified number of factors."""
        analyzer = EFAAnalyzer(n_factors=2, rotation_method='varimax')
        solution = analyzer.fit(self.valid_data)
        
        assert isinstance(solution, FactorSolution)
        assert solution.n_factors == 2
        assert solution.loadings.shape == (self.n_vars, 2)
        assert solution.factor_scores is not None
        assert solution.factor_scores.shape == (self.n_obs, 2)
        assert analyzer.fit_completed
        
    def test_fit_auto_factor_determination(self):
        """Test EFA fitting with automatic factor determination."""
        analyzer = EFAAnalyzer(n_factors=None)
        solution = analyzer.fit(self.valid_data)
        
        assert isinstance(solution, FactorSolution)
        assert solution.n_factors > 0
        assert solution.n_factors <= min(self.n_vars, self.n_obs // 5)
        
    def test_fit_different_rotations(self):
        """Test EFA fitting with different rotation methods."""
        rotations = ['varimax', 'oblimin', 'quartimax']
        
        for rotation in rotations:
            analyzer = EFAAnalyzer(n_factors=2, rotation_method=rotation)
            solution = analyzer.fit(self.valid_data)
            
            assert isinstance(solution, FactorSolution)
            assert solution.n_factors == 2
            assert solution.loadings.shape == (self.n_vars, 2)
            
    def test_fit_invalid_data(self):
        """Test EFA fitting with invalid data raises error."""
        analyzer = EFAAnalyzer(n_factors=2)
        
        with pytest.raises(DataValidationError):
            analyzer.fit(self.small_data)
            
    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation accuracy."""
        analyzer = EFAAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix(self.valid_data)
        
        assert corr_matrix.shape == (self.n_vars, self.n_vars)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric
        
        # Test against pandas correlation
        expected_corr = self.valid_data.corr()
        assert np.allclose(corr_matrix.values, expected_corr.values, atol=1e-10)
    
    # ===== PARALLEL ANALYSIS TESTS =====
    
    def test_parallel_analysis_basic(self):
        """Test basic parallel analysis functionality."""
        analyzer = EFAAnalyzer()
        pa_results = analyzer.parallel_analysis(self.valid_data, n_simulations=50)
        
        required_keys = ['eigenvalues', 'simulated_eigenvalues', 'percentile_eigenvalues', 
                        'suggested_factors', 'percentile', 'n_simulations', 'variable_names']
        
        for key in required_keys:
            assert key in pa_results
            
        assert len(pa_results['eigenvalues']) == self.n_vars
        assert isinstance(pa_results['suggested_factors'], int)
        assert pa_results['suggested_factors'] >= 0
        assert pa_results['n_simulations'] == 50
        
    def test_parallel_analysis_different_percentiles(self):
        """Test parallel analysis with different percentile thresholds."""
        analyzer = EFAAnalyzer()
        
        for percentile in [90, 95, 99]:
            pa_results = analyzer.parallel_analysis(
                self.valid_data, n_simulations=30, percentile=percentile
            )
            assert pa_results['percentile'] == percentile
            assert len(pa_results['percentile_eigenvalues']) == self.n_vars
            
    def test_parallel_analysis_insufficient_data(self):
        """Test parallel analysis with insufficient data."""
        analyzer = EFAAnalyzer()
        
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.parallel_analysis(self.small_data)
    
    # ===== ROTATION COMPARISON TESTS =====
    
    def test_rotation_comparison_basic(self):
        """Test rotation comparison functionality."""
        analyzer = EFAAnalyzer(n_factors=2)
        analyzer.fit(self.valid_data)  # Must fit first
        
        rotations = ['varimax', 'oblimin']
        comparison_results = analyzer.compare_rotations(self.valid_data, rotations)
        
        assert isinstance(comparison_results, dict)
        assert len(comparison_results) == len(rotations)
        
        for rotation in rotations:
            assert rotation in comparison_results
            assert isinstance(comparison_results[rotation], FactorSolution)
            assert comparison_results[rotation].n_factors == 2
            
    def test_rotation_comparison_not_fitted(self):
        """Test rotation comparison without prior fitting."""
        analyzer = EFAAnalyzer(n_factors=2)
        
        with pytest.raises(ValueError, match="Must run fit()"):
            analyzer.compare_rotations(self.valid_data)
            
    def test_rotation_comparison_metrics(self):
        """Test rotation comparison metrics calculation."""
        analyzer = EFAAnalyzer(n_factors=2)
        analyzer.fit(self.valid_data)
        
        comparison_results = analyzer.compare_rotations(self.valid_data, ['varimax', 'oblimin'])
        metrics = analyzer.get_rotation_comparison_metrics(comparison_results)
        
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == 2  # Two rotation methods
        
        expected_columns = ['Rotation', 'Total_Variance_Explained', 'Average_Complexity', 
                           'Max_Factor_Correlation', 'High_Loadings_Count']
        for col in expected_columns:
            assert col in metrics.columns
    
    # ===== FACTOR INTERPRETATION TESTS =====
    
    def test_factor_interpretation(self):
        """Test factor interpretation generation."""
        analyzer = EFAAnalyzer(n_factors=2)
        solution = analyzer.fit(self.valid_data)
        
        interpretation = analyzer.get_factor_interpretation(solution)
        
        assert isinstance(interpretation, dict)
        # Should contain meaningful interpretation information
        
    def test_variance_explained_calculation(self):
        """Test variance explained calculation accuracy."""
        analyzer = EFAAnalyzer(n_factors=2)
        solution = analyzer.fit(self.valid_data)
        
        variance_dict = solution.variance_explained
        
        required_keys = ['eigenvalues', 'proportion_variance', 'cumulative_variance', 
                        'total_variance_explained']
        for key in required_keys:
            assert key in variance_dict
            
        assert len(variance_dict['eigenvalues']) == 2
        assert 0.0 <= variance_dict['total_variance_explained'] <= 1.0
        
        # Cumulative variance should be increasing
        cumulative = variance_dict['cumulative_variance']
        assert all(cumulative[i] <= cumulative[i+1] for i in range(len(cumulative)-1))
        
    def test_factor_scores_calculation(self):
        """Test factor scores calculation and validation."""
        analyzer = EFAAnalyzer(n_factors=2)
        solution = analyzer.fit(self.valid_data)
        
        assert solution.factor_scores is not None
        assert solution.factor_scores.shape == (self.n_obs, 2)
        assert isinstance(solution.factor_scores, pd.DataFrame)
        
        # Factor scores should have reasonable statistical properties
        factor_means = solution.factor_scores.mean()
        assert all(abs(mean) < 0.5 for mean in factor_means)  # Should be roughly centered
    
    # ===== EDGE CASES AND ERROR HANDLING =====
    
    def test_too_many_factors_requested(self):
        """Test error handling when too many factors requested."""
        analyzer = EFAAnalyzer(n_factors=self.n_vars + 1)
        
        with pytest.raises(DataValidationError):
            analyzer.fit(self.valid_data)
            
    def test_missing_data_handling(self):
        """Test proper handling of missing data with listwise deletion."""
        analyzer = EFAAnalyzer(n_factors=2)
        solution = analyzer.fit(self.missing_data)
        
        # Should still produce valid solution with reduced sample size
        assert isinstance(solution, FactorSolution)
        assert solution.factor_scores.shape[0] < self.n_obs
        assert not solution.factor_scores.isnull().any().any()
        
    def test_singular_matrix_handling(self):
        """Test handling of singular correlation matrices."""
        analyzer = EFAAnalyzer(n_factors=2)
        
        # Should either handle gracefully or raise informative error
        try:
            solution = analyzer.fit(self.singular_data)
            # If successful, should handle the singularity
            assert isinstance(solution, FactorSolution)
        except (DataValidationError, EFAError) as e:
            # Expected behavior for singular matrices
            assert "singular" in str(e).lower() or "multicollinearity" in str(e).lower()
    
    # ===== PERFORMANCE AND MEMORY TESTS =====
    
    def test_memory_efficiency_larger_dataset(self):
        """Test memory efficiency with larger datasets."""
        # Create larger test dataset
        np.random.seed(42)
        n_large = 500
        n_vars_large = 15
        
        # Generate realistic factor structure
        factors = np.random.normal(0, 1, (n_large, 3))
        loadings_pattern = np.random.uniform(0.3, 0.8, (n_vars_large, 3))
        error_variance = np.random.normal(0, 0.4, (n_large, n_vars_large))
        
        large_data = pd.DataFrame(
            factors @ loadings_pattern.T + error_variance,
            columns=[f'var{i+1}' for i in range(n_vars_large)]
        )
        
        analyzer = EFAAnalyzer(n_factors=3)
        solution = analyzer.fit(large_data)
        
        assert isinstance(solution, FactorSolution)
        assert solution.n_factors == 3
        assert solution.loadings.shape == (n_vars_large, 3)
        
    def test_reproducibility(self):
        """Test that results are reproducible."""
        # Set same seed for both analyses
        np.random.seed(123)
        analyzer1 = EFAAnalyzer(n_factors=2, rotation_method='varimax')
        solution1 = analyzer1.fit(self.valid_data)
        
        np.random.seed(123)
        analyzer2 = EFAAnalyzer(n_factors=2, rotation_method='varimax')
        solution2 = analyzer2.fit(self.valid_data)
        
        # Results should be very similar (allowing for potential sign flipping)
        loadings1 = solution1.loadings.values
        loadings2 = solution2.loadings.values
        
        # Check similarity (considering possible sign changes)
        similar_direct = np.allclose(loadings1, loadings2, atol=0.1)
        similar_flipped = np.allclose(loadings1, -loadings2, atol=0.1)
        
        assert similar_direct or similar_flipped
    
    # ===== INTEGRATION AND WORKFLOW TESTS =====
    
    def test_complete_workflow(self):
        """Test complete EFA workflow from validation to interpretation."""
        analyzer = EFAAnalyzer()
        
        # Step 1: Data validation
        validation = analyzer.validate_data(self.valid_data)
        assert validation.is_valid
        
        # Step 2: Parallel analysis for factor determination
        pa_results = analyzer.parallel_analysis(self.valid_data, n_simulations=50)
        suggested_factors = pa_results['suggested_factors']
        assert suggested_factors > 0
        
        # Step 3: Factor extraction
        analyzer = EFAAnalyzer(n_factors=min(suggested_factors, 3))
        solution = analyzer.fit(self.valid_data)
        
        # Step 4: Rotation comparison
        rotation_results = analyzer.compare_rotations(self.valid_data, ['varimax', 'oblimin'])
        assert len(rotation_results) == 2
        
        # Step 5: Factor interpretation
        interpretation = analyzer.get_factor_interpretation(solution)
        assert isinstance(interpretation, dict)
        
        # Verify all components are properly connected
        assert solution.n_factors == min(suggested_factors, 3)
        assert solution.loadings.shape[1] == solution.factor_scores.shape[1]
    
    # ===== CONVENIENCE FUNCTION TESTS =====
    
    def test_quick_efa_function(self):
        """Test the quick_efa convenience function."""
        solution = quick_efa(self.valid_data, n_factors=2, rotation='varimax')
        
        assert isinstance(solution, FactorSolution)
        assert solution.n_factors == 2
        assert solution.loadings.shape == (self.n_vars, 2)
        
    # ===== FALLBACK BEHAVIOR TESTS =====
    
    @patch('efa_analyzer.FACTOR_ANALYZER_AVAILABLE', False)
    def test_fallback_without_factor_analyzer(self):
        """Test EFA functionality when factor_analyzer library unavailable."""
        analyzer = EFAAnalyzer(n_factors=2)
        
        # Should still work with manual implementations
        solution = analyzer.fit(self.valid_data)
        assert isinstance(solution, FactorSolution)
        assert solution.n_factors == 2


class TestEFAAnalyzerErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = EFAAnalyzer()
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, TypeError)):
            analyzer.validate_data(empty_df)
            
    def test_non_numeric_data(self):
        """Test handling of non-numeric data."""
        analyzer = EFAAnalyzer()
        text_data = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd', 'e'],
            'col2': ['x', 'y', 'z', 'w', 'v']
        })
        
        with pytest.raises((ValueError, TypeError)):
            analyzer.fit(text_data)


# ===== INTEGRATION TESTS =====

class TestEFAIntegrationWorkflow:
    """Integration tests for complete EFA workflows."""
    
    def setup_method(self):
        """Set up realistic survey data for integration testing."""
        np.random.seed(42)
        n_obs = 200
        
        # Create three correlated latent factors
        academic = np.random.normal(0, 1, n_obs)
        social = np.random.normal(0, 1, n_obs) 
        technical = np.random.normal(0, 1, n_obs)
        
        # Generate survey-like observed variables
        self.survey_data = pd.DataFrame({
            'study_motivation': 0.8 * academic + 0.1 * social + np.random.normal(0, 0.4, n_obs),
            'academic_performance': 0.7 * academic + 0.2 * social + np.random.normal(0, 0.5, n_obs),
            'learning_interest': 0.6 * academic + 0.1 * technical + np.random.normal(0, 0.6, n_obs),
            'social_connection': 0.1 * academic + 0.8 * social + np.random.normal(0, 0.4, n_obs),
            'peer_support': 0.2 * academic + 0.7 * social + np.random.normal(0, 0.5, n_obs),
            'group_activities': 0.0 * academic + 0.6 * social + 0.2 * technical + np.random.normal(0, 0.6, n_obs),
            'tech_comfort': 0.1 * academic + 0.1 * social + 0.8 * technical + np.random.normal(0, 0.4, n_obs),
            'digital_literacy': 0.2 * academic + 0.0 * social + 0.7 * technical + np.random.normal(0, 0.5, n_obs),
            'online_skills': 0.1 * academic + 0.2 * social + 0.6 * technical + np.random.normal(0, 0.6, n_obs)
        })
        
    def test_end_to_end_efa_workflow(self):
        """Test complete end-to-end EFA workflow."""
        # Complete realistic EFA workflow
        analyzer = EFAAnalyzer()
        
        # Validate data quality
        validation = analyzer.validate_data(self.survey_data)
        assert validation.is_valid
        
        # Determine optimal factors via parallel analysis
        pa_results = analyzer.parallel_analysis(self.survey_data, n_simulations=100)
        optimal_factors = pa_results['suggested_factors']
        assert 1 <= optimal_factors <= 5  # Reasonable range
        
        # Extract factors with optimal number
        analyzer = EFAAnalyzer(n_factors=optimal_factors, rotation_method='varimax')
        solution = analyzer.fit(self.survey_data)
        
        # Compare rotation methods
        rotation_comparison = analyzer.compare_rotations(
            self.survey_data, ['varimax', 'oblimin', 'quartimax']
        )
        assert len(rotation_comparison) == 3
        
        # Generate interpretation
        interpretation = analyzer.get_factor_interpretation(solution)
        
        # Validate final solution quality
        assert isinstance(solution, FactorSolution)
        assert solution.n_factors == optimal_factors
        assert solution.variance_explained['total_variance_explained'] > 0.3  # Reasonable threshold
        assert solution.factor_scores is not None
        
        # Check loadings matrix properties
        loadings = solution.loadings
        assert loadings.shape == (9, optimal_factors)  # 9 variables
        assert not loadings.isnull().any().any()  # No missing values
        
        # Factor scores should be reasonable
        scores = solution.factor_scores
        assert scores.shape == (200, optimal_factors)  # All observations
        assert abs(scores.mean().mean()) < 0.5  # Roughly centered


# Test runner configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])