# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Factor Validator Module

Tests cover all statistical validation functionality including:
- KMO (Kaiser-Meyer-Olkin) measure calculation and interpretation
- Bartlett's test of sphericity 
- Sample size adequacy assessment
- Cronbach's alpha reliability calculation
- Enhanced sample adequacy per variable
- Correlation matrix singularity detection
- Data quality validation and reporting
- Edge cases and error handling

Author: Insight Digger Project  
Enhanced: November 24, 2025
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from factor_validator import (FactorValidator, ValidationResults, 
                             StatisticalTest, ReliabilityResult)


class TestFactorValidator:
    """Comprehensive test suite for FactorValidator class."""
    
    def setup_method(self):
        """Set up test fixtures with various data scenarios."""
        np.random.seed(42)
        
        # Create highly correlated data (good for factor analysis)
        self.n_obs = 150
        self.n_vars = 8
        
        # Generate latent factors
        factor1 = np.random.normal(0, 1, self.n_obs)
        factor2 = np.random.normal(0, 1, self.n_obs)
        
        # Create observed variables with strong loadings
        self.good_data = pd.DataFrame({
            'var1': 0.8 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.3, self.n_obs),
            'var2': 0.7 * factor1 + 0.2 * factor2 + np.random.normal(0, 0.4, self.n_obs),
            'var3': 0.6 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.5, self.n_obs),
            'var4': 0.5 * factor1 + 0.2 * factor2 + np.random.normal(0, 0.6, self.n_obs),
            'var5': 0.1 * factor1 + 0.8 * factor2 + np.random.normal(0, 0.3, self.n_obs),
            'var6': 0.2 * factor1 + 0.7 * factor2 + np.random.normal(0, 0.4, self.n_obs),
            'var7': 0.0 * factor1 + 0.6 * factor2 + np.random.normal(0, 0.5, self.n_obs),
            'var8': 0.1 * factor1 + 0.5 * factor2 + np.random.normal(0, 0.6, self.n_obs)
        })
        
        # Create poorly correlated data (bad for factor analysis)
        self.poor_data = pd.DataFrame({
            f'var{i}': np.random.normal(0, 1, self.n_obs) for i in range(1, 9)
        })
        
        # Create data with perfect correlations (singular matrix)
        self.singular_data = self.good_data.copy()
        self.singular_data['var1_copy'] = self.singular_data['var1']
        
        # Create small sample data
        self.small_sample_data = self.good_data.iloc[:20]
        
        # Create data with missing values
        self.missing_data = self.good_data.copy()
        self.missing_data.iloc[0:10, 0] = np.nan
        
        # Compute correlation matrices for direct testing
        self.good_corr = self.good_data.corr()
        self.poor_corr = self.poor_data.corr()
        
        # Sample factor loadings for reliability testing
        self.sample_loadings = pd.DataFrame({
            'Factor1': [0.8, 0.7, 0.6, 0.5, 0.1, 0.2, 0.0, 0.1],
            'Factor2': [0.1, 0.2, 0.1, 0.2, 0.8, 0.7, 0.6, 0.5]
        }, index=self.good_data.columns)
        
    # ===== KMO TESTS =====
    
    def test_kmo_calculation_good_data(self):
        """Test KMO calculation with data suitable for factor analysis."""
        validator = FactorValidator()
        kmo_result = validator.calculate_kmo(self.good_data)
        
        assert isinstance(kmo_result, dict)
        assert 'overall_kmo' in kmo_result
        assert 'variable_kmo' in kmo_result
        assert 'interpretation' in kmo_result
        
        # Good data should have reasonable KMO values
        assert 0.5 <= kmo_result['overall_kmo'] <= 1.0
        assert len(kmo_result['variable_kmo']) == self.n_vars
        
        # All individual KMO values should be valid
        for kmo_value in kmo_result['variable_kmo'].values():
            assert 0.0 <= kmo_value <= 1.0
            
    def test_kmo_calculation_poor_data(self):
        """Test KMO calculation with poorly correlated data."""
        validator = FactorValidator()
        kmo_result = validator.calculate_kmo(self.poor_data)
        
        # Poor data should have low KMO values
        assert kmo_result['overall_kmo'] < 0.6
        assert 'not suitable' in kmo_result['interpretation'].lower()
        
    def test_kmo_from_correlation_matrix(self):
        """Test KMO calculation from correlation matrix directly."""
        validator = FactorValidator()
        
        # Test with good correlation matrix
        kmo_result = validator.calculate_kmo(self.good_corr)
        assert 0.5 <= kmo_result['overall_kmo'] <= 1.0
        
    def test_kmo_edge_cases(self):
        """Test KMO calculation edge cases."""
        validator = FactorValidator()
        
        # Test with identity matrix (no correlations)
        identity_data = pd.DataFrame(np.eye(5))
        kmo_result = validator.calculate_kmo(identity_data)
        assert kmo_result['overall_kmo'] < 0.5
        
        # Test with very small dataset
        with pytest.warns(UserWarning):
            small_kmo = validator.calculate_kmo(self.small_sample_data)
            assert isinstance(small_kmo, dict)
    
    # ===== BARTLETT'S TEST TESTS =====
    
    def test_bartlett_test_good_data(self):
        """Test Bartlett's test with correlated data."""
        validator = FactorValidator()
        bartlett_result = validator.calculate_bartlett_test(self.good_data)
        
        assert isinstance(bartlett_result, StatisticalTest)
        assert hasattr(bartlett_result, 'statistic')
        assert hasattr(bartlett_result, 'p_value')
        assert hasattr(bartlett_result, 'critical_value')
        assert hasattr(bartlett_result, 'interpretation')
        
        # Good data should reject null hypothesis (p < 0.05)
        assert bartlett_result.p_value < 0.05
        assert 'reject' in bartlett_result.interpretation.lower()
        
    def test_bartlett_test_poor_data(self):
        """Test Bartlett's test with uncorrelated data."""
        validator = FactorValidator()
        bartlett_result = validator.calculate_bartlett_test(self.poor_data)
        
        # Poor data might not reject null hypothesis
        assert isinstance(bartlett_result, StatisticalTest)
        assert bartlett_result.p_value >= 0.0  # Valid p-value
        
    def test_bartlett_test_from_correlation(self):
        """Test Bartlett's test from correlation matrix."""
        validator = FactorValidator()
        
        bartlett_result = validator.calculate_bartlett_test(
            self.good_corr, sample_size=self.n_obs
        )
        
        assert isinstance(bartlett_result, StatisticalTest)
        assert bartlett_result.p_value < 0.05  # Should be significant
        
    def test_bartlett_test_insufficient_sample(self):
        """Test Bartlett's test with insufficient sample size."""
        validator = FactorValidator()
        
        with pytest.warns(UserWarning):
            bartlett_result = validator.calculate_bartlett_test(self.small_sample_data)
            assert isinstance(bartlett_result, StatisticalTest)
    
    # ===== SAMPLE ADEQUACY TESTS =====
    
    def test_enhanced_sample_adequacy_good_data(self):
        """Test enhanced sample adequacy assessment with good data."""
        validator = FactorValidator()
        adequacy_result = validator.check_enhanced_sample_adequacy(self.good_data)
        
        assert isinstance(adequacy_result, dict)
        assert 'sample_size' in adequacy_result
        assert 'variables_count' in adequacy_result
        assert 'ratio' in adequacy_result
        assert 'adequacy_level' in adequacy_result
        assert 'recommendations' in adequacy_result
        
        assert adequacy_result['sample_size'] == self.n_obs
        assert adequacy_result['variables_count'] == self.n_vars
        assert adequacy_result['ratio'] > 0
        
    def test_sample_adequacy_ratios(self):
        """Test sample adequacy ratio interpretations."""
        validator = FactorValidator()
        
        # Test various sample sizes
        test_cases = [
            (50, 10, 'Poor'),    # 5:1 ratio
            (100, 10, 'Fair'),   # 10:1 ratio
            (150, 10, 'Good'),   # 15:1 ratio
            (200, 10, 'Excellent')  # 20:1 ratio
        ]
        
        for n_obs, n_vars, expected_level in test_cases:
            test_data = pd.DataFrame(np.random.randn(n_obs, n_vars))
            adequacy = validator.check_enhanced_sample_adequacy(test_data)
            assert adequacy['adequacy_level'] == expected_level
            
    def test_sample_adequacy_warnings(self):
        """Test sample adequacy warnings for small samples."""
        validator = FactorValidator()
        
        with pytest.warns(UserWarning):
            adequacy = validator.check_enhanced_sample_adequacy(self.small_sample_data)
            assert adequacy['adequacy_level'] == 'Poor'
    
    # ===== CRONBACH'S ALPHA TESTS =====
    
    def test_cronbach_alpha_calculation(self):
        """Test Cronbach's alpha calculation."""
        validator = FactorValidator()
        
        # Test with factor loadings
        alpha_result = validator.calculate_cronbach_alpha(
            self.sample_loadings['Factor1'], self.good_data
        )
        
        assert isinstance(alpha_result, ReliabilityResult)
        assert hasattr(alpha_result, 'alpha_value')
        assert hasattr(alpha_result, 'interpretation')
        assert hasattr(alpha_result, 'item_statistics')
        
        assert 0.0 <= alpha_result.alpha_value <= 1.0
        
    def test_cronbach_alpha_high_reliability(self):
        """Test Cronbach's alpha with highly reliable items."""
        validator = FactorValidator()
        
        # Create highly correlated items (should have high alpha)
        reliable_factor = np.random.normal(0, 1, self.n_obs)
        reliable_data = pd.DataFrame({
            'item1': reliable_factor + np.random.normal(0, 0.1, self.n_obs),
            'item2': reliable_factor + np.random.normal(0, 0.1, self.n_obs),
            'item3': reliable_factor + np.random.normal(0, 0.1, self.n_obs),
            'item4': reliable_factor + np.random.normal(0, 0.1, self.n_obs)
        })
        
        high_loadings = pd.Series([0.9, 0.9, 0.9, 0.9], index=reliable_data.columns)
        alpha_result = validator.calculate_cronbach_alpha(high_loadings, reliable_data)
        
        assert alpha_result.alpha_value > 0.8  # Should be high
        assert 'excellent' in alpha_result.interpretation.lower()
        
    def test_cronbach_alpha_low_reliability(self):
        """Test Cronbach's alpha with unreliable items."""
        validator = FactorValidator()
        
        # Create uncorrelated items (should have low alpha)
        unreliable_data = pd.DataFrame({
            'item1': np.random.normal(0, 1, self.n_obs),
            'item2': np.random.normal(0, 1, self.n_obs),
            'item3': np.random.normal(0, 1, self.n_obs)
        })
        
        low_loadings = pd.Series([0.2, 0.1, 0.3], index=unreliable_data.columns)
        alpha_result = validator.calculate_cronbach_alpha(low_loadings, unreliable_data)
        
        assert alpha_result.alpha_value < 0.7  # Should be low
        assert 'poor' in alpha_result.interpretation.lower()
    
    # ===== CORRELATION MATRIX VALIDATION =====
    
    def test_correlation_singularity_check(self):
        """Test correlation matrix singularity detection."""
        validator = FactorValidator()
        
        # Test with good correlation matrix
        singularity_result = validator.check_correlation_singularity(self.good_corr)
        
        assert isinstance(singularity_result, dict)
        assert 'is_singular' in singularity_result
        assert 'determinant' in singularity_result
        assert 'condition_number' in singularity_result
        assert 'problematic_variables' in singularity_result
        
        assert not singularity_result['is_singular']  # Should not be singular
        assert singularity_result['determinant'] > 1e-8  # Reasonable determinant
        
    def test_singular_matrix_detection(self):
        """Test detection of truly singular matrices."""
        validator = FactorValidator()
        
        # Create singular correlation matrix
        singular_corr = self.singular_data.corr()
        singularity_result = validator.check_correlation_singularity(singular_corr)
        
        assert singularity_result['is_singular']  # Should be detected as singular
        assert singularity_result['determinant'] < 1e-8  # Very small determinant
        assert len(singularity_result['problematic_variables']) > 0  # Should identify problematic vars
        
    def test_near_singular_detection(self):
        """Test detection of near-singular matrices."""
        validator = FactorValidator()
        
        # Create near-singular matrix
        near_singular_data = self.good_data.copy()
        near_singular_data['var1_almost'] = near_singular_data['var1'] + np.random.normal(0, 0.001, self.n_obs)
        
        near_singular_corr = near_singular_data.corr()
        singularity_result = validator.check_correlation_singularity(near_singular_corr)
        
        # Should be detected as problematic
        assert singularity_result['condition_number'] > 100  # High condition number
    
    # ===== COMPREHENSIVE VALIDATION TESTS =====
    
    def test_comprehensive_validation_good_data(self):
        """Test comprehensive validation with suitable data."""
        validator = FactorValidator()
        validation_result = validator.comprehensive_validation(self.good_data)
        
        assert isinstance(validation_result, ValidationResults)
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        
        # Should have positive assessments
        assert any('suitable' in msg.lower() for msg in validation_result.warnings)
        
    def test_comprehensive_validation_poor_data(self):
        """Test comprehensive validation with unsuitable data."""
        validator = FactorValidator()
        validation_result = validator.comprehensive_validation(self.poor_data)
        
        # Should have warnings or errors about poor suitability
        total_issues = len(validation_result.warnings) + len(validation_result.errors)
        assert total_issues > 0
        
    def test_comprehensive_validation_small_sample(self):
        """Test comprehensive validation with small sample."""
        validator = FactorValidator()
        
        with pytest.warns(UserWarning):
            validation_result = validator.comprehensive_validation(self.small_sample_data)
            assert isinstance(validation_result, ValidationResults)
            # Should have warnings about sample size
            assert any('sample' in msg.lower() for msg in validation_result.warnings)
    
    # ===== ERROR HANDLING TESTS =====
    
    def test_invalid_input_types(self):
        """Test error handling with invalid input types."""
        validator = FactorValidator()
        
        # Test with non-DataFrame input
        with pytest.raises((TypeError, ValueError)):
            validator.calculate_kmo("not_a_dataframe")
            
        with pytest.raises((TypeError, ValueError)):
            validator.calculate_bartlett_test(123)
            
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        validator = FactorValidator()
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, TypeError)):
            validator.comprehensive_validation(empty_data)
            
    def test_single_variable_handling(self):
        """Test handling of single variable datasets."""
        validator = FactorValidator()
        single_var_data = pd.DataFrame({'var1': np.random.normal(0, 1, 100)})
        
        with pytest.raises((ValueError, TypeError)):
            validator.calculate_kmo(single_var_data)
            
    def test_missing_data_handling(self):
        """Test handling of missing data in validation."""
        validator = FactorValidator()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress missing data warnings
            validation_result = validator.comprehensive_validation(self.missing_data)
            
            # Should handle missing data gracefully
            assert isinstance(validation_result, ValidationResults)
    
    # ===== STATISTICAL ACCURACY TESTS =====
    
    def test_kmo_statistical_accuracy(self):
        """Test KMO calculation accuracy against known values."""
        validator = FactorValidator()
        
        # Create controlled correlation matrix
        controlled_corr = pd.DataFrame([
            [1.0, 0.6, 0.5, 0.4],
            [0.6, 1.0, 0.5, 0.4], 
            [0.5, 0.5, 1.0, 0.6],
            [0.4, 0.4, 0.6, 1.0]
        ])
        
        kmo_result = validator.calculate_kmo(controlled_corr)
        
        # KMO should be reasonable for moderate correlations
        assert 0.6 <= kmo_result['overall_kmo'] <= 0.9
        
    def test_bartlett_statistical_accuracy(self):
        """Test Bartlett's test statistical accuracy."""
        validator = FactorValidator()
        
        # Test with identity matrix (should not reject H0)
        identity_corr = pd.DataFrame(np.eye(5))
        bartlett_result = validator.calculate_bartlett_test(identity_corr, sample_size=100)
        
        # P-value should be high (fail to reject)
        assert bartlett_result.p_value > 0.05
        
    def test_reliability_calculation_accuracy(self):
        """Test Cronbach's alpha calculation accuracy."""
        validator = FactorValidator()
        
        # Create perfect reliability scenario
        perfect_factor = np.random.normal(0, 1, 100)
        perfect_data = pd.DataFrame({
            'item1': perfect_factor,
            'item2': perfect_factor, 
            'item3': perfect_factor
        })
        
        perfect_loadings = pd.Series([1.0, 1.0, 1.0], index=perfect_data.columns)
        alpha_result = validator.calculate_cronbach_alpha(perfect_loadings, perfect_data)
        
        # Should have very high reliability
        assert alpha_result.alpha_value > 0.95


class TestFactorValidatorIntegration:
    """Integration tests for FactorValidator with realistic scenarios."""
    
    def setup_method(self):
        """Set up realistic survey validation scenarios."""
        np.random.seed(42)
        
        # Create realistic survey data for validation testing
        n_obs = 250
        
        # Academic performance factor
        academic = np.random.normal(0, 1, n_obs)
        # Social engagement factor  
        social = np.random.normal(0, 1, n_obs)
        
        self.survey_data = pd.DataFrame({
            'study_habits': 0.8 * academic + 0.1 * social + np.random.normal(0, 0.4, n_obs),
            'test_performance': 0.7 * academic + 0.2 * social + np.random.normal(0, 0.5, n_obs),
            'homework_completion': 0.6 * academic + 0.1 * social + np.random.normal(0, 0.6, n_obs),
            'class_participation': 0.2 * academic + 0.8 * social + np.random.normal(0, 0.4, n_obs),
            'peer_interaction': 0.1 * academic + 0.7 * social + np.random.normal(0, 0.5, n_obs),
            'group_collaboration': 0.0 * academic + 0.6 * social + np.random.normal(0, 0.6, n_obs)
        })
        
    def test_realistic_survey_validation(self):
        """Test validation with realistic survey data."""
        validator = FactorValidator()
        
        # Run comprehensive validation
        validation_result = validator.comprehensive_validation(self.survey_data)
        
        assert isinstance(validation_result, ValidationResults)
        assert validation_result.is_valid  # Should be suitable for factor analysis
        
        # Check individual components
        kmo_result = validator.calculate_kmo(self.survey_data)
        assert kmo_result['overall_kmo'] > 0.5  # Should be adequate
        
        bartlett_result = validator.calculate_bartlett_test(self.survey_data)
        assert bartlett_result.p_value < 0.05  # Should be significant
        
        adequacy_result = validator.check_enhanced_sample_adequacy(self.survey_data)
        assert adequacy_result['adequacy_level'] in ['Good', 'Excellent']


# Test fallback behavior
class TestFactorValidatorFallbacks:
    """Test fallback behavior when optional dependencies unavailable."""
    
    @patch('factor_validator.SCIPY_AVAILABLE', False)
    def test_fallback_without_scipy(self):
        """Test FactorValidator functionality without scipy."""
        validator = FactorValidator()
        
        # Should still work with manual implementations
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 5))
        
        # Basic validation should still work
        validation_result = validator.comprehensive_validation(data)
        assert isinstance(validation_result, ValidationResults)


# Test runner configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])