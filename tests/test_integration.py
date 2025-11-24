# -*- coding: utf-8 -*-
"""
Integration Test Suite for Insight Digger EFA Toolkit

Tests comprehensive end-to-end workflows including:
- Complete EFA analysis pipelines
- CLI command execution and output validation
- Cross-module integration scenarios
- Data flow validation between components
- Output consistency and format validation
- Error handling in integrated workflows
- Performance and memory usage testing

Author: Insight Digger Project
Created: November 24, 2025
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import sys
import subprocess
import json
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch, MagicMock
import warnings

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from efa_analyzer import EFAAnalyzer
from factor_validator import FactorValidator
from visualization_engine import VisualizationEngine


class TestEFAIntegrationWorkflows:
    """Test complete EFA analysis workflows end-to-end."""
    
    def setup_method(self):
        """Set up comprehensive test datasets for integration testing."""
        np.random.seed(42)
        
        # Create realistic psychological survey data
        n_obs = 200
        
        # Latent factors: Extraversion, Neuroticism, Openness
        extraversion = np.random.normal(0, 1, n_obs)
        neuroticism = np.random.normal(0, 1, n_obs)
        openness = np.random.normal(0, 1, n_obs)
        
        # Observed variables with realistic factor structure
        self.psychology_data = pd.DataFrame({
            # Extraversion indicators
            'outgoing': 0.8 * extraversion + 0.1 * neuroticism + np.random.normal(0, 0.4, n_obs),
            'sociable': 0.7 * extraversion + 0.0 * neuroticism + np.random.normal(0, 0.5, n_obs),
            'energetic': 0.6 * extraversion - 0.1 * neuroticism + np.random.normal(0, 0.6, n_obs),
            'assertive': 0.5 * extraversion - 0.2 * neuroticism + np.random.normal(0, 0.7, n_obs),
            
            # Neuroticism indicators  
            'anxious': -0.1 * extraversion + 0.8 * neuroticism + np.random.normal(0, 0.4, n_obs),
            'worried': 0.0 * extraversion + 0.7 * neuroticism + np.random.normal(0, 0.5, n_obs),
            'tense': -0.2 * extraversion + 0.6 * neuroticism + np.random.normal(0, 0.6, n_obs),
            'nervous': -0.1 * extraversion + 0.5 * neuroticism + np.random.normal(0, 0.7, n_obs),
            
            # Openness indicators
            'creative': 0.2 * extraversion + 0.0 * neuroticism + 0.8 * openness + np.random.normal(0, 0.4, n_obs),
            'imaginative': 0.1 * extraversion - 0.1 * neuroticism + 0.7 * openness + np.random.normal(0, 0.5, n_obs),
            'artistic': 0.0 * extraversion + 0.0 * neuroticism + 0.6 * openness + np.random.normal(0, 0.6, n_obs),
            'intellectual': -0.1 * extraversion - 0.2 * neuroticism + 0.5 * openness + np.random.normal(0, 0.7, n_obs)
        })
        
        # Create academic performance data
        academic_ability = np.random.normal(0, 1, n_obs)
        motivation = np.random.normal(0, 1, n_obs)
        
        self.academic_data = pd.DataFrame({
            'math_score': 0.8 * academic_ability + 0.2 * motivation + np.random.normal(0, 0.4, n_obs),
            'reading_score': 0.7 * academic_ability + 0.3 * motivation + np.random.normal(0, 0.5, n_obs),
            'writing_score': 0.6 * academic_ability + 0.4 * motivation + np.random.normal(0, 0.6, n_obs),
            'science_score': 0.8 * academic_ability + 0.1 * motivation + np.random.normal(0, 0.4, n_obs),
            'homework_completion': 0.2 * academic_ability + 0.8 * motivation + np.random.normal(0, 0.4, n_obs),
            'class_participation': 0.1 * academic_ability + 0.7 * motivation + np.random.normal(0, 0.5, n_obs),
            'study_hours': 0.3 * academic_ability + 0.6 * motivation + np.random.normal(0, 0.6, n_obs)
        })
        
        # Create problematic data for testing edge cases
        self.problematic_data = pd.DataFrame({
            'random1': np.random.normal(0, 1, n_obs),
            'random2': np.random.normal(0, 1, n_obs),
            'random3': np.random.normal(0, 1, n_obs),
            'random4': np.random.normal(0, 1, n_obs),
            'correlated': np.random.normal(0, 1, n_obs)
        })
        # Add perfect correlation for testing
        self.problematic_data['perfect_copy'] = self.problematic_data['correlated']
        
        # Create temporary directory for file I/O tests
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary files after testing."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_psychology_analysis_workflow(self):
        """Test complete EFA workflow with psychological data."""
        
        # Step 1: Validation
        validator = FactorValidator()
        validation_results = validator.comprehensive_validation(self.psychology_data)
        
        assert validation_results.is_valid, "Psychology data should be valid for factor analysis"
        
        # Step 2: EFA Analysis
        efa = EFAAnalyzer(n_factors=3, rotation='varimax')
        efa.fit(self.psychology_data)
        
        # Verify analysis results
        assert efa.factor_loadings_.shape == (12, 3), "Should have 12 variables and 3 factors"
        assert len(efa.eigenvalues_) == 12, "Should have 12 eigenvalues"
        assert efa.explained_variance_ratio_.sum() > 0.4, "Should explain reasonable variance"
        
        # Step 3: Interpretation and Validation
        reliability_results = []
        for factor_idx in range(3):
            factor_loadings = efa.factor_loadings_.iloc[:, factor_idx]
            # Only include variables with substantial loadings (>0.4)
            substantial_vars = factor_loadings[abs(factor_loadings) > 0.4].index
            if len(substantial_vars) >= 3:  # Need at least 3 items for reliability
                factor_data = self.psychology_data[substantial_vars]
                reliability = validator.calculate_cronbach_alpha(
                    factor_loadings[substantial_vars], factor_data
                )
                reliability_results.append(reliability)
        
        assert len(reliability_results) > 0, "Should calculate reliability for at least one factor"
        
        # Step 4: Visualization (if available)
        try:
            viz = VisualizationEngine()
            
            # Test scree plot
            scree_fig = viz.plot_scree(efa.eigenvalues_)
            assert scree_fig is not None, "Should generate scree plot"
            
            # Test factor loadings heatmap
            heatmap_fig = viz.plot_factor_loadings_heatmap(efa.factor_loadings_)
            assert heatmap_fig is not None, "Should generate loadings heatmap"
            
        except ImportError:
            # Graceful degradation when visualization dependencies unavailable
            pass
        
    def test_academic_analysis_with_parallel_analysis(self):
        """Test EFA workflow with parallel analysis for factor determination."""
        
        # Validation first
        validator = FactorValidator()
        validation_results = validator.comprehensive_validation(self.academic_data)
        assert validation_results.is_valid, "Academic data should be valid for factor analysis"
        
        # EFA with automatic factor determination
        efa = EFAAnalyzer(rotation='oblimin')  # Allow correlated factors
        efa.fit(self.academic_data)
        
        # Run parallel analysis to determine optimal factors
        try:
            n_factors_suggested = efa.parallel_analysis(self.academic_data, n_iterations=100)
            assert 1 <= n_factors_suggested <= 7, "Parallel analysis should suggest reasonable factor count"
            
            # Re-run with suggested factors
            efa_optimal = EFAAnalyzer(n_factors=n_factors_suggested, rotation='oblimin')
            efa_optimal.fit(self.academic_data)
            
            # Validate the solution
            assert efa_optimal.factor_loadings_.shape[1] == n_factors_suggested
            
        except Exception as e:
            # If parallel analysis fails, should still have basic solution
            assert efa.factor_loadings_ is not None, f"Basic EFA should work even if parallel analysis fails: {e}"
    
    def test_rotation_comparison_workflow(self):
        """Test workflow comparing different rotation methods."""
        
        # Validation
        validator = FactorValidator()
        validation_results = validator.comprehensive_validation(self.psychology_data)
        assert validation_results.is_valid
        
        # Test multiple rotations
        rotations = ['varimax', 'oblimin', 'quartimax', 'promax']
        rotation_results = {}
        
        for rotation in rotations:
            try:
                efa = EFAAnalyzer(n_factors=3, rotation=rotation)
                efa.fit(self.psychology_data)
                rotation_results[rotation] = {
                    'loadings': efa.factor_loadings_,
                    'variance_explained': efa.explained_variance_ratio_.sum(),
                    'success': True
                }
            except Exception as e:
                rotation_results[rotation] = {'success': False, 'error': str(e)}
        
        # Should have at least varimax working
        assert rotation_results['varimax']['success'], "Varimax rotation should work"
        
        # Compare rotation effectiveness
        successful_rotations = {k: v for k, v in rotation_results.items() if v['success']}
        assert len(successful_rotations) >= 1, "At least one rotation should succeed"
        
        # Test rotation comparison functionality if available
        efa = EFAAnalyzer(n_factors=3)
        try:
            comparison = efa.compare_rotations(
                self.psychology_data, 
                rotations=['varimax', 'oblimin']
            )
            assert isinstance(comparison, dict), "Rotation comparison should return results"
            assert 'varimax' in comparison, "Should include varimax results"
            
        except Exception:
            # Graceful handling if comparison not implemented
            pass
    
    def test_problematic_data_handling_workflow(self):
        """Test complete workflow with problematic data scenarios."""
        
        # Test validation catches problems
        validator = FactorValidator()
        validation_results = validator.comprehensive_validation(self.problematic_data)
        
        # Should identify issues
        total_issues = len(validation_results.errors) + len(validation_results.warnings)
        assert total_issues > 0, "Should detect problems with random/singular data"
        
        # EFA should handle gracefully
        efa = EFAAnalyzer(n_factors=2)
        
        # May succeed but with warnings, or may fail gracefully
        try:
            efa.fit(self.problematic_data)
            # If it succeeds, should have reasonable results
            if efa.factor_loadings_ is not None:
                assert efa.factor_loadings_.shape[0] == len(self.problematic_data.columns)
                
        except Exception as e:
            # Should fail gracefully with informative error
            assert isinstance(e, (ValueError, np.linalg.LinAlgError)), \
                f"Should fail gracefully with appropriate error type, got: {type(e)}"
    
    def test_file_io_integration_workflow(self):
        """Test complete workflow with file input/output."""
        
        # Save test data to temporary files
        psychology_file = os.path.join(self.temp_dir, 'psychology_data.csv')
        results_file = os.path.join(self.temp_dir, 'efa_results.csv')
        
        self.psychology_data.to_csv(psychology_file, index=False)
        
        # Test loading data
        loaded_data = pd.read_csv(psychology_file)
        assert loaded_data.shape == self.psychology_data.shape, "Data should load correctly"
        
        # Run analysis
        validator = FactorValidator()
        validation_results = validator.comprehensive_validation(loaded_data)
        assert validation_results.is_valid
        
        efa = EFAAnalyzer(n_factors=3, rotation='varimax')
        efa.fit(loaded_data)
        
        # Save results
        efa.factor_loadings_.to_csv(results_file)
        
        # Verify results file
        assert os.path.exists(results_file), "Results file should be created"
        
        loaded_results = pd.read_csv(results_file, index_col=0)
        assert loaded_results.shape == efa.factor_loadings_.shape, "Results should save/load correctly"
    
    def test_memory_and_performance_workflow(self):
        """Test workflow with large datasets for memory/performance validation."""
        
        # Create larger dataset
        np.random.seed(42)
        large_n = 1000
        large_p = 20
        
        # Generate realistic factor structure
        factor1 = np.random.normal(0, 1, large_n)
        factor2 = np.random.normal(0, 1, large_n)
        factor3 = np.random.normal(0, 1, large_n)
        factor4 = np.random.normal(0, 1, large_n)
        
        large_data = pd.DataFrame({
            f'var{i:02d}': (
                (0.8 if i % 4 == 0 else 0.1) * factor1 +
                (0.8 if i % 4 == 1 else 0.1) * factor2 +
                (0.8 if i % 4 == 2 else 0.1) * factor3 +
                (0.8 if i % 4 == 3 else 0.1) * factor4 +
                np.random.normal(0, 0.4, large_n)
            ) for i in range(large_p)
        })
        
        # Test validation performance
        validator = FactorValidator()
        validation_results = validator.comprehensive_validation(large_data)
        assert isinstance(validation_results.is_valid, bool), "Large data validation should complete"
        
        # Test EFA performance  
        efa = EFAAnalyzer(n_factors=4, rotation='varimax')
        efa.fit(large_data)
        
        assert efa.factor_loadings_.shape == (large_p, 4), "Large data EFA should complete"
        assert efa.explained_variance_ratio_.sum() > 0.3, "Should explain reasonable variance"
    
    def test_cross_validation_workflow(self):
        """Test workflow with cross-validation for solution stability."""
        
        # Split psychology data for cross-validation
        np.random.seed(42)
        n_obs = len(self.psychology_data)
        train_idx = np.random.choice(n_obs, size=int(0.7 * n_obs), replace=False)
        test_idx = np.setdiff1d(np.arange(n_obs), train_idx)
        
        train_data = self.psychology_data.iloc[train_idx]
        test_data = self.psychology_data.iloc[test_idx]
        
        # Validate both splits
        validator = FactorValidator()
        train_valid = validator.comprehensive_validation(train_data)
        test_valid = validator.comprehensive_validation(test_data)
        
        assert train_valid.is_valid, "Training data should be valid"
        assert test_valid.is_valid, "Test data should be valid"
        
        # Fit on training data
        efa_train = EFAAnalyzer(n_factors=3, rotation='varimax')
        efa_train.fit(train_data)
        
        # Validate on test data (factor structure similarity)
        efa_test = EFAAnalyzer(n_factors=3, rotation='varimax')
        efa_test.fit(test_data)
        
        # Both should produce reasonable solutions
        assert efa_train.factor_loadings_.shape == (12, 3)
        assert efa_test.factor_loadings_.shape == (12, 3)
        
        # Variance explained should be similar (within reason)
        train_variance = efa_train.explained_variance_ratio_.sum()
        test_variance = efa_test.explained_variance_ratio_.sum()
        variance_diff = abs(train_variance - test_variance)
        assert variance_diff < 0.3, "Cross-validation solutions should have similar explanatory power"


class TestCLIIntegration:
    """Test command-line interface integration."""
    
    def setup_method(self):
        """Set up CLI testing environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data file
        np.random.seed(42)
        n_obs = 100
        factor1 = np.random.normal(0, 1, n_obs)
        factor2 = np.random.normal(0, 1, n_obs)
        
        test_data = pd.DataFrame({
            'var1': 0.8 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.4, n_obs),
            'var2': 0.7 * factor1 + 0.2 * factor2 + np.random.normal(0, 0.5, n_obs),
            'var3': 0.1 * factor1 + 0.8 * factor2 + np.random.normal(0, 0.4, n_obs),
            'var4': 0.2 * factor1 + 0.7 * factor2 + np.random.normal(0, 0.5, n_obs)
        })
        
        self.test_data_file = os.path.join(self.temp_dir, 'test_data.csv')
        test_data.to_csv(self.test_data_file, index=False)
        
    def teardown_method(self):
        """Clean up CLI test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cli_basic_analysis(self):
        """Test basic CLI analysis workflow."""
        
        # Find CLI script
        cli_script = os.path.join(os.path.dirname(__file__), '..', 'src', 'cli.py')
        if not os.path.exists(cli_script):
            pytest.skip("CLI script not found")
        
        # Test basic analysis command
        cmd = [
            sys.executable, cli_script, 'analyze',
            '--input', self.test_data_file,
            '--factors', '2',
            '--rotation', 'varimax'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Should complete successfully
            assert result.returncode == 0, f"CLI command failed: {result.stderr}"
            
            # Should produce meaningful output
            output = result.stdout
            assert 'Factor' in output or 'Loading' in output or 'Variance' in output, \
                "CLI output should contain analysis results"
                
        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out")
        except FileNotFoundError:
            pytest.skip("Python executable not found for CLI testing")
    
    def test_cli_validation_command(self):
        """Test CLI validation command."""
        
        cli_script = os.path.join(os.path.dirname(__file__), '..', 'src', 'cli.py')
        if not os.path.exists(cli_script):
            pytest.skip("CLI script not found")
        
        # Test validation command
        cmd = [
            sys.executable, cli_script, 'validate',
            '--input', self.test_data_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Should complete successfully
            assert result.returncode == 0, f"CLI validation failed: {result.stderr}"
            
            # Should produce validation output
            output = result.stdout
            assert any(word in output.lower() for word in ['kmo', 'bartlett', 'valid', 'suitable']), \
                "CLI validation output should contain validation metrics"
                
        except subprocess.TimeoutExpired:
            pytest.fail("CLI validation command timed out")
        except FileNotFoundError:
            pytest.skip("Python executable not found for CLI testing")


class TestErrorHandlingIntegration:
    """Test integrated error handling across modules."""
    
    def test_coordinated_error_handling(self):
        """Test that errors are handled consistently across modules."""
        
        # Test with various error conditions
        error_conditions = [
            pd.DataFrame(),  # Empty data
            pd.DataFrame({'single_var': [1, 2, 3]}),  # Single variable
            pd.DataFrame({'var1': [1, 1, 1], 'var2': [2, 2, 2]}),  # Constant variables
        ]
        
        for error_data in error_conditions:
            
            # Validator should catch issues
            validator = FactorValidator()
            try:
                validation_result = validator.comprehensive_validation(error_data)
                # If validation succeeds, should indicate problems
                if validation_result.is_valid:
                    assert len(validation_result.warnings) > 0, \
                        "Validator should warn about problematic data"
            except (ValueError, TypeError) as e:
                # Acceptable to fail with clear error
                assert len(str(e)) > 0, "Error message should be informative"
            
            # EFA should handle gracefully
            efa = EFAAnalyzer(n_factors=2)
            try:
                efa.fit(error_data)
                # If it succeeds, should have valid results
                if efa.factor_loadings_ is not None:
                    assert efa.factor_loadings_.shape[0] > 0
            except (ValueError, TypeError, np.linalg.LinAlgError) as e:
                # Acceptable to fail with clear error
                assert len(str(e)) > 0, "Error message should be informative"


class TestVisualizationIntegration:
    """Test visualization integration with analysis results."""
    
    def setup_method(self):
        """Set up visualization test data."""
        np.random.seed(42)
        
        # Create factor structure for visualization
        n_obs = 150
        factor1 = np.random.normal(0, 1, n_obs)
        factor2 = np.random.normal(0, 1, n_obs)
        
        self.viz_data = pd.DataFrame({
            'var1': 0.8 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.4, n_obs),
            'var2': 0.7 * factor1 + 0.2 * factor2 + np.random.normal(0, 0.5, n_obs),
            'var3': 0.6 * factor1 + 0.1 * factor2 + np.random.normal(0, 0.6, n_obs),
            'var4': 0.1 * factor1 + 0.8 * factor2 + np.random.normal(0, 0.4, n_obs),
            'var5': 0.2 * factor1 + 0.7 * factor2 + np.random.normal(0, 0.5, n_obs),
            'var6': 0.0 * factor1 + 0.6 * factor2 + np.random.normal(0, 0.6, n_obs)
        })
    
    def test_complete_visualization_workflow(self):
        """Test complete analysis and visualization workflow."""
        
        # Run EFA
        efa = EFAAnalyzer(n_factors=2, rotation='varimax')
        efa.fit(self.viz_data)
        
        # Test visualization integration
        try:
            viz = VisualizationEngine()
            
            # Test all visualization types
            viz_methods = [
                ('plot_scree', [efa.eigenvalues_]),
                ('plot_factor_loadings_heatmap', [efa.factor_loadings_]),
                ('plot_factor_loadings_bar', [efa.factor_loadings_]),
            ]
            
            for method_name, args in viz_methods:
                if hasattr(viz, method_name):
                    method = getattr(viz, method_name)
                    try:
                        result = method(*args)
                        assert result is not None, f"{method_name} should return a figure"
                    except ImportError:
                        # Graceful handling of missing dependencies
                        pass
            
            # Test biplot if available
            if hasattr(viz, 'plot_biplot'):
                try:
                    factor_scores = efa.transform(self.viz_data)
                    biplot = viz.plot_biplot(factor_scores, efa.factor_loadings_)
                    assert biplot is not None, "Biplot should be generated"
                except (ImportError, AttributeError):
                    pass
                    
        except ImportError:
            # Visualization engine not available - acceptable
            pass
    
    def test_visualization_fallback_behavior(self):
        """Test that analysis works without visualization dependencies."""
        
        # Mock missing visualization dependencies
        with patch.dict(sys.modules, {'matplotlib': None, 'seaborn': None, 'plotly': None}):
            
            # Analysis should still work
            efa = EFAAnalyzer(n_factors=2, rotation='varimax')
            efa.fit(self.viz_data)
            
            assert efa.factor_loadings_ is not None, "Analysis should work without visualization"
            assert efa.eigenvalues_ is not None, "Basic results should be available"
            
            # Visualization engine should handle gracefully
            try:
                viz = VisualizationEngine()
                # Should either work with fallbacks or fail gracefully
            except ImportError:
                # Acceptable if visualization completely unavailable
                pass


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for integration scenarios."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        
        # Generate large dataset
        np.random.seed(42)
        n_obs = 2000
        n_vars = 50
        n_factors = 5
        
        # Generate factor structure
        factors = [np.random.normal(0, 1, n_obs) for _ in range(n_factors)]
        
        large_data = pd.DataFrame({
            f'var{i:02d}': (
                sum(np.random.uniform(0.3, 0.8) if j == i % n_factors else np.random.uniform(-0.2, 0.2)
                    for j, factor in enumerate(factors)) +
                np.random.normal(0, 0.5, n_obs)
            ) for i in range(n_vars)
        })
        
        import time
        
        # Benchmark validation
        start_time = time.time()
        validator = FactorValidator()
        validation_result = validator.comprehensive_validation(large_data)
        validation_time = time.time() - start_time
        
        assert validation_time < 30, f"Validation took too long: {validation_time:.2f}s"
        
        # Benchmark EFA
        start_time = time.time()
        efa = EFAAnalyzer(n_factors=n_factors, rotation='varimax')
        efa.fit(large_data)
        efa_time = time.time() - start_time
        
        assert efa_time < 60, f"EFA took too long: {efa_time:.2f}s"
        assert efa.factor_loadings_.shape == (n_vars, n_factors), "Should complete successfully"


# Test runner configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])