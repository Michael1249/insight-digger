"""
Comprehensive Test Suite for SocOpros Loader Module
Tests the dynamic data loading and structure handling capabilities.
Follows Insight Digger Constitution: Requirements-First Development
"""

import unittest
import sys
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List

# Add src directory to path
sys.path.append('../src')
sys.path.append('src')

try:
    from soc_opros_loader import SocOprosLoader, load_soc_opros_data
    LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SocOprosLoader not available - {e}")
    SocOprosLoader = None
    load_soc_opros_data = None
    LOADER_AVAILABLE = False


class TestSocOprosLoader(unittest.TestCase):
    """Test suite for SocOpros data loader functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with loader instance."""
        if LOADER_AVAILABLE:
            cls.loader = SocOprosLoader()
            cls.test_data = None
            cls.connection_successful = False
        else:
            cls.loader = None
    
    def test_01_module_availability(self):
        """Test that the SocOprosLoader module can be imported."""
        self.assertTrue(LOADER_AVAILABLE, "SocOprosLoader should be importable")
        self.assertIsNotNone(self.loader, "Loader instance should be created")
    
    def test_02_loader_initialization(self):
        """Test proper initialization of SocOprosLoader."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        loader = SocOprosLoader()
        
        # Check default configuration
        self.assertIsNotNone(loader.sheet_id, "Sheet ID should be configured")
        self.assertIsNotNone(loader.worksheet_gid, "Worksheet GID should be configured")
        self.assertIsNotNone(loader.base_url, "Base URL should be configured")
        
        # Check initial state
        self.assertIsNone(loader.data, "Data should be None initially")
        self.assertEqual(loader.statements, [], "Statements should be empty initially")
        self.assertEqual(loader.respondents, [], "Respondents should be empty initially")
        
        print(f"Loader initialized with sheet: {loader.sheet_id}")
    
    def test_03_data_loading(self):
        """Test data loading from Google Sheets."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        try:
            data = self.loader.load_data()
            TestSocOprosLoader.test_data = data
            TestSocOprosLoader.connection_successful = True
            
            # Validate data structure
            self.assertIsInstance(data, pd.DataFrame, "Should return DataFrame")
            self.assertGreater(data.shape[0], 0, "Should have rows")
            self.assertGreater(data.shape[1], 0, "Should have columns")
            
            print(f"Data loaded successfully - Shape: {data.shape}")
            
        except Exception as e:
            self.fail(f"Data loading failed: {e}")
    
    def test_04_structure_parsing(self):
        """Test parsing of survey structure."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        if not TestSocOprosLoader.connection_successful:
            self.skipTest("Data loading required first")
        
        try:
            structure = self.loader.parse_structure()
            
            # Validate structure information
            self.assertIsInstance(structure, dict, "Should return dictionary")
            
            required_keys = [
                'total_statements', 'total_respondents', 'data_shape',
                'statements_preview', 'respondents_preview'
            ]
            
            for key in required_keys:
                self.assertIn(key, structure, f"Structure should contain {key}")
            
            # Validate counts are positive
            self.assertGreater(structure['total_statements'], 0, "Should have statements")
            self.assertGreater(structure['total_respondents'], 0, "Should have respondents")
            
            print(f"Structure parsed: {structure['total_statements']} statements, "
                  f"{structure['total_respondents']} respondents")
            
        except Exception as e:
            self.fail(f"Structure parsing failed: {e}")
    
    def test_05_responses_matrix(self):
        """Test creation of responses matrix."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        if not TestSocOprosLoader.connection_successful:
            self.skipTest("Data loading required first")
        
        try:
            responses = self.loader.get_responses_matrix()
            
            # Validate responses matrix
            self.assertIsInstance(responses, pd.DataFrame, "Should return DataFrame")
            self.assertGreater(responses.shape[0], 0, "Should have statement rows")
            self.assertGreater(responses.shape[1], 0, "Should have respondent columns")
            
            # Check that statements are properly indexed
            statements = self.loader.get_statements()
            self.assertEqual(len(statements), responses.shape[0], 
                           "Statements count should match matrix rows")
            
            # Check respondents
            respondents = self.loader.get_respondents()
            self.assertEqual(len(respondents), responses.shape[1], 
                           "Respondents count should match matrix columns")
            
            print(f"Responses matrix: {responses.shape}")
            
        except Exception as e:
            self.fail(f"Responses matrix creation failed: {e}")
    
    def test_06_statements_extraction(self):
        """Test extraction of survey statements."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        if not TestSocOprosLoader.connection_successful:
            self.skipTest("Data loading required first")
        
        statements = self.loader.get_statements()
        
        self.assertIsInstance(statements, list, "Should return list")
        self.assertGreater(len(statements), 0, "Should have statements")
        
        # Validate statement content
        for stmt in statements[:5]:  # Check first 5
            self.assertIsInstance(stmt, str, "Statements should be strings")
            self.assertGreater(len(stmt.strip()), 0, "Statements should not be empty")
        
        print(f"Extracted {len(statements)} statements")
        print(f"Sample: {statements[0][:100]}...")
    
    def test_07_respondents_extraction(self):
        """Test extraction of respondent identifiers."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        if not TestSocOprosLoader.connection_successful:
            self.skipTest("Data loading required first")
        
        respondents = self.loader.get_respondents()
        
        self.assertIsInstance(respondents, list, "Should return list")
        self.assertGreater(len(respondents), 0, "Should have respondents")
        
        print(f"Extracted {len(respondents)} respondents")
    
    def test_08_response_summary(self):
        """Test generation of response summary statistics."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        if not TestSocOprosLoader.connection_successful:
            self.skipTest("Data loading required first")
        
        try:
            summary = self.loader.get_response_summary()
            
            # Validate summary structure
            self.assertIsInstance(summary, dict, "Should return dictionary")
            
            required_keys = [
                'total_responses', 'non_null_responses', 'unique_response_values',
                'response_frequencies', 'completion_rate', 'statements_count',
                'respondents_count'
            ]
            
            for key in required_keys:
                self.assertIn(key, summary, f"Summary should contain {key}")
            
            # Validate response values are Likert scale (numeric or text)
            unique_responses = summary['unique_response_values']
            
            # Check for numeric Likert scale (1-5) or text-based responses
            numeric_responses = {'1.0', '2.0', '3.0', '4.0', '5.0', '1', '2', '3', '4', '5'}
            text_responses = {'strongly agree', 'agree', 'indifferent', 'disagree', 'strongly disagree'}
            
            actual_responses = set(str(resp).lower() for resp in unique_responses)
            numeric_overlap = numeric_responses.intersection(actual_responses)
            text_overlap = text_responses.intersection(actual_responses)
            
            # Should match either numeric or text Likert scale
            has_valid_responses = len(numeric_overlap) > 0 or len(text_overlap) > 0
            self.assertTrue(has_valid_responses, f"Should contain Likert scale responses. Found: {unique_responses}")
            
            # Validate completion rate
            self.assertGreaterEqual(summary['completion_rate'], 0, "Completion rate should be non-negative")
            self.assertLessEqual(summary['completion_rate'], 100, "Completion rate should not exceed 100%")
            
            print(f"Response summary - Completion rate: {summary['completion_rate']:.1f}%")
            print(f"Response values: {summary['unique_response_values']}")
            
        except Exception as e:
            self.fail(f"Response summary generation failed: {e}")
    
    def test_09_data_export(self):
        """Test data export functionality."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        if not TestSocOprosLoader.connection_successful:
            self.skipTest("Data loading required first")
        
        try:
            # Test DataFrame export
            df_export = self.loader.export_clean_data('dataframe')
            self.assertIsInstance(df_export, pd.DataFrame, "DataFrame export should return DataFrame")
            
            # Test CSV export
            csv_export = self.loader.export_clean_data('csv')
            self.assertIsInstance(csv_export, str, "CSV export should return string")
            
            # Test JSON export
            json_export = self.loader.export_clean_data('json')
            self.assertIsInstance(json_export, str, "JSON export should return string")
            
            print(f"Data export successful in all formats")
            
        except Exception as e:
            self.fail(f"Data export failed: {e}")
    
    def test_10_convenience_function(self):
        """Test the convenience function for loading data."""
        if not LOADER_AVAILABLE:
            self.skipTest("SocOprosLoader not available")
        
        try:
            responses, structure = load_soc_opros_data()
            
            # Validate return values
            self.assertIsInstance(responses, pd.DataFrame, "Should return DataFrame")
            self.assertIsInstance(structure, dict, "Should return structure dict")
            
            # Check structure keys
            self.assertIn('total_statements', structure)
            self.assertIn('total_respondents', structure)
            
            print(f"Convenience function working - {responses.shape}")
            
        except Exception as e:
            self.fail(f"Convenience function failed: {e}")


class TestDataConsistency(unittest.TestCase):
    """Test suite for data consistency and validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with data for consistency checks."""
        if LOADER_AVAILABLE:
            cls.loader = SocOprosLoader()
            try:
                cls.loader.load_data()
                cls.data_available = True
            except Exception as e:
                print(f"Could not load data for consistency tests: {e}")
                cls.data_available = False
        else:
            cls.data_available = False
    
    def test_data_integrity(self):
        """Test that data maintains integrity across operations."""
        if not self.data_available:
            self.skipTest("Data not available")
        
        # Load data multiple times and ensure consistency
        data1 = self.loader.load_data()
        data2 = self.loader.load_data()
        
        self.assertTrue(data1.equals(data2), "Data should be consistent across loads")
    
    def test_matrix_dimensions(self):
        """Test that matrix dimensions are consistent."""
        if not self.data_available:
            self.skipTest("Data not available")
        
        structure = self.loader.parse_structure()
        responses = self.loader.get_responses_matrix()
        statements = self.loader.get_statements()
        respondents = self.loader.get_respondents()
        
        # Check dimension consistency
        self.assertEqual(len(statements), responses.shape[0], 
                        "Statements count should match matrix rows")
        self.assertEqual(len(respondents), responses.shape[1], 
                        "Respondents count should match matrix columns")
        self.assertEqual(structure['total_statements'], len(statements))
        self.assertEqual(structure['total_respondents'], len(respondents))
    
    def test_response_value_consistency(self):
        """Test that response values are consistent and valid."""
        if not self.data_available:
            self.skipTest("Data not available")
        
        responses = self.loader.get_responses_matrix()
        summary = self.loader.get_response_summary()
        
        # Check that unique values in matrix match summary
        matrix_unique = set()
        for col in responses.columns:
            matrix_unique.update(responses[col].dropna().astype(str))
        
        summary_unique = set(summary['unique_response_values'])
        
        self.assertEqual(matrix_unique, summary_unique, 
                        "Unique values should be consistent between matrix and summary")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)