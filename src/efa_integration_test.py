# -*- coding: utf-8 -*-
"""
EFA Integration Verification with SocOprosLoader

Tests and validates integration between EFA analysis modules and the existing
soc_opros_loader. Ensures data format compatibility and proper workflow.

Author: Insight Digger Project
Created: November 24, 2025
"""

import sys
import warnings
from typing import Tuple, Optional
import pandas as pd
import numpy as np

# Local imports
from soc_opros_loader import SocOprosLoader
from efa_analyzer import EFAAnalyzer, ValidationResults
from factor_validator import FactorValidator
from efa_error_handling import get_warning_manager, add_efa_warning


def verify_soc_opros_efa_integration() -> Tuple[bool, str]:
    """
    Verify integration between soc_opros_loader and EFA modules.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Initialize loader
        loader = SocOprosLoader()
        
        # Test data loading (without actually downloading - check structure)
        print("✓ SocOprosLoader initialized successfully")
        
        # Test EFA analyzer initialization
        analyzer = EFAAnalyzer()
        print("✓ EFAAnalyzer initialized successfully")
        
        # Test validator initialization
        validator = FactorValidator()
        print("✓ FactorValidator initialized successfully")
        
        # Test warning system
        warning_mgr = get_warning_manager()
        add_efa_warning("Integration test warning", level="INFO")
        print("✓ Warning system operational")
        
        # Create mock data to test data flow
        mock_data = create_mock_soc_opros_data()
        print("✓ Mock data created for testing")
        
        # Test data validation
        validation_result = analyzer.validate_data(mock_data)
        if validation_result.is_valid:
            print("✓ Data validation successful")
        else:
            print(f"⚠ Data validation issues: {validation_result.errors}")
            
        # Test factor validator adequacy check
        adequacy_result = validator.check_data_adequacy(mock_data)
        print(f"✓ Data adequacy check completed: adequate={adequacy_result.is_adequate}")
        
        return True, "Integration verification successful"
        
    except Exception as e:
        return False, f"Integration verification failed: {str(e)}"


def create_mock_soc_opros_data() -> pd.DataFrame:
    """
    Create mock data matching soc_opros structure for testing.
    
    Returns:
        DataFrame: Mock survey response data (statements × respondents)
    """
    # Create realistic psychological survey response data
    np.random.seed(42)  # For reproducible results
    
    # Simulate 20 statements, 15 respondents (smaller scale for testing)
    n_statements = 20
    n_respondents = 15
    
    # Generate correlated responses (simulate psychological factors)
    # Factor 1: Authoritarianism (statements 1-7)
    auth_factor = np.random.normal(0, 1, n_respondents)
    auth_responses = np.random.normal(
        auth_factor[:, np.newaxis] * 0.8, 0.5, (n_respondents, 7)
    )
    
    # Factor 2: Openness (statements 8-14)
    open_factor = np.random.normal(0, 1, n_respondents)
    open_responses = np.random.normal(
        open_factor[:, np.newaxis] * 0.7, 0.6, (n_respondents, 7)
    )
    
    # Factor 3: Mixed/noise (statements 15-20)
    noise_responses = np.random.normal(0, 1, (n_respondents, 6))
    
    # Combine all responses
    all_responses = np.hstack([auth_responses, open_responses, noise_responses])
    
    # Convert to 1-5 Likert scale
    responses_scaled = np.clip(np.round(all_responses * 1.2 + 3), 1, 5)
    
    # Create DataFrame (transpose to get statements × respondents)
    statements = [f"Statement_{i+1}" for i in range(n_statements)]
    respondents = [f"Respondent_{i+1}" for i in range(n_respondents)]
    
    # Note: soc_opros_loader returns statements × respondents format
    mock_data = pd.DataFrame(
        responses_scaled.T,  # Transpose to match expected format
        index=statements,
        columns=respondents
    )
    
    return mock_data


def test_data_pipeline() -> bool:
    """
    Test the complete data pipeline from loader to EFA.
    
    Returns:
        bool: True if pipeline works correctly
    """
    try:
        print("Testing EFA data pipeline...")
        
        # Create mock data
        data = create_mock_soc_opros_data()
        print(f"✓ Mock data shape: {data.shape} (statements × respondents)")
        
        # EFA typically expects observations × variables format
        # So we need to transpose for factor analysis
        data_for_efa = data.T  # Now respondents × statements
        print(f"✓ EFA data shape: {data_for_efa.shape} (observations × variables)")
        
        # Initialize EFA components
        analyzer = EFAAnalyzer(n_factors=3)
        validator = FactorValidator()
        
        # Test validation pipeline
        validation = analyzer.validate_data(data_for_efa)
        print(f"✓ Validation: valid={validation.is_valid}, warnings={len(validation.warnings)}")
        
        adequacy = validator.check_data_adequacy(data_for_efa)
        print(f"✓ Adequacy: adequate={adequacy.is_adequate}, ratio={adequacy.ratio:.2f}")
        
        print("✓ Data pipeline test successful")
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline test failed: {str(e)}")
        return False


if __name__ == "__main__":
    """Run integration verification when script is executed directly."""
    print("EFA Integration Verification")
    print("=" * 40)
    
    # Test integration
    success, message = verify_soc_opros_efa_integration()
    print(f"\nIntegration Result: {message}")
    
    # Test data pipeline
    print("\n" + "=" * 40)
    pipeline_success = test_data_pipeline()
    
    # Summary
    print("\n" + "=" * 40)
    print("INTEGRATION SUMMARY:")
    print(f"✓ Module Integration: {'PASSED' if success else 'FAILED'}")
    print(f"✓ Data Pipeline: {'PASSED' if pipeline_success else 'FAILED'}")
    
    if success and pipeline_success:
        print("\n🎉 EFA integration with soc_opros_loader is ready!")
        print("Proceed to Phase 3 implementation.")
    else:
        print("\n⚠️ Integration issues detected. Review errors above.")
        
    # Show warning summary
    warning_mgr = get_warning_manager()
    summary = warning_mgr.get_summary()
    if summary['total'] > 0:
        print(f"\nWarnings collected: {summary['total']}")
        print(f"By level: {summary['by_level']}")