#!/usr/bin/env python3
"""
Analyze the soc opros data to determine the best correlation method for EFA.
This script examines data distribution and compares correlation methods.
"""

import sys
import os
sys.path.append('src')

from soc_opros_loader import SocOprosLoader
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def analyze_data_characteristics():
    """Analyze the basic characteristics of the soc opros data."""
    print("=== LOADING AND EXAMINING SOC OPROS DATA ===")
    
    # Load the data
    loader = SocOprosLoader()
    data = loader.load_data()
    structure = loader.parse_structure()
    responses = loader.get_responses_matrix()
    
    print(f"Data shape: {responses.shape}")
    print(f"Data types: {responses.dtypes.unique()}")
    
    # Convert to numeric, handling any string values
    numeric_responses = responses.apply(pd.to_numeric, errors='coerce')
    
    print(f"After numeric conversion: {numeric_responses.dtypes.unique()}")
    print(f"Missing values after conversion: {numeric_responses.isnull().sum().sum()}")
    
    # Basic statistics
    print(f"\n=== BASIC STATISTICS ===")
    print(f"Value range: {numeric_responses.min().min():.1f} to {numeric_responses.max().max():.1f}")
    unique_vals = sorted([x for x in pd.unique(numeric_responses.values.ravel()) if not pd.isna(x)])
    print(f"Unique values: {unique_vals}")
    print(f"Number of unique values: {len(unique_vals)}")
    
    # Distribution analysis for first 5 variables
    print(f"\n=== DISTRIBUTION ANALYSIS (First 5 Variables) ===")
    for i, col in enumerate(numeric_responses.columns[:5]):
        counts = numeric_responses[col].value_counts().sort_index()
        print(f"{col}: {dict(counts)}")
    
    return numeric_responses

def compare_correlation_methods(responses):
    """Compare different correlation methods on the data."""
    print(f"\n=== CORRELATION METHOD COMPARISON ===")
    
    # Use first 20 variables for detailed comparison (to manage computation time)
    subset = responses.iloc[:, :20].dropna()
    
    print(f"Analysis subset shape: {subset.shape}")
    
    # Calculate different correlation matrices
    corr_pearson = subset.corr(method='pearson')
    corr_spearman = subset.corr(method='spearman')
    
    # Remove diagonal (always 1.0) for comparison
    mask = np.triu(np.ones_like(corr_pearson), k=1).astype(bool)
    pearson_vals = corr_pearson.values[mask]
    spearman_vals = corr_spearman.values[mask]
    
    print(f"\nPearson correlations:")
    print(f"  Range: {pearson_vals.min():.3f} to {pearson_vals.max():.3f}")
    print(f"  Mean: {pearson_vals.mean():.3f}")
    print(f"  Std: {pearson_vals.std():.3f}")
    
    print(f"\nSpearman correlations:")
    print(f"  Range: {spearman_vals.min():.3f} to {spearman_vals.max():.3f}")
    print(f"  Mean: {spearman_vals.mean():.3f}")
    print(f"  Std: {spearman_vals.std():.3f}")
    
    # Compare the two methods
    diff_vals = np.abs(pearson_vals - spearman_vals)
    print(f"\nDifference between Pearson and Spearman:")
    print(f"  Mean absolute difference: {diff_vals.mean():.4f}")
    print(f"  Max absolute difference: {diff_vals.max():.4f}")
    print(f"  95th percentile difference: {np.percentile(diff_vals, 95):.4f}")
    
    # Correlation between the two methods
    method_corr = np.corrcoef(pearson_vals, spearman_vals)[0, 1]
    print(f"  Correlation between methods: {method_corr:.4f}")
    
    return corr_pearson, corr_spearman

def test_normality_assumptions(responses):
    """Test normality assumptions for first few variables."""
    print(f"\n=== NORMALITY TESTING (First 10 Variables) ===")
    
    subset = responses.iloc[:, :10].dropna()
    
    for i, col in enumerate(subset.columns[:10]):
        data_col = subset[col].dropna()
        
        # Shapiro-Wilk test (good for smaller samples)
        if len(data_col) <= 5000:  # Shapiro-Wilk limitation
            stat, p_value = stats.shapiro(data_col)
            normal_test = "Shapiro-Wilk"
        else:
            # Use D'Agostino's test for larger samples
            stat, p_value = stats.normaltest(data_col)
            normal_test = "D'Agostino"
            
        is_normal = p_value > 0.05
        print(f"{col}: {normal_test} p={p_value:.4f} {'(Normal)' if is_normal else '(Non-normal)'}")

def analyze_factor_analysis_suitability(responses):
    """Check basic suitability for factor analysis."""
    print(f"\n=== FACTOR ANALYSIS SUITABILITY ===")
    
    # Use reasonable subset for analysis
    subset = responses.iloc[:, :50].dropna()  # First 50 variables
    
    print(f"Analysis matrix shape: {subset.shape}")
    
    if subset.shape[0] < subset.shape[1]:
        print(f"WARNING: More variables ({subset.shape[1]}) than observations ({subset.shape[0]})")
        print("This violates basic factor analysis assumptions.")
    else:
        ratio = subset.shape[0] / subset.shape[1]
        print(f"Sample size ratio: {ratio:.1f} observations per variable")
        if ratio >= 5:
            print("✓ Meets 5:1 rule of thumb")
        elif ratio >= 3:
            print("⚠ Marginal (3:1 minimum met)")
        else:
            print("✗ Insufficient sample size")
    
    # Basic correlation matrix properties
    corr_matrix = subset.corr()
    
    # Check for potential multicollinearity
    eigenvals = np.linalg.eigvals(corr_matrix)
    min_eigenval = eigenvals.min()
    print(f"Smallest eigenvalue: {min_eigenval:.6f}")
    if min_eigenval < 1e-8:
        print("⚠ Potential multicollinearity (very small eigenvalue)")
    else:
        print("✓ No severe multicollinearity detected")
    
    return subset

def recommend_correlation_method(responses):
    """Provide recommendation based on data analysis."""
    print(f"\n=== CORRELATION METHOD RECOMMENDATION ===")
    
    subset = responses.iloc[:, :10].dropna()
    
    # Check if data is truly ordinal (limited distinct values)
    unique_counts = [len(subset[col].unique()) for col in subset.columns]
    max_unique = max(unique_counts)
    min_unique = min(unique_counts)
    
    print(f"Unique values per variable: min={min_unique}, max={max_unique}")
    
    # Determine recommendation
    if max_unique <= 7:  # Typical Likert scale range
        print("✓ Data appears to be ordinal (limited distinct values)")
        print("✓ Likert scale characteristics detected")
        
        # Check if treating as interval is reasonable
        subset_sample = responses.iloc[:, :20].dropna()
        corr_pearson = subset_sample.corr(method='pearson')
        corr_spearman = subset_sample.corr(method='spearman') 
        
        # Compare correlations
        mask = np.triu(np.ones_like(corr_pearson), k=1).astype(bool)
        pearson_vals = corr_pearson.values[mask]
        spearman_vals = corr_spearman.values[mask]
        
        correlation_between_methods = np.corrcoef(pearson_vals, spearman_vals)[0, 1]
        mean_abs_diff = np.mean(np.abs(pearson_vals - spearman_vals))
        
        print(f"Correlation between Pearson/Spearman: {correlation_between_methods:.4f}")
        print(f"Mean absolute difference: {mean_abs_diff:.4f}")
        
        if correlation_between_methods > 0.95 and mean_abs_diff < 0.05:
            recommendation = "Pearson"
            reasoning = "High agreement between methods, Pearson is standard practice"
        elif correlation_between_methods > 0.90:
            recommendation = "Pearson"
            reasoning = "Good agreement, Pearson preferred for factor analysis compatibility"
        else:
            recommendation = "Spearman"
            reasoning = "Significant differences suggest ordinal treatment is important"
            
    else:
        recommendation = "Pearson"
        reasoning = "Continuous-like data, Pearson is appropriate"
    
    print(f"\n🎯 RECOMMENDATION: {recommendation} correlations")
    print(f"📋 Reasoning: {reasoning}")
    
    # Additional context
    print(f"\n📚 CONTEXT:")
    print(f"• Most factor analysis software uses Pearson by default")
    print(f"• Psychological research typically uses Pearson for Likert scales")
    print(f"• Polychoric correlations are theoretically optimal but computationally complex")
    
    return recommendation

def main():
    """Main analysis function."""
    print("FACTOR ANALYSIS CORRELATION METHOD ANALYSIS")
    print("=" * 60)
    
    try:
        # Step 1: Basic data analysis
        responses = analyze_data_characteristics()
        
        # Step 2: Compare correlation methods
        corr_pearson, corr_spearman = compare_correlation_methods(responses)
        
        # Step 3: Test assumptions
        test_normality_assumptions(responses)
        
        # Step 4: Check FA suitability  
        analyze_factor_analysis_suitability(responses)
        
        # Step 5: Final recommendation
        recommendation = recommend_correlation_method(responses)
        
        print(f"\n" + "=" * 60)
        print(f"FINAL RECOMMENDATION: Use {recommendation} correlations for EFA")
        print(f"=" * 60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()