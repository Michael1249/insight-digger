# Quickstart Guide: Exploratory Factor Analysis (EFA)

**Target Audience**: Researchers, students, and data scientists  
**Prerequisites**: Basic Python knowledge, Jupyter notebooks  
**Time to Complete**: 15-30 minutes  
**Difficulty Level**: Intermediate

## Overview

This guide provides a hands-on introduction to exploratory factor analysis using the Insight Digger EFA module. You'll learn to identify hidden psychological dimensions in survey data through practical examples.

## What You'll Learn

- ✅ **Load and validate survey data** for factor analysis
- ✅ **Perform statistical tests** (KMO, Bartlett's) for data suitability  
- ✅ **Extract and interpret factors** using principal axis factoring
- ✅ **Visualize factor structure** with publication-ready plots
- ✅ **Compute factor scores** for individual respondents
- ✅ **Assess reliability** and solution quality

## Quick Start (5 minutes)

### Step 1: Load the Demo Data

```python
# Load required modules
import sys
sys.path.append('../src')

from soc_opros_loader import SocOprosLoader
from efa_analyzer import EFAAnalyzer
from factor_validator import FactorValidator
from visualization_utils import EFAVisualizer

# Load survey data (265 psychological statements × 15 respondents)
loader = SocOprosLoader()
data = loader.get_responses_matrix()
print(f"Data shape: {data.shape}")
print(f"Sample: {data.iloc[:3, :3]}")  # Preview first 3×3
```

### Step 2: Run Basic Factor Analysis

```python
# Quick factor analysis with defaults
efa = EFAAnalyzer(n_factors=None, rotation_method='oblimin')

# Validate and analyze
validation = efa.validate_data(data)
if validation.is_suitable:
    efa.fit(data)
    print(f"✅ Extracted {efa.factor_loadings_.shape[1]} factors")
    print(f"📊 Total variance explained: {efa.get_reliability_stats()['total_variance_explained']:.1%}")
else:
    print(f"⚠️ Data issues: {validation.warnings}")
```

### Step 3: Visualize Results

```python
# Create summary visualization
viz = EFAVisualizer(publication_ready=True)

# Generate scree plot and factor loadings
scree_fig = viz.plot_scree(efa.eigenvalues_)
loadings_fig = viz.plot_loadings_heatmap(efa.factor_loadings_)

scree_fig.show()
loadings_fig.show()
```

**🎉 Congratulations!** You've completed your first factor analysis. The plots show how many factors were extracted and which survey statements group together.

---

## Complete Tutorial (30 minutes)

### Part 1: Understanding Your Data

#### Data Exploration

```python
# Get detailed information about the survey
statements = loader.get_statements()
respondents = loader.get_respondents()

print(f"📋 Survey contains {len(statements)} statements")
print(f"👥 Responses from {len(respondents)} participants")
print(f"\nFirst few statements:")
for i, stmt in enumerate(statements[:3]):
    print(f"  {i+1}. {stmt[:60]}...")

print(f"\nRespondent names: {respondents[:5]}...")  # First 5 names
```

#### Data Orientation Check

```python
# IMPORTANT: Factor analysis requires more observations than variables
n_statements, n_respondents = data.shape
print(f"Current orientation: {n_statements} statements × {n_respondents} respondents")

if n_statements > n_respondents:
    print("⚠️ WARNING: More variables than observations!")
    print("💡 Consider transposing data: analyze 265 observations × 15 psychological traits")
    
    # Transpose for proper factor analysis
    data_transposed = data.T
    print(f"Transposed shape: {data_transposed.shape}")
    
    # Use transposed data for analysis
    analysis_data = data_transposed
else:
    analysis_data = data
```

### Part 2: Comprehensive Validation

#### Statistical Prerequisites

```python
# Initialize validator with standard thresholds
validator = FactorValidator(
    kmo_threshold=0.6,          # Minimum acceptable KMO
    min_sample_ratio=3.0,       # 3:1 observations to variables
    loading_threshold=0.40      # Practical significance threshold
)

# Step 1: Check data adequacy
adequacy = validator.check_data_adequacy(analysis_data)
print("📊 DATA ADEQUACY ASSESSMENT")
print(f"Overall adequate: {adequacy.is_adequate}")
print(f"Sample size ratio: {adequacy.sample_size_ratio:.1f}:1")
print(f"Missing data: {adequacy.missing_data_percent:.1f}%")

if adequacy.issues:
    print("\n⚠️ Issues found:")
    for issue in adequacy.issues:
        print(f"  • {issue}")

# Step 2: Test factorability
factorability = validator.test_factorability(analysis_data)
print(f"\n🔬 FACTORABILITY TESTS")
print(f"KMO Overall: {factorability.kmo_overall:.3f} ({factorability.kmo_interpretation})")
print(f"Bartlett's test: χ² = {factorability.bartlett_chi_square:.1f}, p < {factorability.bartlett_p_value:.3e}")
print(f"Suitable for factor analysis: {factorability.is_factorable}")

if factorability.warnings:
    print("\n⚠️ Warnings:")
    for warning in factorability.warnings:
        print(f"  • {warning}")
```

#### Assumption Testing

```python
# Test statistical assumptions
assumptions = validator.check_assumptions(analysis_data)
print(f"\n📈 ASSUMPTION CHECKS")

if assumptions.assumption_violations:
    print("Assumption violations detected:")
    for violation in assumptions.assumption_violations:
        severity = assumptions.severity_levels.get(violation, 'UNKNOWN')
        print(f"  • {violation} (Severity: {severity})")
else:
    print("✅ No major assumption violations detected")

# Show robustness notes
if assumptions.robustness_notes:
    print("\n💡 Robustness notes:")
    for note in assumptions.robustness_notes:
        print(f"  • {note}")
```

### Part 3: Advanced Factor Analysis

#### Determining Optimal Factors

```python
# Initialize analyzer for factor determination
efa = EFAAnalyzer(n_factors=None, rotation_method='oblimin')

# Automatically determine optimal number of factors
optimal_factors = efa.determine_factors(analysis_data, max_factors=8)
print(f"🎯 Recommended factors: {optimal_factors}")

# Create scree plot for visual inspection
viz = EFAVisualizer(publication_ready=True)
eigenvalues = np.linalg.eigvals(np.corrcoef(analysis_data.T))
eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

scree_fig = viz.plot_scree(
    eigenvalues=eigenvalues,
    n_factors=optimal_factors,
    title=f"Scree Plot: Optimal Factors = {optimal_factors}"
)
scree_fig.show()
```

#### Factor Extraction and Rotation

```python
# Run factor analysis with optimal number
efa_final = EFAAnalyzer(
    n_factors=optimal_factors,
    extraction_method='principal',    # Principal axis factoring
    rotation_method='oblimin',        # Oblique rotation (allows correlations)
    max_iterations=100
)

# Fit the model
print(f"🔄 Running factor analysis with {optimal_factors} factors...")
efa_final.fit(analysis_data)
print("✅ Factor analysis completed successfully!")

# Display basic results
loadings = efa_final.factor_loadings_
eigenvals = efa_final.eigenvalues_
communalities = efa_final.communalities_

print(f"\n📈 RESULTS SUMMARY")
print(f"Factors extracted: {loadings.shape[1]}")
print(f"Variables analyzed: {loadings.shape[0]}")
print(f"Total variance explained: {eigenvals.sum():.1%}")
print(f"Average communality: {communalities.mean():.3f}")
```

### Part 4: Factor Interpretation

#### Loading Analysis

```python
# Analyze factor loadings
interpretation = efa_final.interpret_factors(
    statement_labels=None,  # Use default indices
    loading_threshold=0.40
)

print(f"🧠 FACTOR INTERPRETATION")
for factor_name, details in interpretation.items():
    if factor_name == 'overall':
        continue
        
    print(f"\n{factor_name.upper()}:")
    print(f"  Reliability (α): {details['reliability']:.3f}")
    
    # Show top loadings
    top_loadings = details['primary_loadings'][:5]  # Top 5
    print(f"  Top indicators:")
    for var_name, loading in top_loadings:
        print(f"    {var_name}: {loading:+.3f}")
    
    # Interpretation suggestions
    if details['interpretation_suggestions']:
        print(f"  Suggested theme: {details['interpretation_suggestions'][0]}")

# Overall solution quality
overall = interpretation['overall']
print(f"\n📊 OVERALL SOLUTION QUALITY")
print(f"Simple structure index: {overall['simple_structure_index']:.3f}")
print(f"Total variance explained: {overall['total_variance_explained']:.1%}")
```

#### Visualization Suite

```python
# Create comprehensive visualization
print("🎨 Generating visualizations...")

# 1. Factor loadings heatmap
loadings_fig = viz.plot_loadings_heatmap(
    loadings=loadings,
    loading_threshold=0.40,
    cluster_variables=True,
    title="Factor Loadings Pattern"
)

# 2. Factor score distributions
scores = efa_final.compute_factor_scores(analysis_data)
scores_fig = viz.plot_factor_scores_distribution(
    factor_scores=scores,
    plot_type='violin',
    title="Factor Score Distributions"
)

# 3. Biplot (first 2 factors)
if loadings.shape[1] >= 2:
    biplot_fig = viz.plot_biplot(
        factor_scores=scores,
        loadings=loadings,
        factor_x=0, factor_y=1,
        title="Factor Biplot (F1 vs F2)"
    )
    biplot_fig.show()

loadings_fig.show()
scores_fig.show()
```

### Part 5: Quality Assessment

#### Solution Validation

```python
# Validate the extracted solution
solution_quality = validator.validate_factor_solution(
    loadings=loadings,
    eigenvalues=eigenvals,
    communalities=communalities,
    data=analysis_data
)

print(f"✅ SOLUTION QUALITY ASSESSMENT")
print(f"Simple structure index: {solution_quality.simple_structure_index:.3f}")
print(f"Cross-loadings (>0.4): {solution_quality.cross_loadings_count}")
print(f"Low communalities (<0.2): {len(solution_quality.low_communalities)}")
print(f"Quality rating: {solution_quality.quality_rating}")

# Reliability statistics
reliability = solution_quality.reliability_stats
print(f"\n🔍 RELIABILITY ANALYSIS")
for factor, alpha in reliability.items():
    status = "✅ Good" if alpha >= 0.7 else "⚠️ Marginal" if alpha >= 0.6 else "❌ Poor"
    print(f"  {factor}: α = {alpha:.3f} {status}")

# Heywood cases (if any)
if solution_quality.heywood_cases:
    print(f"\n⚠️ Heywood cases detected: {solution_quality.heywood_cases}")
    print("   (Communalities > 1.0 indicate estimation problems)")
```

#### Improvement Recommendations

```python
# Get actionable recommendations
recommendations = validator.recommend_improvements(solution_quality)

if recommendations:
    print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print(f"\n🎉 No major improvements needed! Solution quality is satisfactory.")
```

### Part 6: Export and Reporting

#### Factor Scores Export

```python
# Create factor scores with meaningful names
factor_names = [f"Factor_{i+1}" for i in range(scores.shape[1])]
scores_df = pd.DataFrame(scores, 
                        index=analysis_data.index,
                        columns=factor_names)

# Add descriptive statistics
scores_summary = scores_df.describe()
print(f"📊 FACTOR SCORES SUMMARY")
print(scores_summary)

# Export for further analysis
scores_df.to_csv("factor_scores.csv")
print(f"\n💾 Factor scores exported to 'factor_scores.csv'")
```

#### Comprehensive Report

```python
# Generate markdown report
report = f"""
# Factor Analysis Report

## Data Overview
- **Variables**: {analysis_data.shape[1]}
- **Observations**: {analysis_data.shape[0]}
- **Factors Extracted**: {loadings.shape[1]}

## Statistical Validation
- **KMO Overall**: {factorability.kmo_overall:.3f} ({factorability.kmo_interpretation})
- **Bartlett's Test**: p < {factorability.bartlett_p_value:.3e}
- **Suitable**: {factorability.is_factorable}

## Solution Quality
- **Total Variance Explained**: {eigenvals.sum():.1%}
- **Average Reliability**: {np.mean(list(reliability.values())):.3f}
- **Simple Structure Index**: {solution_quality.simple_structure_index:.3f}

## Factor Interpretation
"""

for i, (factor_name, details) in enumerate(interpretation.items()):
    if factor_name == 'overall':
        continue
    
    report += f"""
### {factor_name}
- **Reliability (α)**: {details['reliability']:.3f}
- **Primary Indicators**: {len(details['primary_loadings'])} variables
- **Top Loading**: {details['primary_loadings'][0][1]:.3f}
"""

# Save report
with open("efa_report.md", "w") as f:
    f.write(report)
    
print(f"📄 Comprehensive report saved to 'efa_report.md'")
```

---

## Common Issues & Solutions

### Problem: "Data not suitable for factor analysis"

**Symptoms**: Low KMO (<0.6), high Bartlett's p-value
**Solutions**:
```python
# Check correlation matrix
corr_matrix = analysis_data.corr()
print(f"Average correlation: {corr_matrix.abs().mean().mean():.3f}")

# Remove variables with low communalities
low_comm_vars = communalities[communalities < 0.2].index
if len(low_comm_vars) > 0:
    print(f"Consider removing: {list(low_comm_vars)}")
    data_filtered = analysis_data.drop(columns=low_comm_vars)
```

### Problem: "Too many/too few factors extracted"

**Symptoms**: Unclear factor structure, poor interpretability
**Solutions**:
```python
# Try different factor numbers
for n in range(2, 7):
    efa_test = EFAAnalyzer(n_factors=n, rotation_method='oblimin')
    efa_test.fit(analysis_data)
    reliability = efa_test.get_reliability_stats()
    
    print(f"{n} factors: Total variance = {reliability['total_variance_explained']:.1%}")
```

### Problem: "Factor scores don't make sense"

**Symptoms**: Unexpected score distributions, poor reliability
**Solutions**:
```python
# Try different rotation methods
for rotation in ['oblimin', 'varimax', 'quartimax']:
    efa_rot = EFAAnalyzer(n_factors=optimal_factors, rotation_method=rotation)
    efa_rot.fit(analysis_data)
    
    interpretation = efa_rot.interpret_factors()
    simple_structure = interpretation['overall']['simple_structure_index']
    print(f"{rotation}: Simple structure = {simple_structure:.3f}")
```

## Next Steps

### Advanced Topics
1. **Confirmatory Factor Analysis**: Test specific factor models
2. **Multi-group Analysis**: Compare factors across groups
3. **Longitudinal Factor Analysis**: Track factors over time
4. **Robust Methods**: Handle outliers and non-normality

### Learning Resources
- **Fabrigar et al. (1999)**: "Evaluating the use of exploratory factor analysis"
- **Costello & Osborne (2005)**: "Best practices in exploratory factor analysis"
- **Python Factor Analysis**: `factor_analyzer` library documentation

### Practical Applications
- **Psychology**: Personality traits, cognitive abilities
- **Marketing**: Consumer preferences, brand perception
- **Education**: Learning styles, academic motivation
- **Health**: Quality of life measures, symptom clusters

---

## Troubleshooting

### Common Error Messages

**`ValueError: Sample size insufficient`**
- Increase sample size or reduce variables
- Consider transposing data matrix

**`LinAlgError: Singular matrix`**  
- Remove perfectly correlated variables
- Check for constant variables (zero variance)

**`ConvergenceError: Factor analysis failed to converge`**
- Increase max_iterations
- Try different extraction method
- Check data quality

### Getting Help

- **Documentation**: Check module docstrings and contracts
- **Community**: Post issues on project GitHub
- **Examples**: Browse additional notebooks in `/notebooks/`

**🎓 You're now ready to perform professional-quality factor analysis!**