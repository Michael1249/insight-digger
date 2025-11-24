# Comprehensive Best Practices for Implementing Exploratory Factor Analysis in Python for Psychological Research

*Research conducted November 24, 2025*

## Executive Summary

This document provides evidence-based recommendations for implementing Exploratory Factor Analysis (EFA) in Python for psychological research. The recommendations are derived from established statistical literature, Python library documentation, and current best practices in the field.

---

## 1. Technical Implementation

### 1.1 Python Library Recommendations

#### **Primary Recommendation: `factor_analyzer` Package**
- **Best for**: Comprehensive EFA with psychological research focus
- **Strengths**: 
  - Purpose-built for EFA/CFA with multiple extraction methods
  - MINRES, Maximum Likelihood, and Principal Factor solutions
  - Built-in KMO and Bartlett's tests
  - Multiple rotation options (varimax, oblimin, promax)
  - Compatible with scikit-learn ecosystem
- **Installation**: `pip install factor_analyzer`

#### **Secondary Option: `scikit-learn` FactorAnalysis**
- **Best for**: Integration with ML pipelines
- **Limitations**: 
  - Limited to Maximum Likelihood estimation
  - No built-in rotation options
  - Fewer diagnostic tools
- **Use case**: When EFA is part of larger ML workflow

#### **Not Recommended: `scikit-learn` PCA**
- PCA ≠ EFA: Different assumptions and interpretations
- PCA maximizes variance; EFA models latent constructs

### 1.2 Extraction Methods

#### **Recommended Extraction Method Hierarchy:**

1. **Principal Axis Factoring (PAF)** - *Most recommended*
   - Best for psychological constructs
   - Removes unique variance, focuses on common variance
   - Robust to violations of multivariate normality

```python
from factor_analyzer import FactorAnalyzer

# Principal Axis Factoring implementation
fa = FactorAnalyzer(
    n_factors=None,  # Determine optimal number first
    rotation=None,   # Apply rotation after extraction
    method='principal',  # Principal axis factoring
    use_smc=True    # Use squared multiple correlations
)
```

2. **Maximum Likelihood (ML)**
   - When data meets multivariate normality assumptions
   - Provides fit statistics and significance tests
   - Required for some advanced analyses

```python
fa_ml = FactorAnalyzer(
    n_factors=3,
    method='ml',  # Maximum likelihood
    rotation='varimax'
)
```

3. **Minimum Residual (MINRES)**
   - Good compromise between PAF and ML
   - Less sensitive to normality violations

### 1.3 Rotation Methods

#### **Orthogonal Rotations:**
- **Varimax** (most common): Maximizes variance of squared loadings
- **Quartimax**: Maximizes loadings on first factor

#### **Oblique Rotations:**
- **Direct Oblimin**: Allows factors to correlate
- **Promax**: Computationally efficient oblique rotation

**Decision Rule**: Use oblique rotation unless theoretical reasons require orthogonal factors.

```python
# Oblique rotation example
fa = FactorAnalyzer(
    n_factors=4,
    rotation='oblimin',
    method='principal'
)
```

---

## 2. Statistical Validation

### 2.1 Pre-Analysis Diagnostics

#### **Kaiser-Meyer-Olkin (KMO) Test**
```python
from factor_analyzer import calculate_kmo

# Calculate KMO
kmo_all, kmo_model = calculate_kmo(data)
```

**Interpretation Thresholds:**
- KMO > 0.9: "Marvelous" (Kaiser's terminology)
- KMO 0.8-0.9: "Meritorious" 
- KMO 0.7-0.8: "Middling"
- KMO 0.6-0.7: "Mediocre"
- KMO 0.5-0.6: "Miserable"
- KMO < 0.5: "Unacceptable"

**Minimum Acceptable**: KMO ≥ 0.6 (conservative: ≥ 0.5)

#### **Bartlett's Test of Sphericity**
```python
from factor_analyzer import calculate_bartlett_sphericity

# Test for sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(data)
```

**Interpretation**: p < 0.05 indicates correlations exist (good for factor analysis)

### 2.2 Factor Score Computation

#### **Regression Method (Recommended)**
```python
# Compute factor scores using regression method
factor_scores = fa.transform(data)
```

**Alternative Methods:**
- **Bartlett method**: Unbiased but potentially correlated
- **Anderson-Rubin**: Orthogonal and standardized

### 2.3 Missing Data Handling

#### **Recommended Approaches:**

1. **Pairwise Deletion** (Default recommendation)
   - Uses all available data for each correlation
   - Preserves sample size where possible
   - May lead to non-positive definite matrices

```python
# Pandas correlation with pairwise deletion
correlation_matrix = data.corr(method='pearson', min_periods=1)
```

2. **Listwise Deletion** (Conservative approach)
   - Complete case analysis only
   - Ensures consistent sample across all analyses
   - May significantly reduce sample size

```python
# Complete cases only
data_complete = data.dropna()
```

3. **Multiple Imputation** (Advanced approach)
   - Most sophisticated method
   - Requires additional assumptions
   - Use libraries like `scikit-learn.impute` or `fancyimpute`

---

## 3. Factor Interpretation

### 3.1 Determining Optimal Number of Factors

#### **Multi-Criteria Approach (Recommended):**

1. **Kaiser Criterion**: Eigenvalues > 1.0
2. **Scree Plot Analysis**: Visual inspection of eigenvalue plot
3. **Parallel Analysis**: Compare with random data eigenvalues
4. **Theoretical Considerations**: Domain knowledge

```python
import matplotlib.pyplot as plt
import numpy as np

# Eigenvalue analysis
eigenvalues, _ = np.linalg.eig(correlation_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot for Factor Analysis')
plt.legend()
plt.grid(True)
plt.show()
```

#### **Parallel Analysis Implementation:**
```python
from factor_analyzer import FactorAnalyzer
import numpy as np

def parallel_analysis(data, n_iterations=100):
    """
    Perform parallel analysis to determine number of factors
    """
    n_vars = data.shape[1]
    n_obs = data.shape[0]
    
    # Real data eigenvalues
    fa_real = FactorAnalyzer(rotation=None)
    fa_real.fit(data)
    real_eigenvals = fa_real.get_eigenvalues()[0]
    
    # Random data eigenvalues
    random_eigenvals = []
    for _ in range(n_iterations):
        random_data = np.random.normal(size=(n_obs, n_vars))
        fa_random = FactorAnalyzer(rotation=None)
        fa_random.fit(random_data)
        random_eigenvals.append(fa_random.get_eigenvalues()[0])
    
    # Average random eigenvalues
    mean_random = np.mean(random_eigenvals, axis=0)
    
    # Determine number of factors
    n_factors = np.sum(real_eigenvals > mean_random)
    
    return n_factors, real_eigenvals, mean_random
```

### 3.2 Factor Loading Interpretation

#### **Loading Significance Thresholds:**

**Field Standards:**
- **≥ |0.30|**: Minimal level for interpretation
- **≥ |0.40|**: More stringent criterion
- **≥ |0.50|**: Practically significant
- **≥ |0.70|**: Well-defined factor structure

**Sample Size Considerations:**
- Larger samples can support lower loading thresholds
- Small samples (n < 100) require higher thresholds (≥ |0.50|)

#### **Simple Structure Assessment:**
```python
def assess_simple_structure(loadings, threshold=0.40):
    """
    Assess the simple structure of factor loadings
    """
    # Count significant loadings per variable
    significant_loadings = np.abs(loadings) >= threshold
    loadings_per_var = np.sum(significant_loadings, axis=1)
    
    # Ideal: each variable loads on only one factor
    simple_structure_vars = np.sum(loadings_per_var == 1)
    total_vars = loadings.shape[0]
    
    simple_structure_pct = (simple_structure_vars / total_vars) * 100
    
    return simple_structure_pct, loadings_per_var
```

### 3.3 Factor Reliability Assessment

#### **Cronbach's Alpha for Internal Consistency:**
```python
from scipy import stats
import numpy as np

def cronbach_alpha(items):
    """
    Calculate Cronbach's Alpha for a set of items
    """
    items_df = items.dropna()
    n_items = items_df.shape[1]
    
    # Variance of each item
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    
    # Cronbach's Alpha formula
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    
    return alpha
```

**Reliability Thresholds:**
- α ≥ 0.90: Excellent
- α ≥ 0.80: Good
- α ≥ 0.70: Acceptable
- α ≥ 0.60: Questionable
- α < 0.60: Poor

---

## 4. Visualization Standards

### 4.1 Required Plots for EFA

#### **1. Scree Plot**
```python
def create_scree_plot(eigenvalues, title="Scree Plot"):
    """
    Create publication-ready scree plot
    """
    plt.figure(figsize=(10, 6))
    factors = range(1, len(eigenvalues) + 1)
    
    plt.plot(factors, eigenvalues, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2, 
                label='Kaiser Criterion (eigenvalue = 1)')
    
    plt.xlabel('Factor Number', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()
```

#### **2. Factor Loading Plot**
```python
import seaborn as sns

def plot_factor_loadings(loadings, factor_names=None, variable_names=None):
    """
    Create heatmap of factor loadings
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(loadings, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.3f',
                xticklabels=factor_names or [f'Factor {i+1}' for i in range(loadings.shape[1])],
                yticklabels=variable_names or [f'Var {i+1}' for i in range(loadings.shape[0])])
    
    plt.title('Factor Loading Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Variables', fontsize=12)
    plt.xlabel('Factors', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()
```

#### **3. Factor Score Plot (Biplot)**
```python
def create_factor_biplot(scores, loadings, pc1=0, pc2=1):
    """
    Create biplot of factor scores and loadings
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scores
    ax.scatter(scores[:, pc1], scores[:, pc2], alpha=0.6, s=50)
    
    # Plot loading vectors
    for i, (x, y) in enumerate(loadings[:, [pc1, pc2]]):
        ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                fc='red', ec='red', alpha=0.8)
        ax.text(x*1.1, y*1.1, f'Var{i+1}', fontsize=9, 
               ha='center', va='center')
    
    ax.set_xlabel(f'Factor {pc1+1}', fontsize=12)
    ax.set_ylabel(f'Factor {pc2+1}', fontsize=12)
    ax.set_title('Factor Analysis Biplot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig
```

### 4.2 Publication Standards

#### **Figure Quality Guidelines:**
- **DPI**: Minimum 300 for publications
- **Format**: Vector formats (PDF, SVG) preferred
- **Colors**: Use colorblind-friendly palettes
- **Fonts**: 10-12pt for labels, 14pt for titles

```python
# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans']
})
```

---

## 5. Integration with Survey Data

### 5.1 Likert Scale Data Preprocessing

#### **Recommended Preprocessing Steps:**

```python
def preprocess_likert_data(data, likert_cols, reverse_items=None):
    """
    Preprocess Likert scale data for factor analysis
    """
    processed_data = data[likert_cols].copy()
    
    # Reverse scoring if specified
    if reverse_items:
        max_value = processed_data[likert_cols].max().max()
        min_value = processed_data[likert_cols].min().min()
        
        for item in reverse_items:
            if item in processed_data.columns:
                processed_data[item] = (max_value + min_value) - processed_data[item]
    
    # Check for out-of-range values
    print("Data range check:")
    print(f"Min: {processed_data.min().min()}")
    print(f"Max: {processed_data.max().max()}")
    
    # Basic descriptive statistics
    print("\nDescriptive Statistics:")
    print(processed_data.describe())
    
    return processed_data
```

### 5.2 Correlation Matrix for Ordinal Data

#### **Polychoric Correlations (Ideal but Complex):**
```python
# Note: Requires specialized libraries
# from factor_analyzer.utils import corr
# polychoric_corr = corr(data, method='polychoric')
```

#### **Pearson Correlations (Practical Alternative):**
```python
def compute_correlation_matrix(data, method='pearson'):
    """
    Compute correlation matrix with diagnostics
    """
    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)
    
    # Check for multicollinearity
    eigenvals = np.linalg.eigvals(corr_matrix)
    condition_number = np.max(eigenvals) / np.min(eigenvals)
    
    print(f"Condition Number: {condition_number:.2f}")
    if condition_number > 1000:
        print("Warning: Potential multicollinearity detected")
    
    # Check determinant
    det = np.linalg.det(corr_matrix)
    print(f"Determinant: {det:.6f}")
    if det < 0.00001:
        print("Warning: Matrix near-singular")
    
    return corr_matrix
```

### 5.3 Sample Size Requirements

#### **General Guidelines:**

**Minimum Requirements:**
- **Absolute minimum**: 50 cases
- **Conservative minimum**: 100 cases
- **Preferred minimum**: 200 cases

**Ratio-Based Guidelines:**
- **5:1 ratio**: 5 participants per variable (minimum)
- **10:1 ratio**: 10 participants per variable (preferred)
- **20:1 ratio**: 20 participants per variable (ideal)

```python
def assess_sample_adequacy(n_participants, n_variables):
    """
    Assess sample size adequacy for factor analysis
    """
    ratio = n_participants / n_variables
    
    print(f"Sample size: {n_participants}")
    print(f"Number of variables: {n_variables}")
    print(f"Participant-to-variable ratio: {ratio:.1f}:1")
    
    if ratio >= 20:
        print("✓ Excellent sample size")
    elif ratio >= 10:
        print("✓ Good sample size")
    elif ratio >= 5:
        print("⚠ Adequate sample size (minimum acceptable)")
    else:
        print("✗ Insufficient sample size")
    
    return ratio
```

#### **Power Analysis for Factor Analysis:**
```python
def estimate_power_fa(n_sample, n_factors, n_variables, alpha=0.05):
    """
    Rough estimate of power for factor analysis
    Based on communality and factor loading expectations
    """
    # Simplified power estimation
    # This is a rough approximation
    effect_size = 0.30  # Medium effect size assumption
    
    # Degrees of freedom
    df = ((n_variables * (n_variables - 1)) / 2) - (n_variables * n_factors - (n_factors * (n_factors - 1)) / 2)
    
    # Very rough power estimate
    if n_sample >= 200:
        estimated_power = 0.90
    elif n_sample >= 100:
        estimated_power = 0.80
    else:
        estimated_power = 0.60
    
    print(f"Estimated power: {estimated_power:.2f}")
    print("Note: This is a rough approximation. Use specialized software for precise power analysis.")
    
    return estimated_power
```

---

## 6. Complete Implementation Template

### 6.1 Full EFA Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import chi2

def complete_efa_analysis(data, variable_names=None, max_factors=None):
    """
    Complete EFA workflow with all diagnostic checks
    """
    print("="*60)
    print("EXPLORATORY FACTOR ANALYSIS - COMPLETE WORKFLOW")
    print("="*60)
    
    # 1. Data preprocessing
    print("\n1. DATA PREPROCESSING")
    print("-" * 30)
    
    # Remove missing data
    data_clean = data.dropna()
    n_original = len(data)
    n_clean = len(data_clean)
    print(f"Original sample size: {n_original}")
    print(f"Complete cases: {n_clean}")
    print(f"Missing data removed: {n_original - n_clean} ({((n_original - n_clean)/n_original)*100:.1f}%)")
    
    # Basic descriptive statistics
    print("\nDescriptive Statistics:")
    print(data_clean.describe().round(3))
    
    # 2. Factorability assessment
    print("\n\n2. FACTORABILITY ASSESSMENT")
    print("-" * 30)
    
    # KMO test
    kmo_all, kmo_model = calculate_kmo(data_clean)
    print(f"KMO Test:")
    print(f"  Overall KMO: {kmo_model:.3f}")
    
    if kmo_model >= 0.9:
        kmo_interpretation = "Marvelous"
    elif kmo_model >= 0.8:
        kmo_interpretation = "Meritorious"
    elif kmo_model >= 0.7:
        kmo_interpretation = "Middling"
    elif kmo_model >= 0.6:
        kmo_interpretation = "Mediocre"
    elif kmo_model >= 0.5:
        kmo_interpretation = "Miserable"
    else:
        kmo_interpretation = "Unacceptable"
    
    print(f"  Interpretation: {kmo_interpretation}")
    
    # Bartlett's test
    chi_square, p_value = calculate_bartlett_sphericity(data_clean)
    print(f"\nBartlett's Test of Sphericity:")
    print(f"  Chi-square: {chi_square:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # 3. Determine number of factors
    print("\n\n3. FACTOR NUMBER DETERMINATION")
    print("-" * 30)
    
    # Initial analysis for eigenvalues
    fa_initial = FactorAnalyzer(rotation=None)
    fa_initial.fit(data_clean)
    eigenvalues, _ = fa_initial.get_eigenvalues()
    
    # Kaiser criterion
    kaiser_factors = np.sum(eigenvalues > 1.0)
    print(f"Kaiser Criterion (eigenvalue > 1): {kaiser_factors} factors")
    
    # Parallel analysis (simplified)
    n_factors_parallel, real_eigs, random_eigs = parallel_analysis(data_clean)
    print(f"Parallel Analysis suggests: {n_factors_parallel} factors")
    
    # Create scree plot
    create_scree_plot(eigenvalues, "Scree Plot - Eigenvalues")
    plt.show()
    
    # 4. Final factor analysis
    print("\n\n4. FACTOR ANALYSIS RESULTS")
    print("-" * 30)
    
    # Use parallel analysis result or user specification
    n_factors_final = max_factors if max_factors else n_factors_parallel
    print(f"Extracting {n_factors_final} factors...")
    
    # Final EFA with rotation
    fa_final = FactorAnalyzer(
        n_factors=n_factors_final,
        rotation='varimax',  # Can be changed to 'oblimin' for oblique
        method='principal'   # Principal axis factoring
    )
    fa_final.fit(data_clean)
    
    # Get results
    loadings = fa_final.loadings_
    communalities = fa_final.get_communalities()
    variance = fa_final.get_factor_variance()
    
    # Display loadings
    print("\nFactor Loadings:")
    loadings_df = pd.DataFrame(
        loadings,
        index=variable_names or data_clean.columns,
        columns=[f'Factor {i+1}' for i in range(n_factors_final)]
    )
    print(loadings_df.round(3))
    
    # Display communalities
    print(f"\nCommunalities:")
    comm_df = pd.DataFrame({
        'Variable': variable_names or data_clean.columns,
        'Communality': communalities
    })
    print(comm_df.round(3))
    
    # Variance explained
    print(f"\nVariance Explained:")
    variance_df = pd.DataFrame({
        'Factor': [f'Factor {i+1}' for i in range(n_factors_final)],
        'Eigenvalue': variance[0],
        'Variance %': variance[1] * 100,
        'Cumulative %': np.cumsum(variance[1]) * 100
    })
    print(variance_df.round(3))
    
    # 5. Visualization
    print("\n\n5. VISUALIZATIONS")
    print("-" * 30)
    
    # Loading plot
    plot_factor_loadings(loadings, variable_names=variable_names or data_clean.columns)
    plt.show()
    
    # Factor scores
    factor_scores = fa_final.transform(data_clean)
    
    # Biplot (if 2+ factors)
    if n_factors_final >= 2:
        create_factor_biplot(factor_scores, loadings)
        plt.show()
    
    # 6. Interpretation guidelines
    print("\n\n6. INTERPRETATION GUIDELINES")
    print("-" * 30)
    
    print("Factor Loading Interpretation:")
    print("  |0.30|+ : Minimal level for interpretation")
    print("  |0.40|+ : More stringent criterion")
    print("  |0.50|+ : Practically significant")
    print("  |0.70|+ : Well-defined factor")
    
    # Simple structure assessment
    simple_pct, loadings_per_var = assess_simple_structure(loadings, 0.40)
    print(f"\nSimple Structure Assessment:")
    print(f"  Variables with single significant loading: {simple_pct:.1f}%")
    
    return {
        'fa_model': fa_final,
        'loadings': loadings,
        'communalities': communalities,
        'variance_explained': variance,
        'factor_scores': factor_scores,
        'kmo': kmo_model,
        'bartlett_p': p_value,
        'eigenvalues': eigenvalues
    }

# Example usage
if __name__ == "__main__":
    # Load your data
    # data = pd.read_csv('your_survey_data.csv')
    # variable_names = ['Item1', 'Item2', 'Item3', ...]
    
    # Run complete EFA analysis
    # results = complete_efa_analysis(data[variable_names], variable_names)
    pass
```

---

## 7. Recommendations Summary

### 7.1 Critical Decision Points

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| **Python Library** | `factor_analyzer` | Purpose-built, comprehensive features |
| **Extraction Method** | Principal Axis Factoring | Best for psychological constructs |
| **Rotation** | Oblique (oblimin) unless theoretical constraints | Factors likely correlated in psychology |
| **Missing Data** | Pairwise deletion (default) | Maximizes data utilization |
| **Factor Number** | Parallel analysis + theory | Most robust method |
| **Loading Threshold** | |0.40|+ (conservative) | Balances interpretation and rigor |
| **Sample Size** | 10:1 participant-to-variable ratio | Ensures stable factor structure |

### 7.2 Quality Assurance Checklist

**Before Analysis:**
- [ ] KMO ≥ 0.6
- [ ] Bartlett's test p < 0.05
- [ ] Sample size ≥ 5:1 ratio (preferably 10:1)
- [ ] Data screening completed
- [ ] Reverse scoring applied where needed

**During Analysis:**
- [ ] Multiple methods for determining factor number
- [ ] Appropriate extraction method selected
- [ ] Rotation method justified
- [ ] Convergence achieved

**After Analysis:**
- [ ] Simple structure assessment
- [ ] Communalities reasonable (0.40-0.90)
- [ ] Loadings interpretable
- [ ] Factors theoretically meaningful
- [ ] Reliability analysis conducted

### 7.3 Jupyter Notebook Implementation

For educational and research purposes, organize analysis in clear sections:

```markdown
# Exploratory Factor Analysis - [Study Name]

## 1. Data Import and Preprocessing
## 2. Factorability Assessment  
## 3. Factor Number Determination
## 4. Factor Extraction and Rotation
## 5. Results Interpretation
## 6. Visualizations
## 7. Reliability Analysis
## 8. Conclusions and Recommendations
```

---

## References

- Bryant, F. B., & Yarnold, P. R. (1995). Principal components analysis and exploratory and confirmatory factor analysis. *Reading and understanding multivariate analysis*.

- Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J. (1999). Evaluating the use of exploratory factor analysis in psychological research. *Psychological Methods, 4*(3), 272-299.

- Hair, J. F., Jr., Anderson, R. E., Tatham, R. L., & Black, W. C. (1995). *Multivariate data analysis with readings* (4th ed.). Prentice-Hall.

- Kaiser, H. F. (1970). A second generation little jiffy. *Psychometrika, 35*(4), 401-415.

- Kaiser, H. F., & Rice, J. (1974). Little Jiffy, Mark IV. *Educational and Psychological Measurement, 34*, 111-117.

---

*This research report provides comprehensive guidance for implementing EFA in Python for psychological research. The recommendations balance statistical rigor with practical implementation considerations.*