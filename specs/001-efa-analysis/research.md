# Research: Exploratory Factor Analysis (EFA) Best Practices

**Feature**: `001-efa-analysis`  
**Research Date**: November 24, 2025  
**Phase**: 0 (Outline & Research)

## Research Summary

This document consolidates findings on EFA implementation best practices for psychological research using Python. Research focused on technical implementation patterns, statistical validation standards, and integration approaches for survey data analysis.

## 1. Technical Implementation Decisions

### Python Libraries Comparison

**Decision**: Use `factor_analyzer` as primary library
**Rationale**: 
- Purpose-built for EFA/CFA with psychological research focus
- Comprehensive statistical outputs (KMO, Bartlett's, communalities)
- Established validation in academic research
- Better handling of rotation methods compared to scikit-learn

**Alternatives considered**:
- `scikit-learn` PCA/TruncatedSVD: Limited EFA support, no psychological validation metrics
- `statsmodels`: Good statistical foundation but requires more manual implementation
- `pingouin`: Good but less established, fewer advanced features

### Factor Extraction Method

**Decision**: Principal Axis Factoring (PAF) as default
**Rationale**:
- Standard method for psychological constructs discovery
- Handles common variance extraction (vs PCA which includes unique variance)
- Robust to violations of multivariate normality
- Established in psychological research literature

**Implementation Pattern**:
```python
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(n_factors=None, rotation='oblimin', method='principal')
```

### Rotation Method Hierarchy

**Decision**: Oblimin (oblique) as default, Varimax as option
**Rationale**:
- Psychological factors typically correlate in real-world data
- Oblique rotation allows for more realistic factor relationships
- Varimax available for theoretical constraints requiring orthogonal factors

## 2. Statistical Validation Standards

### KMO (Kaiser-Meyer-Olkin) Interpretation
- **≥ 0.90**: Marvelous factorability
- **≥ 0.80**: Meritorious factorability  
- **≥ 0.70**: Middling factorability
- **≥ 0.60**: Mediocre but acceptable
- **< 0.60**: Miserable, inadequate for factor analysis

### Bartlett's Test of Sphericity
- **Requirement**: p < 0.05 (typically p < 0.001)
- **Interpretation**: Correlation matrix significantly different from identity matrix
- **Action if failed**: Warning with option to proceed (educational contexts)

### Factor Loading Significance
- **|0.30|**: Minimal level for interpretation
- **|0.40|**: More important (recommended threshold)
- **|0.50|**: Practically significant
- **|0.70|**: Well-defined structure

### Sample Size Guidelines
- **Minimum**: 5:1 participant-to-variable ratio
- **Preferred**: 10:1 ratio (200+ cases for most surveys)
- **Current data**: 15 respondents × 265 variables = 0.06:1 (INADEQUATE)
- **Recommendation**: Transpose data matrix (265 cases × 15 variables) for proper analysis

## 3. Factor Interpretation Framework

### Optimal Factor Number Determination

**Multi-criteria approach**:
1. **Eigenvalue > 1.0** (Kaiser criterion)
2. **Scree plot** visual inspection
3. **Parallel analysis** (Monte Carlo simulation)
4. **Theoretical interpretability**

**Implementation Pattern**:
```python
# Parallel analysis for factor number
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import (FactorAnalyzer, calculate_bartlett_sphericity)

# Scree plot analysis
eigenvals, _ = calculate_bartlett_sphericity(data)
plt.plot(range(1, len(eigenvals)+1), eigenvals)
plt.axhline(y=1, color='r', linestyle='--')
```

### Simple Structure Assessment
- **Target**: Each variable loads primarily on one factor
- **Cross-loadings**: Minimize |0.40|+ loadings on multiple factors  
- **Factor purity**: At least 3 variables per factor with significant loadings

### Reliability Standards
- **Cronbach's α ≥ 0.70**: Acceptable internal consistency
- **Cronbach's α ≥ 0.80**: Good internal consistency
- **Cronbach's α ≥ 0.90**: Excellent internal consistency

## 4. Visualization Standards

### Required Plots
1. **Scree Plot**: Factor number determination
2. **Loading Heatmap**: Factor structure visualization
3. **Biplot**: Cases and variables in factor space
4. **Parallel Analysis Plot**: Statistical factor number validation

### Publication Standards
- **Resolution**: 300+ DPI for print quality
- **Format**: Vector formats (SVG, PDF) preferred
- **Color schemes**: Colorblind-friendly palettes
- **Annotation**: Clear factor labels and loading values

**Implementation Pattern**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality heatmap
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
            square=True, cbar_kws={'shrink': 0.8})
```

## 5. Survey Data Integration Patterns

### Likert Scale Data Preprocessing
**Decision**: Treat as interval data with Pearson correlations
**Rationale**:
- Standard practice in psychological research
- Factor analysis assumes linear relationships
- Robust with 5+ response categories
- Enables comparison with published research

### Missing Data Handling Strategy
**Decision**: Pairwise deletion for correlation matrix, listwise for factor scores
**Rationale**:
- Maximizes available data for correlation estimation
- Maintains case-wise integrity for factor score computation
- Standard approach in psychological research software

**Implementation Pattern**:
```python
# Pairwise correlation matrix
corr_matrix = data.corr(method='pearson', min_periods=1)

# Factor analysis with listwise deletion for scores
fa.fit(data.dropna())
scores = fa.transform(data.dropna())
```

### Data Orientation Handling
**Critical Issue**: Current data (265 statements × 15 respondents) violates sample size requirements
**Solution**: Transpose for analysis (265 cases × 15 variables)
**Implementation**: 
```python
# Correct orientation for factor analysis
if n_variables > n_cases:
    data_transposed = data.T
    analysis_data = data_transposed
```

## 6. Quality Assurance Framework

### Pre-Analysis Validation Checklist
- [ ] Sample size adequate (≥ 5:1 ratio)
- [ ] Data orientation correct (cases > variables)
- [ ] Missing data pattern assessed
- [ ] Correlation matrix non-singular
- [ ] KMO ≥ 0.60
- [ ] Bartlett's test p < 0.05

### Post-Analysis Validation Checklist  
- [ ] Factor solution interpretable
- [ ] Simple structure achieved
- [ ] Communalities reasonable (0.20-0.90)
- [ ] Cronbach's α ≥ 0.70 for factors
- [ ] Total variance explained ≥ 60%

### Error Handling Patterns
```python
def validate_efa_assumptions(data):
    """Comprehensive EFA assumption validation."""
    results = {}
    
    # Sample size check
    n_cases, n_vars = data.shape
    results['sample_ratio'] = n_cases / n_vars
    results['adequate_sample'] = results['sample_ratio'] >= 5
    
    # KMO test
    kmo_all, kmo_model = calculate_kmo(data)
    results['kmo'] = kmo_model
    results['kmo_adequate'] = kmo_model >= 0.6
    
    # Bartlett's test
    chi_square, p_value = calculate_bartlett_sphericity(data)
    results['bartlett_p'] = p_value
    results['bartlett_significant'] = p_value < 0.05
    
    return results
```

## 7. Integration Workflow Template

### Phase 1: Data Preparation
```python
# Load data using existing soc_opros_loader
from soc_opros_loader import SocOprosLoader
loader = SocOprosLoader()
raw_data = loader.get_responses_matrix()

# Handle orientation and missing data
analysis_data = prepare_for_efa(raw_data)
```

### Phase 2: Statistical Validation
```python
# Run assumption tests
validation_results = validate_efa_assumptions(analysis_data)
if not validation_results['adequate_sample']:
    print(f"WARNING: Sample size ratio {validation_results['sample_ratio']:.2f} < 5.0")
```

### Phase 3: Factor Analysis
```python
# Determine optimal factors
optimal_factors = determine_factor_number(analysis_data)

# Run EFA
fa = FactorAnalyzer(n_factors=optimal_factors, rotation='oblimin')
fa.fit(analysis_data)
```

### Phase 4: Interpretation & Visualization  
```python
# Generate comprehensive output
generate_efa_report(fa, analysis_data, output_format='jupyter')
```

## Implementation Priorities

### Phase 0 Outcomes (This Document)
- ✅ Technical approach validated (`factor_analyzer` + PAF + oblimin)
- ✅ Statistical standards documented (KMO, Bartlett's, loading thresholds)
- ✅ Data orientation issue identified and solution proposed
- ✅ Quality assurance framework established

### Next Steps (Phase 1)
1. Design data model for factor analysis objects
2. Create API contracts for EFA functionality
3. Generate quickstart guide for researchers
4. Update agent context with new technical decisions

## References & Sources

- **Fabrigar et al. (1999)**: Evaluating the use of EFA in psychological research
- **Costello & Osborne (2005)**: Best practices in EFA for psychological research  
- **factor_analyzer documentation**: Technical implementation patterns
- **APA Style Guide**: Statistical reporting standards for factor analysis
- **Psychological Methods journals**: Contemporary best practices review

## Validation Status

**Research Completeness**: ✅ COMPLETE  
**Technical Decisions**: ✅ RESOLVED  
**Implementation Ready**: ✅ YES  
**Phase 1 Prerequisites**: ✅ SATISFIED