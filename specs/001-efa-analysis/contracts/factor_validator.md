# Factor Validator API Contract

**Module**: `src.factor_validator`  
**Version**: 1.0.0  
**Created**: November 24, 2025

## Class: FactorValidator

### Purpose
Provides comprehensive statistical validation for factor analysis prerequisites and results quality assessment.

### Constructor

```python
def __init__(
    self,
    kmo_threshold: float = 0.6,
    bartlett_alpha: float = 0.05,
    min_sample_ratio: float = 5.0,
    loading_threshold: float = 0.40
) -> None
```

**Parameters**:
- `kmo_threshold`: Minimum acceptable KMO value for analysis
- `bartlett_alpha`: Significance level for Bartlett's sphericity test
- `min_sample_ratio`: Minimum observations-to-variables ratio
- `loading_threshold`: Minimum factor loading for practical significance

---

### Method: check_data_adequacy

```python
def check_data_adequacy(self, data: pd.DataFrame) -> DataAdequacyResult
```

**Purpose**: Comprehensive data quality assessment for factor analysis

**Parameters**:
- `data`: Survey responses matrix (variables × observations)

**Returns**:
```python
@dataclass
class DataAdequacyResult:
    is_adequate: bool
    sample_size_ratio: float
    missing_data_percent: float
    variable_variance_check: Dict[str, bool]
    correlation_determinant: float
    issues: List[str]
    recommendations: List[str]
```

**Validation Checks**:
1. **Sample Size**: n_observations ≥ min_sample_ratio × n_variables
2. **Missing Data**: < 20% missing values per variable
3. **Variance**: All variables have variance > 0
4. **Correlation Matrix**: Determinant > 1e-8 (not singular)
5. **Multicollinearity**: No perfect correlations (r = 1.0)

---

### Method: test_factorability

```python
def test_factorability(self, data: pd.DataFrame) -> FactorabilityResult
```

**Purpose**: Statistical tests for factor analysis appropriateness

**Parameters**:
- `data`: Survey responses matrix

**Returns**:
```python
@dataclass  
class FactorabilityResult:
    kmo_overall: float
    kmo_variables: Dict[str, float]
    kmo_interpretation: str
    bartlett_chi_square: float
    bartlett_p_value: float
    bartlett_df: int
    is_factorable: bool
    warnings: List[str]
```

**Statistical Tests**:
1. **Kaiser-Meyer-Olkin (KMO)**:
   - Overall KMO measure
   - Individual variable KMO values
   - Interpretation guidelines
2. **Bartlett's Test of Sphericity**:
   - Chi-square statistic
   - p-value and degrees of freedom
   - Null hypothesis testing

**Interpretation Scale**:
- KMO ≥ 0.9: Marvelous
- KMO ≥ 0.8: Meritorious  
- KMO ≥ 0.7: Middling
- KMO ≥ 0.6: Mediocre (acceptable)
- KMO < 0.6: Miserable (inadequate)

---

### Method: validate_factor_solution

```python
def validate_factor_solution(
    self,
    loadings: pd.DataFrame,
    eigenvalues: np.ndarray,
    communalities: pd.Series,
    data: pd.DataFrame
) -> SolutionQualityResult
```

**Purpose**: Validates quality of extracted factor solution

**Parameters**:
- `loadings`: Factor loadings matrix
- `eigenvalues`: Eigenvalues of factors
- `communalities`: Variable communalities
- `data`: Original data for reliability computation

**Returns**:
```python
@dataclass
class SolutionQualityResult:
    simple_structure_index: float
    cross_loadings_count: int
    low_communalities: List[str]
    factor_determinacy: Dict[str, float]
    reliability_stats: Dict[str, float]
    variance_explained: float
    heywood_cases: List[str]
    quality_rating: str
```

**Quality Checks**:
1. **Simple Structure**: Percentage of variables with single high loading
2. **Cross-loadings**: Variables loading >0.40 on multiple factors
3. **Communalities**: Variables with h² < 0.20 (poorly explained)
4. **Heywood Cases**: Communalities > 1.0 (estimation problems)
5. **Factor Determinacy**: Correlation between factor scores and true factors
6. **Reliability**: Cronbach's alpha for each factor

---

### Method: check_assumptions

```python
def check_assumptions(self, data: pd.DataFrame) -> AssumptionTestResult
```

**Purpose**: Tests statistical assumptions for factor analysis

**Parameters**:
- `data`: Survey responses matrix

**Returns**:
```python
@dataclass
class AssumptionTestResult:
    linearity_assessment: Dict[str, float]
    normality_tests: Dict[str, Dict[str, float]]
    outlier_detection: Dict[str, List[int]]
    assumption_violations: List[str]
    severity_levels: Dict[str, str]
    robustness_notes: List[str]
```

**Assumption Tests**:
1. **Linearity**: Correlation linearity assessment
2. **Normality**: Shapiro-Wilk tests for multivariate normality
3. **Outliers**: Mahalanobis distance outlier detection
4. **Homoscedasticity**: Variance homogeneity assessment

**Severity Levels**: 'MINOR', 'MODERATE', 'SEVERE', 'CRITICAL'

---

### Method: recommend_improvements

```python
def recommend_improvements(
    self,
    validation_results: Union[DataAdequacyResult, FactorabilityResult, SolutionQualityResult]
) -> List[str]
```

**Purpose**: Provides actionable recommendations for improving factor analysis

**Parameters**:
- `validation_results`: Any validation result object

**Returns**: List of specific improvement recommendations

**Recommendation Categories**:
1. **Data Collection**: Sample size, response quality
2. **Variable Selection**: Remove problematic variables
3. **Method Adjustment**: Alternative extraction/rotation methods
4. **Interpretation**: Cautionary notes for marginal results

---

## Utility Functions

### Function: interpret_kmo

```python
def interpret_kmo(kmo_value: float) -> str
```

**Purpose**: Converts KMO value to interpretive text

**Returns**: Interpretation string ('Marvelous', 'Meritorious', etc.)

---

### Function: format_validation_report

```python
def format_validation_report(
    adequacy: DataAdequacyResult,
    factorability: FactorabilityResult,
    assumptions: AssumptionTestResult,
    output_format: str = 'markdown'
) -> str
```

**Purpose**: Generates comprehensive validation report

**Parameters**:
- `adequacy`: Data adequacy results
- `factorability`: Factorability test results
- `assumptions`: Assumption test results
- `output_format`: Report format ('markdown', 'html', 'text')

**Returns**: Formatted validation report

---

## Usage Examples

### Basic Validation Workflow

```python
from src.factor_validator import FactorValidator
from src.soc_opros_loader import SocOprosLoader

# Initialize validator
validator = FactorValidator(kmo_threshold=0.6, min_sample_ratio=3.0)

# Load and validate data
loader = SocOprosLoader()
data = loader.get_responses_matrix()

# Step 1: Check data adequacy
adequacy = validator.check_data_adequacy(data)
print(f"Data adequate: {adequacy.is_adequate}")
print(f"Sample ratio: {adequacy.sample_size_ratio:.1f}:1")

# Step 2: Test factorability
factorability = validator.test_factorability(data)
print(f"KMO Overall: {factorability.kmo_overall:.3f} ({factorability.kmo_interpretation})")
print(f"Bartlett's p-value: {factorability.bartlett_p_value:.3e}")

# Step 3: Check assumptions
assumptions = validator.check_assumptions(data)
if assumptions.assumption_violations:
    print(f"Assumption violations: {assumptions.assumption_violations}")

# Generate comprehensive report
report = validator.format_validation_report(adequacy, factorability, assumptions)
print(report)
```

### Post-Analysis Validation

```python
# After running factor analysis
from src.efa_analyzer import EFAAnalyzer

efa = EFAAnalyzer()
efa.fit(data)

# Validate solution quality
solution_quality = validator.validate_factor_solution(
    loadings=efa.factor_loadings_,
    eigenvalues=efa.eigenvalues_,
    communalities=efa.communalities_,
    data=data
)

print(f"Simple structure index: {solution_quality.simple_structure_index:.3f}")
print(f"Factors with α ≥ 0.7: {sum(1 for α in solution_quality.reliability_stats.values() if α >= 0.7)}")

# Get improvement recommendations
recommendations = validator.recommend_improvements(solution_quality)
for rec in recommendations:
    print(f"• {rec}")
```

### Custom Validation Thresholds

```python
# Strict validation for publication
strict_validator = FactorValidator(
    kmo_threshold=0.8,
    bartlett_alpha=0.01,
    min_sample_ratio=10.0,
    loading_threshold=0.5
)

# Lenient validation for exploratory analysis  
lenient_validator = FactorValidator(
    kmo_threshold=0.5,
    bartlett_alpha=0.1,
    min_sample_ratio=3.0,
    loading_threshold=0.3
)
```

## Error Handling

### Custom Exceptions

```python
class ValidationError(Exception):
    """Base validation exception"""
    pass

class InsufficientDataError(ValidationError):
    """Data insufficient for reliable validation"""
    pass

class StatisticalTestError(ValidationError):
    """Statistical test computation failed"""
    pass
```

### Graceful Degradation

When statistical tests fail due to edge cases:
1. **Singular correlation matrix**: Report determinant and suggest variable removal
2. **Perfect correlations**: Identify and flag problematic variable pairs
3. **No variance**: List constant variables for removal
4. **Extreme outliers**: Provide outlier indices and impact assessment

## Performance Characteristics

### Computational Complexity
- `check_data_adequacy()`: O(n_vars × n_obs)
- `test_factorability()`: O(n_vars² × n_obs + n_vars³)
- `validate_factor_solution()`: O(n_vars × n_factors + reliability_computation)
- `check_assumptions()`: O(n_vars² × n_obs)

### Memory Requirements
- Peak memory: ~2x correlation matrix size
- Temporary storage: O(n_vars²) for statistical computations

### Performance Targets
- Validation completion: <5 seconds for 500 variables
- Memory usage: <1GB for typical datasets
- Graceful handling of edge cases without crashes