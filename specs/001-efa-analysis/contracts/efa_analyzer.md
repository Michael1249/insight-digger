# EFA Analyzer API Contract

**Module**: `src.efa_analyzer`  
**Version**: 1.0.0  
**Created**: November 24, 2025

## Class: EFAAnalyzer

### Constructor

```python
def __init__(
    self,
    n_factors: Optional[int] = None,
    extraction_method: str = 'principal',
    rotation_method: str = 'oblimin',
    convergence_tolerance: float = 1e-4,
    max_iterations: int = 100
) -> None
```

**Parameters**:
- `n_factors`: Number of factors to extract. If None, determined automatically.
- `extraction_method`: Factor extraction method ('principal', 'ml', 'minres')
- `rotation_method`: Factor rotation method ('oblimin', 'varimax', 'quartimax')
- `convergence_tolerance`: Convergence criterion for iterative algorithms
- `max_iterations`: Maximum iterations for convergence

**Raises**:
- `ValueError`: Invalid method names or parameter ranges
- `TypeError`: Invalid parameter types

---

### Method: validate_data

```python
def validate_data(self, data: pd.DataFrame) -> ValidationResults
```

**Purpose**: Validates data suitability for factor analysis

**Parameters**:
- `data`: Survey responses matrix (statements × respondents)

**Returns**:
```python
@dataclass
class ValidationResults:
    kmo_overall: float
    kmo_individual: Dict[str, float]
    bartlett_statistic: float
    bartlett_p_value: float
    is_suitable: bool
    warnings: List[str]
    assumptions_check: Dict[str, bool]
    sample_ratio: float
```

**Raises**:
- `ValueError`: Data is not numeric or has invalid dimensions
- `RuntimeError`: Cannot compute correlation matrix (singular data)

**Side Effects**: None (pure validation function)

---

### Method: fit

```python
def fit(self, data: pd.DataFrame, validate: bool = True) -> 'EFAAnalyzer'
```

**Purpose**: Performs exploratory factor analysis on the provided data

**Parameters**:
- `data`: Survey responses matrix (statements × respondents)
- `validate`: Whether to run pre-analysis validation

**Returns**: Self (for method chaining)

**Raises**:
- `ValueError`: Data validation fails and validate=True
- `RuntimeError`: Factor analysis fails to converge
- `LinAlgError`: Correlation matrix is singular or ill-conditioned

**Side Effects**: 
- Sets `self.factor_solution_`
- Sets `self.validation_results_`
- Sets `self.fit_completed = True`

**Preconditions**:
- `data` must be numeric DataFrame
- If `validate=True`, must pass validation checks

**Postconditions**:
- `self.factor_solution_` contains complete factor analysis results
- `self.fit_completed == True`

---

### Method: determine_factors

```python
def determine_factors(self, data: pd.DataFrame, max_factors: Optional[int] = None) -> int
```

**Purpose**: Determines optimal number of factors using multiple criteria

**Parameters**:
- `data`: Survey responses matrix
- `max_factors`: Maximum factors to consider (default: min(n_vars/3, n_obs-1))

**Returns**: Recommended number of factors (int)

**Raises**:
- `ValueError`: Invalid data or max_factors parameter
- `RuntimeError`: Cannot determine factors (insufficient data)

**Algorithm**: 
1. Eigenvalue > 1.0 criterion (Kaiser)
2. Scree plot visual inspection (automated elbow detection)
3. Parallel analysis (Monte Carlo simulation)
4. Return consensus recommendation

---

### Method: compute_factor_scores

```python
def compute_factor_scores(
    self, 
    data: Optional[pd.DataFrame] = None,
    method: str = 'regression'
) -> pd.DataFrame
```

**Purpose**: Computes factor scores for individual observations

**Parameters**:
- `data`: Data for score computation (if different from training data)
- `method`: Scoring method ('regression', 'bartlett', 'anderson')

**Returns**: DataFrame with factor scores (respondents × factors)

**Raises**:
- `RuntimeError`: Called before fit() or fit failed
- `ValueError`: Invalid scoring method
- `DimensionError`: Data dimensions incompatible with fitted model

**Preconditions**: 
- `self.fit_completed == True`
- If data provided, must have same variables as training data

---

### Method: get_reliability_stats

```python
def get_reliability_stats(self) -> Dict[str, float]
```

**Purpose**: Computes reliability statistics for extracted factors

**Returns**:
```python
{
    'factor_1_cronbach_alpha': float,
    'factor_2_cronbach_alpha': float,
    # ... for each factor
    'overall_kmo': float,
    'total_variance_explained': float
}
```

**Raises**:
- `RuntimeError`: Called before successful fit()

**Notes**: Cronbach's alpha computed using variables with |loading| ≥ 0.40

---

### Method: interpret_factors

```python
def interpret_factors(
    self, 
    statement_labels: Optional[List[str]] = None,
    loading_threshold: float = 0.40
) -> Dict[str, Any]
```

**Purpose**: Provides factor interpretation assistance

**Parameters**:
- `statement_labels`: Human-readable labels for statements
- `loading_threshold`: Minimum loading for interpretation

**Returns**:
```python
{
    'factor_1': {
        'primary_loadings': List[Tuple[str, float]],  # (statement, loading)
        'interpretation_suggestions': List[str],
        'reliability': float
    },
    # ... for each factor
    'overall': {
        'simple_structure_index': float,
        'total_variance_explained': float,
        'recommended_labels': List[str]
    }
}
```

**Raises**:
- `RuntimeError`: Called before successful fit()
- `ValueError`: Invalid loading_threshold

---

### Properties

```python
@property
def factor_loadings_(self) -> pd.DataFrame
    """Factor loadings matrix (variables × factors)"""

@property  
def eigenvalues_(self) -> np.ndarray
    """Eigenvalues of correlation matrix"""

@property
def communalities_(self) -> pd.Series
    """Communalities for each variable"""

@property
def rotation_matrix_(self) -> np.ndarray
    """Rotation matrix applied to factors"""

@property
def factor_correlations_(self) -> Optional[pd.DataFrame]
    """Factor correlations (None if orthogonal rotation)"""
```

**Availability**: Only after successful fit()

---

## Usage Examples

### Basic Usage

```python
from src.efa_analyzer import EFAAnalyzer
from src.soc_opros_loader import SocOprosLoader

# Load data
loader = SocOprosLoader()
data = loader.get_responses_matrix()

# Run factor analysis
efa = EFAAnalyzer(n_factors=None, rotation_method='oblimin')
validation = efa.validate_data(data)

if validation.is_suitable:
    efa.fit(data)
    print(f"Extracted {efa.factor_loadings_.shape[1]} factors")
    print(f"Total variance explained: {efa.get_reliability_stats()['total_variance_explained']:.2%}")
else:
    print(f"Data may not be suitable: {validation.warnings}")
```

### Advanced Usage

```python
# Custom factor determination
optimal_factors = efa.determine_factors(data, max_factors=8)
print(f"Recommended factors: {optimal_factors}")

# Refit with specific number of factors
efa_final = EFAAnalyzer(n_factors=optimal_factors, rotation_method='oblimin')
efa_final.fit(data)

# Compute factor scores
scores = efa_final.compute_factor_scores(data)

# Get interpretation assistance
interpretation = efa_final.interpret_factors(
    statement_labels=loader.get_statements(),
    loading_threshold=0.50
)

# Display results
for factor, details in interpretation.items():
    if factor != 'overall':
        print(f"\n{factor}:")
        print(f"  Reliability: {details['reliability']:.3f}")
        print(f"  Key indicators: {[item[0] for item in details['primary_loadings'][:3]]}")
```

## Error Handling Contract

### Exception Hierarchy

```python
class EFAError(Exception):
    """Base exception for EFA-related errors"""
    pass

class ValidationError(EFAError):
    """Data validation failed"""
    pass

class ConvergenceError(EFAError):
    """Factor analysis failed to converge"""
    pass

class InsufficientDataError(EFAError):
    """Insufficient data for reliable analysis"""
    pass
```

### Error Response Format

```python
@dataclass
class ErrorInfo:
    error_type: str
    message: str
    suggestions: List[str]
    recoverable: bool
    context: Dict[str, Any]
```

## Performance Guarantees

### Time Complexity
- `validate_data()`: O(n_vars² × n_obs) 
- `fit()`: O(n_vars³ + n_factors × iterations)
- `compute_factor_scores()`: O(n_obs × n_vars × n_factors)

### Memory Usage
- Peak memory: ~3x input data size
- Persistent storage: Factor solution size ≈ n_vars × n_factors

### Scalability Targets
- **Comfortable**: 500 variables × 100 observations
- **Maximum**: 1000 variables × 200 observations
- **Timeout**: 30 seconds for analysis completion

## Backward Compatibility

### Version Strategy
- **Major version** (X.0.0): Breaking API changes
- **Minor version** (1.X.0): New features, backward compatible
- **Patch version** (1.0.X): Bug fixes, no API changes

### Deprecated Features
None in version 1.0.0 (initial release)

### Migration Path
For future versions, deprecated features will be supported for at least 2 minor versions with warnings.