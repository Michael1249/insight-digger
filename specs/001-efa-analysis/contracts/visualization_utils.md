# Visualization Utils API Contract

**Module**: `src.visualization_utils`  
**Version**: 1.0.0  
**Created**: November 24, 2025

## Class: EFAVisualizer

### Purpose
Provides publication-quality visualizations for exploratory factor analysis results with customizable styling and export options.

### Constructor

```python
def __init__(
    self,
    style: str = 'seaborn-v0_8',
    publication_ready: bool = False,
    color_palette: str = 'viridis',
    figure_size: Tuple[float, float] = (10, 8),
    dpi: int = 100
) -> None
```

**Parameters**:
- `style`: Matplotlib style sheet ('seaborn-v0_8', 'ggplot', 'bmh')
- `publication_ready`: Use publication-quality settings (300 DPI, vector fonts)
- `color_palette`: Color scheme ('viridis', 'plasma', 'RdBu', 'Set1')
- `figure_size`: Default figure size in inches (width, height)
- `dpi`: Dots per inch for raster outputs

---

### Method: plot_scree

```python
def plot_scree(
    self,
    eigenvalues: np.ndarray,
    n_factors: Optional[int] = None,
    show_kaiser_line: bool = True,
    title: str = "Scree Plot",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Creates scree plot for factor number determination

**Parameters**:
- `eigenvalues`: Eigenvalues from correlation matrix decomposition
- `n_factors`: Number of factors to highlight (if extracted)
- `show_kaiser_line`: Show eigenvalue = 1.0 reference line
- `title`: Plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object

**Visual Elements**:
- Line plot of eigenvalues by factor number
- Kaiser criterion line (eigenvalue = 1.0)
- Highlighted point for extracted factors
- Grid for easy reading
- Proper axis labels and legend

---

### Method: plot_loadings_heatmap

```python
def plot_loadings_heatmap(
    self,
    loadings: pd.DataFrame,
    loading_threshold: float = 0.40,
    cluster_variables: bool = True,
    title: str = "Factor Loadings Matrix",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Visualizes factor loadings as annotated heatmap

**Parameters**:
- `loadings`: Factor loadings matrix (variables × factors)
- `loading_threshold`: Threshold for highlighting significant loadings
- `cluster_variables`: Cluster variables by factor structure
- `title`: Plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object

**Visual Features**:
- Color-coded loadings with diverging colormap
- Bold/highlighted significant loadings (≥threshold)
- Optional hierarchical clustering of variables
- Proper factor and variable labels
- Colorbar with loading magnitude scale

---

### Method: plot_biplot

```python
def plot_biplot(
    self,
    factor_scores: pd.DataFrame,
    loadings: pd.DataFrame,
    factor_x: int = 0,
    factor_y: int = 1,
    loading_threshold: float = 0.40,
    point_labels: Optional[List[str]] = None,
    title: str = "Factor Biplot",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Creates biplot showing cases and variables in factor space

**Parameters**:
- `factor_scores`: Individual factor scores (observations × factors)
- `loadings`: Factor loadings matrix
- `factor_x`: Factor index for x-axis (0-based)
- `factor_y`: Factor index for y-axis (0-based)
- `loading_threshold`: Minimum loading to display variable vectors
- `point_labels`: Labels for observations (optional)
- `title`: Plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object

**Visual Elements**:
- Scatter plot of observations in factor space
- Variable loading vectors as arrows
- Axis labels with variance explained percentages
- Optional observation labels
- Legend distinguishing observations and variables

---

### Method: plot_factor_scores_distribution

```python
def plot_factor_scores_distribution(
    self,
    factor_scores: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    plot_type: str = 'violin',
    title: str = "Factor Score Distributions",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Visualizes distribution of factor scores

**Parameters**:
- `factor_scores`: Individual factor scores matrix
- `factor_names`: Custom factor names for labeling
- `plot_type`: Distribution plot type ('violin', 'box', 'histogram', 'kde')
- `title`: Plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object

**Plot Types**:
- **violin**: Violin plots showing full distribution shape
- **box**: Box plots with quartiles and outliers
- **histogram**: Histograms with density curves
- **kde**: Kernel density estimation plots

---

### Method: plot_parallel_analysis

```python
def plot_parallel_analysis(
    self,
    observed_eigenvalues: np.ndarray,
    random_eigenvalues: np.ndarray,
    confidence_intervals: Optional[np.ndarray] = None,
    title: str = "Parallel Analysis",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Creates parallel analysis plot for factor number determination

**Parameters**:
- `observed_eigenvalues`: Actual eigenvalues from data
- `random_eigenvalues`: Mean eigenvalues from random data
- `confidence_intervals`: 95% confidence intervals for random eigenvalues
- `title`: Plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object

**Visual Elements**:
- Line plot comparing observed vs random eigenvalues
- Confidence intervals (if provided)
- Intersection point highlighting suggested factors
- Clear legend and axis labels

---

### Method: create_factor_summary_plot

```python
def create_factor_summary_plot(
    self,
    loadings: pd.DataFrame,
    eigenvalues: np.ndarray,
    variance_explained: np.ndarray,
    reliability_stats: Dict[str, float],
    title: str = "Factor Analysis Summary",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Comprehensive multi-panel summary visualization

**Parameters**:
- `loadings`: Factor loadings matrix
- `eigenvalues`: Factor eigenvalues
- `variance_explained`: Variance explained by each factor
- `reliability_stats`: Cronbach's alpha for each factor
- `title`: Overall plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object with subplots

**Panel Layout** (2×2 grid):
1. **Loadings Heatmap**: Primary factor structure
2. **Scree Plot**: Factor number validation
3. **Variance Explained**: Bar chart of factor contributions
4. **Reliability Plot**: Bar chart of Cronbach's alpha values

---

### Method: plot_factor_correlations

```python
def plot_factor_correlations(
    self,
    factor_correlations: pd.DataFrame,
    title: str = "Factor Correlations",
    save_path: Optional[str] = None
) -> plt.Figure
```

**Purpose**: Visualizes correlations between extracted factors

**Parameters**:
- `factor_correlations`: Factor correlation matrix (factors × factors)
- `title`: Plot title
- `save_path`: Path to save plot (optional)

**Returns**: matplotlib Figure object

**Note**: Only applicable for oblique rotations (factors can correlate)

---

## Utility Functions

### Function: setup_publication_style

```python
def setup_publication_style() -> None
```

**Purpose**: Configures matplotlib for publication-quality output

**Effects**:
- Sets DPI to 300
- Configures vector fonts
- Sets appropriate font sizes
- Enables tight layout

---

### Function: export_plot

```python
def export_plot(
    figure: plt.Figure,
    path: str,
    format: str = 'png',
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None
```

**Purpose**: Exports plot with specified quality settings

**Parameters**:
- `figure`: matplotlib Figure to export
- `path`: Output file path (without extension)
- `format`: File format ('png', 'pdf', 'svg', 'eps')
- `dpi`: Resolution for raster formats
- `bbox_inches`: Bounding box setting

---

### Function: create_colormap

```python
def create_colormap(
    name: str,
    n_colors: int,
    categorical: bool = False
) -> Union[ListedColormap, str]
```

**Purpose**: Creates appropriate colormap for factor analysis plots

**Parameters**:
- `name`: Colormap name ('diverging', 'sequential', 'categorical')
- `n_colors`: Number of colors needed
- `categorical`: Whether colors represent distinct categories

**Returns**: Colormap object or name string

---

## Usage Examples

### Basic Visualization Workflow

```python
from src.visualization_utils import EFAVisualizer
from src.efa_analyzer import EFAAnalyzer

# Initialize visualizer
viz = EFAVisualizer(publication_ready=True, color_palette='RdBu')

# After running factor analysis
efa = EFAAnalyzer()
efa.fit(data)

# Create individual plots
scree_fig = viz.plot_scree(
    eigenvalues=efa.eigenvalues_,
    n_factors=efa.factor_loadings_.shape[1],
    save_path="output/scree_plot"
)

loadings_fig = viz.plot_loadings_heatmap(
    loadings=efa.factor_loadings_,
    cluster_variables=True,
    save_path="output/loadings_heatmap"
)

# Compute factor scores
scores = efa.compute_factor_scores(data)
biplot_fig = viz.plot_biplot(
    factor_scores=scores,
    loadings=efa.factor_loadings_,
    save_path="output/biplot"
)
```

### Comprehensive Summary Visualization

```python
# Create multi-panel summary
reliability = efa.get_reliability_stats()
variance_explained = efa.eigenvalues_ / efa.eigenvalues_.sum()

summary_fig = viz.create_factor_summary_plot(
    loadings=efa.factor_loadings_,
    eigenvalues=efa.eigenvalues_,
    variance_explained=variance_explained,
    reliability_stats=reliability,
    save_path="output/factor_summary"
)

# Display in Jupyter notebook
summary_fig.show()
```

### Custom Styling

```python
# Custom visualizer for specific aesthetic
custom_viz = EFAVisualizer(
    style='ggplot',
    publication_ready=True,
    color_palette='Set1',
    figure_size=(12, 10),
    dpi=300
)

# Apply custom styling to all plots
custom_viz.setup_publication_style()
```

### Parallel Analysis Visualization

```python
# After running parallel analysis
from factor_analyzer import calculate_bartlett_sphericity
import numpy as np

# Generate random eigenvalues for comparison
n_vars = data.shape[1]
random_eigenvals = np.random.random((100, n_vars))  # 100 random datasets
random_mean = random_eigenvals.mean(axis=0)
random_ci = np.percentile(random_eigenvals, [2.5, 97.5], axis=0)

# Plot parallel analysis
parallel_fig = viz.plot_parallel_analysis(
    observed_eigenvalues=efa.eigenvalues_,
    random_eigenvalues=random_mean,
    confidence_intervals=random_ci,
    save_path="output/parallel_analysis"
)
```

## Configuration Options

### Style Presets

```python
STYLE_PRESETS = {
    'academic': {
        'style': 'seaborn-v0_8-whitegrid',
        'color_palette': 'Set1',
        'figure_size': (8, 6),
        'publication_ready': True
    },
    'presentation': {
        'style': 'seaborn-v0_8-darkgrid',
        'color_palette': 'viridis',
        'figure_size': (12, 8),
        'publication_ready': False
    },
    'print': {
        'style': 'classic',
        'color_palette': 'RdBu',
        'figure_size': (10, 8),
        'publication_ready': True
    }
}
```

### Export Formats

```python
EXPORT_FORMATS = {
    'png': {'dpi': 300, 'transparent': False},
    'pdf': {'dpi': 300, 'bbox_inches': 'tight'},
    'svg': {'bbox_inches': 'tight', 'transparent': True},
    'eps': {'dpi': 300, 'bbox_inches': 'tight'},
    'jpg': {'dpi': 300, 'quality': 95, 'optimize': True}
}
```

## Error Handling

### Custom Exceptions

```python
class VisualizationError(Exception):
    """Base visualization exception"""
    pass

class InvalidDataError(VisualizationError):
    """Data format invalid for visualization"""
    pass

class StyleError(VisualizationError):
    """Style configuration error"""
    pass
```

### Graceful Degradation

- **Missing data**: Skip missing observations with warning
- **Invalid factor indices**: Default to first two factors
- **Export failures**: Fallback to display-only mode
- **Style errors**: Revert to default matplotlib style

## Performance Considerations

### Optimization Strategies
- **Large datasets**: Automatic downsampling for scatter plots (>1000 points)
- **Memory management**: Clear figures after export to prevent accumulation
- **Computation caching**: Cache expensive calculations (clustering, correlations)

### Resource Limits
- **Maximum points in biplot**: 10,000 (with automatic sampling)
- **Maximum variables in heatmap**: 500 (with warning)
- **Memory usage**: <500MB for typical visualizations