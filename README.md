# Insight Digger - Advanced EFA Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Comprehensive-brightgreen)](tests/)

A comprehensive Exploratory Factor Analysis (EFA) toolkit for discovering hidden psychological dimensions in survey data. Built with advanced statistical validation, interactive visualizations, and production-ready error handling.

## 🚀 Features

### Core EFA Capabilities
- **🔍 Advanced Factor Discovery**: Principal axis factoring with multiple rotation methods
- **📊 Statistical Validation**: KMO, Bartlett's test, sample adequacy assessment
- **🎯 Parallel Analysis**: Automated factor number determination with performance optimization
- **🔄 Rotation Comparison**: Compare varimax, oblimin, quartimax, promax rotations
- **📈 Factor Reliability**: Cronbach's alpha calculation with item analysis

### Data Integration & Loading
- **🔄 Dynamic Data Loading**: Automatically adapts to variable number of survey statements and respondents
- **🌐 Cyrillic Text Support**: Proper UTF-8 encoding for Cyrillic respondent names
- **📊 Clean Data Interface**: Structured responses matrix ready for statistical analysis
- **🔗 Google Sheets Integration**: Direct CSV export connection without authentication
- **🔗 Flexible Input**: CSV, Excel, JSON data formats supported

### Visualization & Analysis
- **📊 Interactive Plots**: Scree plots, loadings heatmaps, biplots with Plotly
- **📋 Comprehensive Reports**: Factor interpretation guidelines and loading tables  
- **🎨 Customizable Charts**: Enhanced visualizations with professional styling
- **📱 Export Options**: Save results as CSV, PDF, or interactive HTML

### Performance & Reliability  
- **⚡ Performance Optimized**: Parallel processing for large datasets and simulations
- **🛡️ Robust Error Handling**: Comprehensive validation and graceful degradation
- **🧪 Extensively Tested**: 150+ unit tests covering all functionality plus comprehensive integration testing
- **🔧 Memory Efficient**: Chunked processing for large correlation matrices

### Integration Features
- **🖥️ Command Line Interface**: Full CLI for automated analysis pipelines
- **📓 Jupyter Integration**: Interactive notebooks for exploratory analysis:
  - `01_soc_opros_demo.ipynb`: Data loading and initial exploration
  - `02_efa_basic.ipynb`: Step-by-step basic EFA analysis with validation  
  - `03_efa_advanced_visualization.ipynb`: Advanced visualizations and interpretation
- **⚡ Binder Ready**: Run online instantly with pre-configured environment
- **🌐 Cross-Platform**: Windows, macOS, Linux compatibility

## 🏃‍♂️ Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from src.efa_analyzer import EFAAnalyzer
import pandas as pd

# Load your survey data
data = pd.read_csv('your_survey_data.csv')

# Run EFA with automatic factor determination
efa = EFAAnalyzer(rotation='varimax')
solution = efa.fit(data)

# View factor loadings
print(solution.loadings)
print(f"Variance explained: {solution.variance_explained.sum():.1%}")
```

### Advanced Analysis
```python
from src.factor_validator import FactorValidator
from src.visualization_engine import VisualizationEngine

# Comprehensive data validation
validator = FactorValidator()
validation = validator.comprehensive_validation(data)

if validation.is_valid:
    # Run parallel analysis for optimal factors
    n_factors = efa.parallel_analysis(data, n_simulations=1000)
    
    # Perform EFA with optimal factors
    efa = EFAAnalyzer(n_factors=n_factors, rotation='oblimin')
    solution = efa.fit(data)
    
    # Create visualizations
    viz = VisualizationEngine()
    scree_plot = viz.plot_scree(efa.eigenvalues_)
    heatmap = viz.plot_factor_loadings_heatmap(solution.loadings)
```

## 📚 Documentation

### Core Components

#### EFAAnalyzer
```python
efa = EFAAnalyzer(
    n_factors=None,              # Auto-determine if None
    rotation='oblimin',          # 'varimax', 'oblimin', 'quartimax', 'promax'
    extraction_method='principal', # 'principal', 'ml', 'minres'
    max_iterations=1000          # Convergence limit
)
```

#### FactorValidator
```python
validator = FactorValidator()

# Check data suitability
validation = validator.comprehensive_validation(data)
kmo_result = validator.calculate_kmo(data)
bartlett = validator.calculate_bartlett_test(data)
adequacy = validator.check_enhanced_sample_adequacy(data)
```

#### VisualizationEngine
```python
viz = VisualizationEngine()

# Core visualizations
scree_plot = viz.plot_scree(eigenvalues)
loadings_heatmap = viz.plot_factor_loadings_heatmap(loadings)
loadings_bar = viz.plot_factor_loadings_bar(loadings)
biplot = viz.plot_biplot(factor_scores, loadings)

# Advanced visualizations  
parallel_plot = viz.plot_parallel_analysis(pa_results)
rotation_comparison = viz.plot_rotation_comparison(comparison_results)
```

## 🛠️ Command Line Interface

### Data Validation
```bash
python src/cli.py validate --input data.csv
```

### Basic Analysis
```bash
python src/cli.py analyze --input data.csv --factors 3 --rotation varimax --output results/
```

### Advanced Analysis with Parallel Analysis
```bash
python src/cli.py analyze --input data.csv --parallel-analysis --rotation oblimin --export-plots
```

### Full Pipeline
```bash
python src/cli.py pipeline --input data.csv --validate --parallel-analysis --compare-rotations --output results/
```

## 📊 Example Results

### Factor Loadings Matrix
```
Variables         Factor_1  Factor_2  Factor_3  Communalities
outgoing            0.82      0.15     -0.08        0.71
sociable            0.78      0.22      0.01        0.66
energetic           0.65     -0.31      0.12        0.54
anxious            -0.12      0.89      0.15        0.83
worried             0.05      0.76      0.22        0.63
creative            0.18      0.11      0.84        0.75
imaginative         0.08     -0.02      0.78        0.61
```

### Validation Results
```
KMO Measure: 0.847 (Good)
Bartlett's Test: χ² = 892.3, p < 0.001 (Significant)
Sample Adequacy: 200 observations, 8 variables (Excellent)
Suggested Factors (Parallel Analysis): 3
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/test_efa_analyzer.py tests/test_factor_validator.py -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_integration.py::TestPerformanceBenchmarks -v
```

## 📁 Project Structure

```
Insight Digger/
├── src/                          # Core source code
│   ├── efa_analyzer.py          # Main EFA analysis engine
│   ├── factor_validator.py      # Statistical validation tools
│   ├── visualization_engine.py  # Plotting and visualization
│   ├── cli.py                   # Command line interface
│   ├── output_formatter.py      # Multi-format output generation
│   └── memory_optimizer.py      # Memory management utilities
├── notebooks/                    # Interactive Jupyter notebooks
│   ├── 01_soc_opros_demo.ipynb  # Data loading and initial exploration
│   ├── 02_efa_basic.ipynb       # Basic EFA analysis walkthrough
│   └── 03_efa_advanced_visualization.ipynb  # Advanced visualizations
├── tests/                        # Comprehensive test suite
│   ├── test_efa_analyzer.py     # EFA unit tests (40+ tests)
│   ├── test_factor_validator.py # Validation unit tests (35+ tests)
│   └── test_integration.py      # Integration tests (20+ tests)
├── docs/                        # Documentation and examples
├── examples/                    # Example analyses and datasets
└── requirements.txt             # Dependencies and versions
```

## 🔧 Dependencies

### Core Requirements
- `pandas >= 1.3.0` - Data manipulation and analysis
- `numpy >= 1.21.0` - Numerical computing
- `scipy >= 1.7.0` - Statistical functions (optional but recommended)

### Optional Enhanced Features  
- `factor_analyzer >= 0.4.0` - Advanced EFA algorithms
- `matplotlib >= 3.4.0` - Basic plotting
- `seaborn >= 0.11.0` - Statistical visualizations
- `plotly >= 5.0.0` - Interactive visualizations
- `scikit-learn >= 1.0.0` - Additional statistical tools

### Development & Testing
- `pytest >= 6.0.0` - Testing framework
- `pytest-cov` - Coverage reporting

## 🎯 Use Cases

### Psychology Research
```python
# Personality factor analysis
personality_data = pd.read_csv('big_five_survey.csv')
efa = EFAAnalyzer(n_factors=5, rotation='varimax')
solution = efa.fit(personality_data)

# Interpret Big Five factors
interpretation = efa.get_factor_interpretation(solution)
```

### Market Research
```python
# Customer satisfaction dimensions
satisfaction_data = pd.read_csv('customer_survey.csv') 
validation = validator.comprehensive_validation(satisfaction_data)

if validation.is_valid:
    n_factors = efa.parallel_analysis(satisfaction_data)
    solution = efa.fit(satisfaction_data)
```

### Educational Assessment
```python
# Academic skill dimensions
academic_data = pd.read_csv('academic_assessment.csv')
solution = efa.fit(academic_data)

# Calculate factor reliability
for factor in solution.loadings.columns:
    reliability = validator.calculate_cronbach_alpha(
        solution.loadings[factor], academic_data
    )
    print(f"{factor}: α = {reliability.alpha_value:.3f}")
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/your-repo/insight-digger.git
cd insight-digger
pip install -r requirements-dev.txt
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📞 Support

- 📧 Issues: [GitHub Issues](https://github.com/your-repo/insight-digger/issues)
- 📖 Documentation: [Full Documentation](docs/)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/insight-digger/discussions)

## 🙏 Acknowledgments

Built with inspiration from:
- `factor_analyzer` library for core EFA algorithms
- `scikit-learn` for statistical computing patterns  
- `plotly` for interactive visualization capabilities
- The broader Python scientific computing community

## Data Source

The project analyzes the 'soc opros' survey containing:
- **265 philosophical/psychological statements** (variable content)
- **15 respondents** with Cyrillic names (Амелия, Итанио, Отец, etc.)
- **5-point Likert scale** responses (strongly disagree → strongly agree)
- **68.2% completion rate** with 2,723 valid responses

## Project Structure

```
insight-digger/
├── src/
│   ├── soc_opros_loader.py      # Main data loading module
│   └── data_connector.py        # Generic data connection utilities
├── notebooks/
│   └── 01_soc_opros_demo.ipynb  # Interactive demonstration
├── tests/
│   └── test_soc_opros_loader.py # Comprehensive test suite
├── config/
│   └── google_sheets_config.py  # Configuration settings
├── .specify/                    # Spec-driven development framework
└── requirements.txt             # Dependencies
```

## Technical Features

### Encoding Handling
- Automatic UTF-8 encoding for Google Sheets CSV export
- Proper Cyrillic text decoding for respondent names
- Column name extraction from first data row

### Dynamic Structure
- Adapts to any number of survey statements
- Handles variable respondent counts
- Validates data consistency across operations

### Analysis Ready
- Clean pandas DataFrame output
- Named columns with respondent identifiers  
- Structured responses matrix for statistical analysis
- Export options (DataFrame, CSV, JSON)

## Testing

Run the comprehensive test suite:
```bash
pytest tests/test_soc_opros_loader.py -v
```

All 13 tests cover:
- Module availability and initialization ✓
- Google Sheets connectivity ✓
- Dynamic structure parsing ✓  
- Cyrillic text encoding ✓
- Data consistency validation ✓

## Binder Configuration

The project includes complete Binder configuration:
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment  
- `runtime.txt` - Python version specification
- `postBuild` - Post-installation setup

## Spec-Driven Development

This project follows the Specify framework with:
- Constitutional governance in `.specify/memory/constitution.md`
- Requirements-first development methodology
- Structured project memory and decision tracking

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Follow the constitutional principles in `.specify/memory/constitution.md`
3. Add tests for new features
4. Ensure Binder compatibility
5. Submit pull request

---

**Ready for Analysis**: Load 265 survey statements from 15 respondents with proper Cyrillic encoding in just one line of code.