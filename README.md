# Insight Digger

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Michael1249/insight-digger/master?filepath=notebooks/01_soc_opros_demo.ipynb)

A spec-driven Jupyter notebook project for analyzing survey data from Google Sheets with Cyrillic text support.

## Features

- 🔄 **Dynamic Data Loading**: Automatically adapts to variable number of survey statements and respondents
- 🌐 **Cyrillic Text Support**: Proper UTF-8 encoding for Russian respondent names and content
- 📊 **Clean Data Interface**: Structured responses matrix ready for statistical analysis
- 🔗 **Google Sheets Integration**: Direct CSV export connection without authentication
- ⚡ **Binder Ready**: Run online instantly with pre-configured environment
- 🧪 **Comprehensive Testing**: Full test suite with 13 passing tests

## Quick Start

### Online (Recommended)
Click the Binder badge above to run the demo notebook online instantly.

### Local Setup
```bash
git clone https://github.com/Michael1249/insight-digger.git
cd insight-digger
pip install -r requirements.txt
jupyter lab notebooks/01_soc_opros_demo.ipynb
```

## Usage

### Simple Data Loading
```python
from soc_opros_loader import load_soc_opros_data

# Quick method - get clean data in one line
responses, structure = load_soc_opros_data()
print(f"Loaded {structure['total_statements']} statements from {structure['total_respondents']} respondents")
```

### Detailed Analysis
```python
from soc_opros_loader import SocOprosLoader

# Detailed method with full control
loader = SocOprosLoader()
data = loader.load_data()
statements = loader.get_statements()
summary = loader.get_response_summary()
```

## Data Source

The project analyzes the 'soc opros' survey containing:
- **265 philosophical/psychological statements** (variable content)
- **15 Russian respondents** with Cyrillic names (Амелия, Итанио, Отец, etc.)
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
- Proper Cyrillic text decoding for Russian content
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

**Ready for Analysis**: Load 265 survey statements from 15 Russian respondents with proper Cyrillic encoding in just one line of code.