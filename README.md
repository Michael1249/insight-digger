# 🔍 Insight Digger

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR_USERNAME/insight-digger/HEAD)

A spec-driven data analysis toolkit that transforms Google Sheets data into actionable insights through interactive Jupyter notebooks. Built with GitHub's Spec Kit for reproducible, collaborative data analysis.

## 🚀 Quick Start

### Option 1: Interactive Online (Recommended)
Click the Binder badge above to launch an interactive environment in your browser. No installation required!

### Option 2: Local Development
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/insight-digger.git
cd insight-digger

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

## 📊 What You Can Do

- **Connect to Google Sheets**: Automatically pull data from your Google Sheets
- **Analyze Data**: Use powerful pandas and numpy operations
- **Visualize Insights**: Create interactive charts with Plotly and Matplotlib
- **Share Results**: Deploy notebooks on Binder for collaborative analysis
- **Export Findings**: Save processed data and visualizations

## 📁 Project Structure

```
insight-digger/
├── notebooks/                     # Interactive Jupyter notebooks
│   ├── 01_data_ingestion.ipynb   # Google Sheets integration
│   ├── 02_data_analysis.ipynb    # Main analysis workflows
│   ├── 03_visualization.ipynb    # Data visualization examples
│   └── utils/                     # Utility notebooks
├── src/                          # Python modules
│   ├── data_connector.py         # Google Sheets API integration
│   ├── data_processor.py         # Data processing utilities
│   └── visualization.py          # Visualization helpers
├── data/                         # Data storage
│   ├── raw/                      # Raw data from sources
│   ├── processed/                # Cleaned, processed data
│   └── exports/                  # Final outputs
├── config/                       # Configuration files
└── spec/                         # Specification documents (Spec-driven development)
```

## 🔧 Data Sources

### Primary: Google Sheets API
- Real-time data access
- Automatic data refresh
- Supports authentication via service accounts

### Fallback: CSV Import
- Manual CSV upload support
- Simple file-based workflow
- No API limits or authentication required

## 🌐 Deployment Options

### Binder (Current Setup)
- **Free interactive environment**
- **2GB RAM, 1-6 hour sessions**
- **Perfect for sharing and collaboration**
- **Automatic environment setup**

### Local Development
- Full control over environment
- Persistent data storage
- Custom package installations
- Development and testing

## 🔍 Spec-Driven Development

This project follows GitHub's Spec Kit methodology:

1. **Constitution**: Project principles and development guidelines
2. **Specification**: Data sources, analysis goals, expected outputs
3. **Planning**: Technical implementation strategy
4. **Tasks**: Actionable implementation steps
5. **Implementation**: Iterative development with validation

### Available Specify Commands
- `/speckit.constitution` - Define project principles
- `/speckit.specify` - Create feature specifications
- `/speckit.plan` - Technical implementation planning
- `/speckit.tasks` - Generate actionable task lists
- `/speckit.implement` - Execute implementation

## 📋 Getting Started Guide

### For Data Analysts
1. Click the Binder badge to launch the environment
2. Open `01_data_ingestion.ipynb` to connect your data
3. Follow `02_data_analysis.ipynb` for analysis workflows
4. Use `03_visualization.ipynb` for creating charts

### For Developers
1. Clone the repository locally
2. Review the specification documents in `spec/`
3. Use Specify commands for feature development
4. Test changes in Binder environment

## 🔐 Authentication Setup

### Google Sheets API (Optional)
1. Create a Google Cloud Console project
2. Enable Google Sheets API
3. Create service account credentials
4. Share your Google Sheet with the service account email

### Local Development
```bash
# Set environment variables
export GOOGLE_SHEETS_CREDENTIALS_PATH="path/to/credentials.json"
export GOOGLE_SHEET_ID="your_sheet_id_here"
```

### Binder Deployment
- Use public sheets or pre-processed CSV files
- Avoid storing credentials in public repositories

## 📊 Example Analyses

- **Sales Performance**: Revenue trends, product analysis, regional insights
- **Survey Data**: Response analysis, sentiment tracking, demographic breakdowns
- **Financial Data**: Budget tracking, expense analysis, forecasting
- **Operations**: Process metrics, efficiency analysis, capacity planning

## 🤝 Contributing

1. Fork the repository
2. Use `/speckit.specify` to document new features
3. Follow the spec-driven development workflow
4. Create pull request with specification documents
5. Ensure Binder compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [GitHub Spec Kit](https://github.com/github/spec-kit) for spec-driven development
- Powered by [Binder](https://mybinder.org/) for interactive deployment
- Data visualization with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)

## 🔗 Links

- [Binder Documentation](https://mybinder.readthedocs.io/)
- [Google Sheets API Guide](https://developers.google.com/sheets/api)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [Spec-Driven Development](https://github.com/github/spec-kit/blob/main/spec-driven.md)

---

**Ready to dig into your data?** Click the Binder badge above and start analyzing!