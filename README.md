# 🧬 LFA Data Analysis Platform

A comprehensive, production-ready web application for automated analysis of lateral flow immunoassay (LFA) development data. Built specifically for R&D scientists at Eli Health to optimize assay sensitivity, specificity, and robustness through advanced machine learning and statistical analysis.

## 🚀 Features

### Core Capabilities
- **Automated ML Pipeline**: PyCaret-powered regression and classification with hyperparameter tuning
- **LUMOS Data Integration**: Specialized processing for LUMOS system outputs
- **DoE Analysis**: Design of Experiments support with interaction plots and response surfaces
- **Advanced Visualizations**: Interactive plots, 3D surfaces, correlation matrices, and diagnostic plots
- **Model Interpretability**: SHAP analysis and feature importance visualization
- **Batch Processing**: Large-scale predictions and data processing
- **Export & Reports**: Comprehensive HTML, PDF, and Excel reports

### Scientific Focus
- **LFA-Specific Features**: Automatic T/C ratio calculations, normalized responses
- **DoE Support**: Central Composite Design, full factorial, response surface methodology
- **Optimization Tools**: Response optimization and experimental design generation
- **Statistical Validation**: Model assumption testing and diagnostic analysis

## 📋 Requirements

### System Requirements
- Python 3.11+
- 8GB+ RAM (recommended for large datasets)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- PyCaret 3.3.2
- Gradio 4.44.0
- Pandas 2.1.4
- Plotly 5.18.0
- Scikit-learn 1.3.2

## 🛠️ Installation

### Quick Start
```bash
# Clone or download the application files
# Ensure you have app.py and requirements.txt

# Create virtual environment (recommended)
python -m venv lfa_env
source lfa_env/bin/activate  # On Windows: lfa_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
python app.py
```

### Production Deployment
```bash
# Install with production dependencies
pip install -r requirements.txt gunicorn

# Launch with Gunicorn
gunicorn -w 4 -b 0.0.0.0:7860 app:app
```

## 📊 Usage Guide

### 1. Data Upload & Processing

#### Supported Formats
- **CSV files**: Standard comma-separated values
- **Excel files**: .xlsx and .xls formats
- **LUMOS data**: Automatically detected and processed

#### LUMOS Data Processing
When LUMOS data is detected, the system automatically:
- Splits `strip_name` by common delimiters
- Renames `line_peak_above_background_1` → `T`
- Renames `line_peak_above_background_2` → `C`
- Generates derived features:
  - `T_norm = T/(T+C)`
  - `C_norm = C/(T+C)`
  - `T_minus_C = T - C`
  - `C_over_T = C/T`
  - `T_over_C = T/C`

#### Data Cleaning
- Automatic outlier detection and capping
- Missing value handling
- Data type inference and conversion
- Whitespace and formatting cleanup

### 2. ML Analysis Setup

#### Target Selection
Choose your response variable from the dropdown. The system supports:
- **Continuous variables**: Sensitivity, specificity, signal intensity
- **Categorical variables**: Pass/fail, quality grades
- **Auto-detection**: Automatically determines analysis type

#### Analysis Types
- **Auto**: Intelligent detection based on target variable
- **Regression**: For continuous responses
- **Classification**: For categorical outcomes

### 3. AutoML Pipeline

The automated pipeline includes:
- **Preprocessing**: Scaling, encoding, feature engineering
- **Model Selection**: Tests multiple algorithms (RF, XGBoost, LightGBM, etc.)
- **Hyperparameter Tuning**: Grid search optimization
- **Cross-Validation**: Robust performance estimation
- **Feature Engineering**: Polynomial and interaction features

### 4. Visualization Dashboard

#### Available Visualizations
- **Correlation Matrix**: Feature relationships heatmap
- **Distribution Plots**: Individual feature distributions
- **Scatter Matrix**: Pairwise feature relationships
- **3D Plots**: Three-dimensional feature space
- **Response Surfaces**: DoE optimization surfaces

#### Interactive Features
- Zoom, pan, and hover tooltips
- Downloadable high-resolution images
- Customizable color schemes and layouts

### 5. Model Diagnostics

#### Regression Diagnostics
- **Residual Plots**: Check for patterns and outliers
- **Prediction Error**: Actual vs predicted scatter
- **Learning Curves**: Training and validation performance

#### Classification Diagnostics
- **Confusion Matrix**: Classification accuracy breakdown
- **ROC Curves**: Receiver operating characteristic
- **Precision-Recall**: Performance across thresholds

### 6. Predictions & Optimization

#### Single Predictions
Upload new data files for batch predictions with confidence intervals.

#### Response Optimization
- Define optimization targets (maximize/minimize)
- Set variable bounds and constraints
- Generate optimal experimental conditions

### 7. Export & Reporting

#### Available Exports
- **JSON Reports**: Machine-readable analysis summaries
- **Excel Workbooks**: Multi-sheet data and results
- **Model Files**: Serialized models for reuse
- **Visualization Images**: High-resolution plots

## 🔬 Scientific Workflows

### Typical DoE Analysis Workflow
1. **Upload Data**: Load DoE experimental results
2. **Setup Analysis**: Select response variable and factors
3. **Run AutoML**: Generate predictive models
4. **Visualize**: Create response surfaces and interaction plots
5. **Optimize**: Find optimal factor settings
6. **Validate**: Check model assumptions and diagnostics
7. **Export**: Generate comprehensive reports

### LFA Development Workflow
1. **LUMOS Integration**: Upload LUMOS system outputs
2. **Feature Engineering**: Automatic T/C calculations
3. **Sensitivity Analysis**: Model assay performance
4. **Optimization**: Maximize sensitivity/specificity
5. **Batch Prediction**: Evaluate new formulations
6. **Report Generation**: Document findings and recommendations

## 🎯 Best Practices

### Data Preparation
- **Clean Data**: Remove obvious errors before upload
- **Consistent Units**: Ensure all measurements use consistent units
- **Complete Records**: Minimize missing values where possible
- **Metadata**: Include experimental conditions and batch information

### Model Selection
- **Start Simple**: Begin with linear models for interpretability
- **Cross-Validate**: Always use the built-in cross-validation
- **Check Assumptions**: Validate model assumptions before deploying
- **Document Everything**: Use the export features to maintain records

### Experimental Design
- **Balanced Designs**: Use proper DoE principles
- **Replication**: Include technical and biological replicates
- **Randomization**: Randomize experimental order
- **Controls**: Include appropriate positive and negative controls

## 🔧 Troubleshooting

### Common Issues

#### Memory Errors
- Reduce dataset size or increase system RAM
- Use data sampling for initial exploration
- Close other applications to free memory

#### Model Training Failures
- Check for missing values in target variable
- Ensure sufficient data for training (>20 samples recommended)
- Verify data types are appropriate

#### Visualization Errors
- Ensure numeric data for continuous plots
- Check for infinite or NaN values
- Reduce data complexity for 3D plots

#### File Upload Issues
- Verify file format (CSV, Excel only)
- Check file encoding (UTF-8 recommended)
- Ensure reasonable file size (<100MB)

### Performance Optimization
- Use sampling for large datasets (>10,000 rows)
- Limit feature engineering for high-dimensional data
- Consider data reduction techniques for speed

## 📈 Advanced Features

### Custom Model Development
The platform supports custom model integration through PyCaret's extensible framework.

### API Integration
The Gradio interface can be integrated with existing laboratory information systems (LIMS).

### Batch Processing
Large-scale analysis capabilities for high-throughput experimental data.

### Version Control
Built-in experiment tracking and model versioning for reproducible research.

## 🤝 Support & Contribution

### Getting Help
- Check the troubleshooting section above
- Review error messages in the interface
- Ensure all requirements are properly installed

### Contributing
This is a specialized tool for Eli Health R&D. For modifications or enhancements:
1. Document the scientific rationale
2. Test with representative LFA data
3. Ensure backward compatibility
4. Update documentation accordingly

## 📄 License & Disclaimer

This tool is designed for research purposes in LFA development. Users are responsible for:
- Validating results with appropriate controls
- Following good laboratory practices
- Maintaining data security and confidentiality
- Complying with relevant regulations

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with full AutoML pipeline
- LUMOS data integration
- DoE analysis capabilities
- Comprehensive visualization dashboard
- Export and reporting features

---

**Built for Eli Health R&D Team**  
*Optimizing lateral flow immunoassay development through data-driven insights*