# Eli Health LFA Data Analysis App

This web application helps R&D scientists at Eli Health analyze experimental data from Lateral Flow Immunoassay (LFA) development. It leverages PyCaret's AutoML capabilities to build robust regression models, optimize assay performance, and provide insightful, interactive visualizations.

## Features

* **Data Upload:** Supports CSV and Excel file uploads for experimental data.
* **Automated Machine Learning:** Utilizes PyCaret for end-to-end AutoML, including:
    * Robust preprocessing (handling missing values, scaling, etc.)
    * Automatic categorical variable encoding.
    * Generation of polynomial features and interaction terms.
    * Hyperparameter tuning of various regression models.
    * Cross-validation for reliable model evaluation.
* **Interactive Visualizations:** Dashboard view of model performance metrics and key plots (Actual vs. Predicted, Residuals, Feature Importance).
* **Interpretability:** Clear explanations of results tailored for scientists, not necessarily data scientists.
* **Download Options:** Download model predictions and the trained PyCaret pipeline for future use or deployment.
* **User-Friendly Interface:** Built with NiceGUI for an intuitive and responsive user experience.

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

* Python 3.11 (due to PyCaret compatibility requirements). You mentioned you've already downgraded your codespace to 3.11, which is perfect.
* `pip` (Python package installer)

### 1. Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git clone [https://github.com/your-username/eli_health_lfa_analysis.git](https://github.com/your-username/eli_health_lfa_analysis.git)
cd eli_health_lfa_analysis