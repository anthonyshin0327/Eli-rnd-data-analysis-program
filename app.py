import gradio as gr
import pandas as pd
from pycaret.regression import *
import plotly.express as px
import plotly.graph_objects as go
from pandas_profiling import ProfileReport
import webbrowser
import os
import secrets
import shutil

# --- Global Variables ---
DATA = None
CLEANED_DATA = None
PYCARET_SETUP = None
BEST_MODEL = None
REPORT_PATH = None

# --- Helper Functions ---
def process_lumos_data(df):
    """
    Processes LUMOS data by renaming columns and creating new features.
    """
    if 'strip_name' in df.columns:
        # Assuming the delimiter is consistent, e.g., '_' or '-'
        # This might need adjustment based on the actual strip name format
        df[['B7BSA', 'Fab', 'OD']] = df['strip_name'].str.split('-', expand=True, n=2)

    if 'line_peak_above_background_1' in df.columns:
        df = df.rename(columns={'line_peak_above_background_1': 'T'})
    if 'line_peak_above_background_2' in df.columns:
        df = df.rename(columns={'line_peak_above_background_2': 'C'})

    if 'T' in df.columns and 'C' in df.columns:
        df['T_norm'] = df['T'] / (df['T'] + df['C'])
        df['C_norm'] = df['C'] / (df['T'] + df['C'])
        df['T-C'] = df['T'] - df['C']
        df['C/T'] = df['C'] / df['T']
        df['T/C'] = df['T'] / df['C']
    return df

def generate_profile_report(df):
    """
    Generates a pandas profiling report for the given DataFrame.
    """
    profile = ProfileReport(df, title="Data Profile", explorative=True)
    report_path = f"report_{secrets.token_hex(4)}.html"
    profile.to_file(report_path)
    return report_path

# --- Gradio Functions ---
def upload_data(file):
    """
    Handles data upload and initial processing.
    """
    global DATA
    if file is not None:
        try:
            df = pd.read_csv(file.name)
            DATA = process_lumos_data(df.copy())
            profile_path = generate_profile_report(DATA)
            webbrowser.open(f'file://{os.path.realpath(profile_path)}')
            return DATA, gr.update(visible=True), gr.update(visible=True)
        except Exception as e:
            return str(e), gr.update(visible=False), gr.update(visible=False)
    return None, gr.update(visible=False), gr.update(visible=False)


def run_automl(target_variable, numeric_features, categorical_features):
    """
    Runs the AutoML pipeline using PyCaret.
    """
    global CLEANED_DATA, PYCARET_SETUP, BEST_MODEL
    if DATA is not None and target_variable:
        try:
            # Drop rows where target is NaN
            CLEANED_DATA = DATA.dropna(subset=[target_variable])

            # Setup PyCaret
            PYCARET_SETUP = setup(data=CLEANED_DATA,
                                  target=target_variable,
                                  numeric_features=numeric_features,
                                  categorical_features=categorical_features,
                                  silent=True,
                                  html=False)

            # Compare models and get the best one
            BEST_MODEL = compare_models()

            # Get the results
            results = pull()

            # Generate and save plots
            os.makedirs("plots", exist_ok=True)
            plot_model(BEST_MODEL, plot='residuals', save='plots/')
            plot_model(BEST_MODEL, plot='feature', save='plots/')


            return (results,
                    gr.update(value="plots/Residuals.png", visible=True),
                    gr.update(value="plots/Feature Importance.png", visible=True),
                    gr.update(visible=True)
                    )
        except Exception as e:
            return str(e), None, None, gr.update(visible=False)

    return "No data or target variable selected.", None, None, gr.update(visible=False)


def get_column_names_for_feature_selection():
    if DATA is not None:
        return list(DATA.columns)
    return []

def update_feature_lists(target_variable):
    if DATA is not None:
        all_cols = list(DATA.columns)
        if target_variable in all_cols:
            all_cols.remove(target_variable)

        numeric_features = list(DATA.select_dtypes(include=['number']).columns)
        if target_variable in numeric_features:
            numeric_features.remove(target_variable)
            
        categorical_features = list(DATA.select_dtypes(include=['object', 'category']).columns)
        
        return gr.update(choices=numeric_features, value=numeric_features), gr.update(choices=categorical_features, value=categorical_features)
    return gr.update(choices=[], value=[]), gr.update(choices=[], value=[])

def create_dashboard():
    """
    Creates the PyCaret dashboard.
    """
    if BEST_MODEL:
        try:
            dashboard(BEST_MODEL, display_format='inline')
            # The dashboard opens in a new tab. We will provide a link.
            # A bit of a hack to get the dashboard URL
            # This is not ideal and might break with future pycaret updates
            return "Dashboard is generated. Check your default browser."
        except Exception as e:
            return str(e)
    return "No model has been trained yet."


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Eli Health: LFA Data Analysis Platform")

    with gr.Tabs():
        with gr.TabItem("1. Data Upload & Profiling"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload CSV")
                    upload_button = gr.Button("Upload and Profile")
                with gr.Column():
                    data_output = gr.DataFrame(label="Raw Data")

        with gr.TabItem("2. AutoML Modeling"):
            with gr.Row():
                with gr.Column(scale=1):
                    target_variable_dropdown = gr.Dropdown(label="Select Target Variable", choices=get_column_names_for_feature_selection())
                    numeric_features_checkbox = gr.CheckboxGroup(label="Numeric Features")
                    categorical_features_checkbox = gr.CheckboxGroup(label="Categorical Features")
                    automl_button = gr.Button("Run AutoML")
                with gr.Column(scale=3):
                    model_leaderboard = gr.DataFrame(label="Model Leaderboard")
            
            target_variable_dropdown.change(
                fn=update_feature_lists,
                inputs=target_variable_dropdown,
                outputs=[numeric_features_checkbox, categorical_features_checkbox]
            )

        with gr.TabItem("3. Model Analysis & Visualization"):
            with gr.Row():
                residuals_plot = gr.Image(label="Residuals Plot", visible=False)
                feature_importance_plot = gr.Image(label="Feature Importance Plot", visible=False)
            
            with gr.Row():
                dashboard_button = gr.Button("Generate PyCaret Dashboard")
                dashboard_status = gr.Textbox(label="Dashboard Status", interactive=False)


    # --- Event Handlers ---
    upload_button.click(
        fn=upload_data,
        inputs=file_input,
        outputs=[data_output, target_variable_dropdown, automl_button]
    )

    automl_button.click(
        fn=run_automl,
        inputs=[target_variable_dropdown, numeric_features_checkbox, categorical_features_checkbox],
        outputs=[model_leaderboard, residuals_plot, feature_importance_plot, dashboard_button]
    )
    
    dashboard_button.click(
        fn=create_dashboard,
        inputs=[],
        outputs=[dashboard_status]
    )


if __name__ == "__main__":
    app.launch(debug=True)
