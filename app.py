import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import shutil
import numpy as np
from itertools import product
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import curve_fit
import re
from collections import defaultdict
import pycaret.regression as reg
import pycaret.classification as clf

# --- Configuration & Setup ---
# Ensure kaleido is installed: pip install kaleido
TEMP_PLOT_DIR = "temp_plots"

def setup_temp_dir():
    """Clears and recreates the temporary directory for storing plot images."""
    if os.path.exists(TEMP_PLOT_DIR):
        shutil.rmtree(TEMP_PLOT_DIR)
    os.makedirs(TEMP_PLOT_DIR)

# --- Core Data & Naming Logic ---

def get_backend_name(original_name):
    """Creates a backend-safe identifier from a user-facing column name."""
    name = str(original_name)
    name = name.replace('/', '_div_')
    name = name.replace('-', '_minus_')
    name = re.sub(r'[^A-Za-z0-9_]+', '_', name)
    if name.startswith('_'): name = 'col' + name
    return name

def create_name_map(columns):
    """Creates a mapping from backend-safe names to original, user-friendly names."""
    return {get_backend_name(col): col for col in columns}

def process_lumos_data(df):
    """Processes raw data to create analytical features with user-friendly names."""
    processed_df = df.copy()
    rename_map = {
        'line_peak_above_background_1': 'T',
        'line_peak_above_background_2': 'C'
    }
    processed_df.rename(columns=rename_map, inplace=True)
    
    if 'T' in processed_df.columns and 'C' in processed_df.columns:
        processed_df['T'] = pd.to_numeric(processed_df['T'], errors='coerce')
        processed_df['C'] = pd.to_numeric(processed_df['C'], errors='coerce')
        processed_df.dropna(subset=['T', 'C'], inplace=True)

        sum_tc = processed_df['T'] + processed_df['C']
        processed_df['T_norm'] = np.divide(processed_df['T'], sum_tc, where=sum_tc!=0, out=np.zeros_like(processed_df['T'], dtype=float))
        processed_df['C_norm'] = np.divide(processed_df['C'], sum_tc, where=sum_tc!=0, out=np.zeros_like(processed_df['C'], dtype=float))
        processed_df['T-C'] = processed_df['T'] - processed_df['C']
        processed_df['C/T'] = np.divide(processed_df['C'], processed_df['T'], where=processed_df['T']!=0, out=np.full_like(processed_df['C'], np.inf, dtype=float))
        processed_df['T/C'] = np.divide(processed_df['T'], processed_df['C'], where=processed_df['C']!=0, out=np.full_like(processed_df['T'], np.inf, dtype=float))
    return processed_df

def five_pl(x, a, b, c, d, g):
    """Five-parameter logistic function for curve fitting."""
    return d + ((a - d) / (1 + (x / c)**b)**g)

def fit_5pl_and_plot(fig, df, x_col, y_col, group_col=None):
    """Fits 5PL curves to data and adds them to a Plotly figure."""
    df_to_fit = df.copy()
    df_to_fit[x_col] = pd.to_numeric(df_to_fit[x_col], errors='coerce')
    df_to_fit[y_col] = pd.to_numeric(df_to_fit[y_col], errors='coerce')
    df_to_fit.dropna(subset=[x_col, y_col], inplace=True)
    
    if df_to_fit.empty: return

    unique_groups = sorted(df_to_fit[group_col].unique()) if group_col and group_col in df_to_fit.columns else [None]
    color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, label in enumerate(unique_groups)}
    
    for group in unique_groups:
        group_df = df_to_fit if group is None else df_to_fit[df_to_fit[group_col] == group]
        if len(group_df) < 5: continue
        xdata, ydata = group_df[x_col], group_df[y_col]
        try:
            p0 = [min(ydata), 1, np.median(xdata), max(ydata), 1]
            params, _ = curve_fit(five_pl, xdata, ydata, p0=p0, maxfev=10000)
            x_fit = np.linspace(min(xdata), max(xdata), 200)
            y_fit = five_pl(x_fit, *params)
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'5PL Fit ({group})' if group else '5PL Fit', line=dict(color=color_map.get(group, 'cyan'), width=2, dash='solid')))
        except Exception as e:
            print(f"5PL fit failed for group {group}: {e}")

def save_plots_to_temp_dir(plots):
    """Saves a list of Plotly figures to a temp directory and returns file paths."""
    setup_temp_dir()
    plot_files = []
    for i, fig in enumerate(plots):
        if fig is not None:
            plot_path = os.path.join(TEMP_PLOT_DIR, f"plot_{i}.png")
            try:
                fig.write_image(plot_path, engine="kaleido")
                plot_files.append(plot_path)
            except Exception as e:
                print(f"Failed to save plot {i}: {e}")
    return plot_files

# --- Gradio UI Functions ---

def upload_and_process_data(file, progress=gr.Progress()):
    """Handles file upload, data processing, and state initialization with new defaults."""
    error_return_values = (
        None, None, {}, "Please upload a CSV file.", 
        gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[], value=[]), 
        gr.update(choices=[], value=[]), gr.update(choices=[], value=None), 
        gr.update(choices=[], value=None), gr.update(interactive=False),
        [], [], None, None
    )
    if file is None: return error_return_values

    try:
        progress(0, desc="Reading file...")
        raw_df = pd.read_csv(file.name)
        progress(0.4, desc="Processing data...")
        processed_df = process_lumos_data(raw_df)
        name_map = create_name_map(processed_df.columns)
        
        progress(0.9, desc="Updating UI...")
        all_cols = ["None"] + list(processed_df.columns)
        numeric_cols = list(processed_df.select_dtypes(include=np.number).columns)
        
        return (
            raw_df, processed_df, name_map, "Data uploaded and processed.",
            gr.update(choices=numeric_cols, value=numeric_cols, interactive=True), # Pre-select X
            gr.update(choices=numeric_cols, value=numeric_cols, interactive=True), # Pre-select Y
            gr.update(choices=all_cols, interactive=True),
            gr.update(choices=list(processed_df.columns), value=None, interactive=True),
            gr.update(interactive=True), # Enable model type selector
            gr.update(choices=list(processed_df.columns), value=None, interactive=True),
            gr.update(interactive=False),
            [], [],
            gr.update(value=raw_df), 
            gr.update(value=processed_df)
        )
    except Exception as e:
        error_with_msg = list(error_return_values)
        error_with_msg[3] = f"An error occurred: {e}"
        return tuple(error_with_msg)

def generate_manual_eda_plot(data, x_col, y_col, color_col):
    """Generates a single, manually configured plot."""
    if data is None: return None, "Upload data to begin EDA."
    if not x_col or not y_col: return go.Figure(), "Please select both X and Y axes."
    
    title = f'Scatter Plot: {y_col} vs. {x_col}' + (f' (Colored by {color_col})' if color_col != "None" else '')
    fig = px.scatter(data, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=title, template="plotly_dark")
    fit_5pl_and_plot(fig, data, x_col, y_col, color_col if color_col != "None" else None)
    return fig, f"Displaying: {title}"

def generate_automated_eda(data, x_cols, y_cols, color_col, progress=gr.Progress()):
    """Generates a gallery of plots for high-throughput EDA."""
    if data is None: return [], "Upload data to generate plots.", []
    if not x_cols or not y_cols: return [], "Please select variables for both X and Y axes.", []

    plots = []
    plot_combinations = list(product(x_cols, y_cols))
    
    for i, (x_col, y_col) in enumerate(plot_combinations):
        if x_col == y_col: continue
        progress(i / len(plot_combinations), desc=f"Plotting: {y_col} vs. {x_col}")
        
        title = f'{y_col} vs. {x_col}' + (f' (Colored by {color_col})' if color_col != "None" else '')
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=title, template="plotly_dark")
        fit_5pl_and_plot(fig, data, x_col, y_col, color_col if color_col != "None" else None)
        plots.append(fig)

    if not plots: return [], "No plots generated.", []
    
    plot_files = save_plots_to_temp_dir(plots)
    return plot_files, f"{len(plots)} plots generated.", plots

def run_automl(data, name_map, target_variable, model_type, progress=gr.Progress()):
    """Runs the AutoML pipeline for either Regression or Classification."""
    interactive_plots = []
    try:
        if data is None or target_variable is None:
            raise ValueError("Data or target variable not provided.")

        pc = reg if model_type == 'Regression' else clf
        optimize_metric = 'R2' if model_type == 'Regression' else 'Accuracy'
        
        backend_df = data.rename(columns={v: k for k, v in name_map.items()})
        backend_target = get_backend_name(target_variable)

        yield None, None, gr.update(visible=True), "Initializing AutoML...", [], interactive_plots

        n_samples = len(backend_df.dropna(subset=[backend_target]))
        fold_strategy = 'loocv' if n_samples < 50 else 10
        
        yield None, None, gr.update(visible=True), f"Setting up PyCaret ({fold_strategy} folds)...", [], interactive_plots
        
        s = pc.setup(data=backend_df, target=backend_target, 
                     html=False, verbose=False, session_id=123,
                     polynomial_features=False, transformation=True, normalize=True,
                     feature_selection=True, fold_strategy=fold_strategy)
        
        yield None, None, gr.update(visible=True), "Comparing models...", [], interactive_plots
        best_model = pc.compare_models(sort=optimize_metric)
        leaderboard = pc.pull()
        
        tuned_model = None
        try:
            yield leaderboard, None, gr.update(visible=True), f"Tuning best model ({optimize_metric})...", [], interactive_plots
            tuned_model = pc.tune_model(best_model, optimize=optimize_metric, verbose=False)
        except Exception as e:
            print(f"Model tuning failed, proceeding with original best model. Error: {e}")
            tuned_model = None

        final_model_obj = best_model
        if tuned_model is not None:
            tuned_results = pc.pull()
            if not tuned_results.empty and tuned_results[optimize_metric][0] > leaderboard[optimize_metric][0]:
                final_model_obj = tuned_model
                print(f"Using tuned model as it performed better on {optimize_metric}.")
            else:
                print(f"Using original best model as tuning did not improve {optimize_metric}.")
        else:
            print("Using original best model because tuning failed.")

        yield leaderboard, None, gr.update(visible=True), "Finalizing model...", [], interactive_plots
        final_model = pc.finalize_model(final_model_obj)
        final_results = pc.pull()
        kpi_html = create_kpi_dashboard(final_results, name_map, model_type)

        plot_names = ['residuals', 'feature_all', 'prediction_error'] if model_type == 'Regression' else ['auc', 'confusion_matrix', 'feature_all']
        for plot_name in plot_names:
            try:
                fig = pc.plot_model(final_model, plot=plot_name, save=False, display_format='plotly', use_train_data=True)
                if fig:
                    fig.update_layout(template='plotly_dark', title_text=f"{plot_name.replace('_', ' ').title()} Plot")
                    interactive_plots.append(fig)
            except Exception as e:
                print(f"Could not generate plot: {plot_name}. Error: {e}")
        
        plot_files = save_plots_to_temp_dir(interactive_plots)
        yield final_results, kpi_html, gr.update(visible=False), "AutoML complete.", plot_files, interactive_plots

    except Exception as e:
        yield None, None, gr.update(visible=False), f"An error occurred: {e}", [], []

def create_kpi_dashboard(results_df, name_map, model_type):
    """Creates an HTML dashboard for key performance indicators."""
    if results_df.empty: return ""
    model_name = results_df.index[0]
    
    if model_type == 'Regression':
        metric1_name, metric1_val = 'R²', f"{results_df.loc[model_name, 'R2']:.4f}"
        metric2_name, metric2_val = 'MAE', f"{results_df.loc[model_name, 'MAE']:.2f}"
        metric3_name, metric3_val = 'RMSE', f"{results_df.loc[model_name, 'RMSE']:.2f}"
    else: # Classification
        metric1_name, metric1_val = 'Accuracy', f"{results_df.loc[model_name, 'Accuracy']:.4f}"
        metric2_name, metric2_val = 'AUC', f"{results_df.loc[model_name, 'AUC']:.4f}"
        metric3_name, metric3_val = 'F1', f"{results_df.loc[model_name, 'F1']:.4f}"

    kpi_style = "padding: 10px; margin: 5px; border-radius: 8px; background-color: #2F2F2F; text-align: center; color: white; flex-grow: 1;"
    return f"""
    <div style="display: flex; justify-content: space-around; width: 100%; gap: 10px;">
        <div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">Best Model</h3><p style="margin:0;">{model_name}</p></div>
        <div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">{metric1_name}</h3><p style="margin:0;">{metric1_val}</p></div>
        <div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">{metric2_name}</h3><p style="margin:0;">{metric2_val}</p></div>
        <div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">{metric3_name}</h3><p style="margin:0;">{metric3_val}</p></div>
    </div>"""

# --- Main App UI ---
css = """
#main_container { background-color: #121212; }
#control_sidebar { background-color: #1E1E1E; padding: 15px; border-radius: 10px; }
.gradio-container { max-width: 100% !important; }
.section_accordion .label-wrap { background-color: #2F2F2F !important; color: white !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.orange), css=css, elem_id="main_container") as app:
    raw_data, processed_data, name_map = gr.State(), gr.State(), gr.State({})
    automl_interactive_plots, eda_interactive_plots = gr.State([]), gr.State([])

    gr.Markdown("# Eli Health: LFA Data Analysis Platform", elem_id="main_title")

    with gr.Row():
        with gr.Column(scale=1, min_width=350, elem_id="control_sidebar"):
            gr.Markdown("## Controls")
            
            with gr.Accordion("1. Data Upload", open=True, elem_classes="section_accordion"):
                file_input = gr.File(label="Upload CSV File")
                upload_button = gr.Button("Upload and Process", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False, lines=2)
            
            with gr.Accordion("2. EDA Setup", open=False, elem_classes="section_accordion"):
                with gr.Tabs():
                    with gr.TabItem("High-Throughput", id=0):
                        ht_x_cols = gr.CheckboxGroup(label="Select X-Axis Variables", choices=[], value=[])
                        ht_y_cols = gr.CheckboxGroup(label="Select Y-Axis Variables", choices=[], value=[])
                        ht_eda_color_col = gr.Dropdown(label="Color By (Categorical)", choices=[])
                        ht_eda_button = gr.Button("Generate Automated Plots", variant="primary")
                    with gr.TabItem("Manual Plot", id=1):
                        manual_x_axis = gr.Dropdown(label="Select X-Axis", choices=[])
                        manual_y_axis = gr.Dropdown(label="Select Y-Axis", choices=[])
                        manual_color_col = gr.Dropdown(label="Color By", choices=[])
                        manual_plot_button = gr.Button("Generate Manual Plot", variant="secondary")

            with gr.Accordion("3. AutoML Setup", open=False, elem_classes="section_accordion"):
                model_type_selector = gr.Radio(choices=['Regression', 'Classification'], label="Select Analysis Type", value='Regression', interactive=False)
                target_variable_dropdown = gr.Dropdown(label="Select Target Variable", choices=[], interactive=False)
                automl_button = gr.Button("Run AutoML", variant="primary", interactive=False)

        with gr.Column(scale=3):
            with gr.Accordion("Data Preview", open=True, elem_classes="section_accordion"):
                with gr.Tabs():
                    with gr.TabItem("Processed Data"): data_output_preview = gr.DataFrame(wrap=True)
                    with gr.TabItem("Raw Data"): raw_data_preview = gr.DataFrame(wrap=True)

            with gr.Accordion("Exploratory Analysis", open=True, elem_classes="section_accordion"):
                interactive_eda_plot = gr.Plot(label="Interactive EDA Plot", visible=False)
                ht_eda_status = gr.Textbox(label="Status", interactive=False, visible=True)
                gr.Markdown("Click any plot below to view it interactively above.")
                ht_eda_gallery = gr.Gallery(label="Automated EDA Plots", columns=3, object_fit="contain", height="auto", type="filepath")
            
            with gr.Accordion("AutoML Modeling", open=True, elem_classes="section_accordion"):
                interactive_model_plot = gr.Plot(label="Interactive Model Plot", visible=False)
                status_box_automl = gr.Textbox(label="Live Status", interactive=False)
                status_spinner = gr.HTML('<div class="generating"></div>', visible=False)
                kpi_dashboard_html = gr.HTML()
                with gr.Tabs():
                    with gr.TabItem("Model Leaderboard"): model_leaderboard = gr.DataFrame()
                    with gr.TabItem("Model Diagnostic Plots"):
                        gr.Markdown("Click any plot below to view it interactively above.")
                        model_plot_gallery = gr.Gallery(label="Generated Model Plots", columns=3, object_fit="contain", height="auto")

    # --- Event Handlers ---
    upload_outputs = [
        raw_data, processed_data, name_map, upload_status, ht_x_cols, ht_y_cols, ht_eda_color_col,
        target_variable_dropdown, model_type_selector, manual_x_axis, manual_y_axis, manual_color_col,
        automl_button, ht_eda_gallery, model_plot_gallery, raw_data_preview, data_output_preview
    ]
    
    @upload_button.click(inputs=[file_input], outputs=upload_outputs)
    def full_upload_process(file, progress=gr.Progress()):
        error_return_values = (None, None, {}, "Please upload a CSV file.", *([gr.update(choices=[], value=[])] * 3), *([gr.update(choices=[], value=None, interactive=False)]*5), *([gr.update(value=None)]*4))
        if file is None: return error_return_values
        try:
            progress(0, desc="Reading file...")
            raw_df = pd.read_csv(file.name)
            progress(0.4, desc="Processing data...")
            processed_df = process_lumos_data(raw_df)
            name_map = create_name_map(processed_df.columns)
            
            progress(0.9, desc="Updating UI...")
            all_cols = ["None"] + list(processed_df.columns)
            numeric_cols = list(processed_df.select_dtypes(include=np.number).columns)
            
            return (raw_df, processed_df, name_map, "Data uploaded and processed.",
                    gr.update(choices=numeric_cols, value=numeric_cols, interactive=True), # ht_x_cols
                    gr.update(choices=numeric_cols, value=numeric_cols, interactive=True), # ht_y_cols
                    gr.update(choices=all_cols, interactive=True), # ht_eda_color_col
                    gr.update(choices=list(processed_df.columns), interactive=True), # target_variable_dropdown
                    gr.update(interactive=True), # model_type_selector
                    gr.update(choices=numeric_cols, interactive=True), # manual_x_axis
                    gr.update(choices=numeric_cols, interactive=True), # manual_y_axis
                    gr.update(choices=all_cols, interactive=True), # manual_color_col
                    gr.update(interactive=False), # automl_button
                    gr.update(value=[]), gr.update(value=[]), # galleries
                    gr.update(value=raw_df), gr.update(value=processed_df))
        except Exception as e:
            return (None, None, {}, f"An error occurred: {e}", *([gr.update(choices=[], value=[])] * 3), *([gr.update(choices=[], value=None, interactive=False)]*5), *([gr.update(value=None)]*4))

    manual_plot_button.click(fn=generate_manual_eda_plot, inputs=[processed_data, manual_x_axis, manual_y_axis, manual_color_col], outputs=[interactive_eda_plot, ht_eda_status]).then(lambda: gr.update(visible=True), None, interactive_eda_plot)
    ht_eda_button.click(fn=generate_automated_eda, inputs=[processed_data, ht_x_cols, ht_y_cols, ht_eda_color_col], outputs=[ht_eda_gallery, ht_eda_status, eda_interactive_plots])
    target_variable_dropdown.change(lambda target: gr.update(interactive=target is not None), target_variable_dropdown, automl_button)
    automl_button.click(fn=run_automl, inputs=[processed_data, name_map, target_variable_dropdown, model_type_selector], outputs=[model_leaderboard, kpi_dashboard_html, status_spinner, status_box_automl, model_plot_gallery, automl_interactive_plots])

    def show_interactive_plot(evt: gr.SelectData, plots_list: list):
        if plots_list and evt.index < len(plots_list):
            return gr.update(value=plots_list[evt.index], visible=True)
        return gr.update(visible=False)

    ht_eda_gallery.select(fn=show_interactive_plot, inputs=[eda_interactive_plots], outputs=[interactive_eda_plot])
    model_plot_gallery.select(fn=show_interactive_plot, inputs=[automl_interactive_plots], outputs=[interactive_model_plot])

if __name__ == "__main__":
    setup_temp_dir()
    app.launch(debug=True, show_error=True)

