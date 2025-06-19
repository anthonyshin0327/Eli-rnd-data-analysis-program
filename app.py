import gradio as gr
import pandas as pd
from pycaret.regression import *
import plotly.express as px
import plotly.graph_objects as go
import os
import secrets
import shutil
import numpy as np
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import io
from scipy.optimize import curve_fit
import re

# --- Scientific & Helper Functions ---

def five_pl(x, a, b, c, d, g):
    """Five-parameter logistic function for curve fitting."""
    return d + ((a - d) / (1 + (x / c)**b)**g)

def fit_5pl_and_plot(fig, df, x_col, y_col):
    """Fits a 5PL curve to data and adds the trace and parameters to a Plotly figure."""
    df_clean = df[[x_col, y_col]].dropna()
    xdata = df_clean[x_col]
    ydata = df_clean[y_col]

    if len(df_clean) < 5:
        return "Not enough data for 5PL fit."

    try:
        p0 = [min(ydata), 1, np.median(xdata), max(ydata), 1]
        params, _ = curve_fit(five_pl, xdata, ydata, p0=p0, maxfev=10000)
        
        x_fit = np.linspace(min(xdata), max(xdata), 200)
        y_fit = five_pl(x_fit, *params)
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='5PL Fit', line=dict(color='cyan', width=2)))
        
        equation_text = (f"5PL Fit Parameters:<br>"
                         f"a(min)={params[0]:.2f}, b(slope)={params[1]:.2f}, c(EC50)={params[2]:.2f}<br>"
                         f"d(max)={params[3]:.2f}, g(asym)={params[4]:.2f}")
        return equation_text
    except Exception as e:
        print(f"5PL fit failed for {y_col} vs {x_col}: {e}")
        return "5PL fit failed to converge."

def sanitize_filename(name):
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]',"_", name)

def process_lumos_data(df):
    """Processes LUMOS data by renaming columns and creating new features."""
    processed_df = df.copy()
    if 'strip_name' in processed_df.columns:
        try:
            split_data = processed_df['strip_name'].str.split('-', expand=True)
            for i in range(split_data.shape[1]):
                processed_df[f'strip_part_{i+1}'] = split_data[i]
        except Exception as e:
            print(f"Could not process 'strip_name': {e}")

    if 'line_peak_above_background_1' in processed_df.columns:
        processed_df = processed_df.rename(columns={'line_peak_above_background_1': 'T'})
    if 'line_peak_above_background_2' in processed_df.columns:
        processed_df = processed_df.rename(columns={'line_peak_above_background_2': 'C'})

    if 'T' in processed_df.columns and 'C' in processed_df.columns:
        for col in ['T', 'C']:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df.dropna(subset=['T', 'C'], inplace=True)
        
        sum_tc = processed_df['T'] + processed_df['C']
        processed_df['T_norm'] = np.divide(processed_df['T'], sum_tc, where=sum_tc != 0, out=np.zeros_like(processed_df['T'], dtype=float))
        processed_df['C_norm'] = np.divide(processed_df['C'], sum_tc, where=sum_tc != 0, out=np.zeros_like(df['C'], dtype=float))
        processed_df['T-C'] = processed_df['T'] - processed_df['C']
        processed_df['C/T'] = np.divide(processed_df['C'], processed_df['T'], where=processed_df['T'] != 0, out=np.full_like(processed_df['C'], np.inf, dtype=float))
        processed_df['T/C'] = np.divide(processed_df['T'], processed_df['C'], where=processed_df['C'] != 0, out=np.full_like(processed_df['T'], np.inf, dtype=float))
    return processed_df

# --- Gradio UI Functions ---

def upload_and_process_data(file, progress=gr.Progress()):
    if file is None:
        return None, None, None, "Please upload a CSV file.", gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
    try:
        progress(0, desc="Reading file...")
        raw_df = pd.read_csv(file.name)
        progress(0.4, desc="Processing data...")
        processed_df = process_lumos_data(raw_df)
        progress(0.9, desc="Updating UI...")
        all_cols = ["None"] + list(processed_df.columns)
        numeric_cols = list(processed_df.select_dtypes(include=np.number).columns)
        
        return (
            raw_df, processed_df, processed_df, "Data uploaded. Ready for EDA and Modeling.",
            gr.update(choices=numeric_cols, interactive=True), gr.update(choices=numeric_cols, interactive=True),
            gr.update(choices=all_cols, interactive=True), gr.update(choices=list(processed_df.columns), interactive=True)
        )
    except Exception as e:
        return None, None, None, f"An error occurred: {e}", gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

def generate_manual_eda_plot(data, x_col, y_col, color_col, fit_5pl):
    if data is None: return None, "Upload data to begin EDA."
    if not x_col or not y_col: return go.Figure(), "Please select both X and Y axes."
    
    data_to_plot = data.copy()
    color_arg = color_col if color_col != "None" else None

    if color_arg and pd.api.types.is_numeric_dtype(data_to_plot[color_arg]) and data_to_plot[color_arg].nunique() > 10:
        binned_col_name = f"{color_arg} (binned)"
        data_to_plot[binned_col_name] = pd.qcut(data_to_plot[color_arg], q=5, duplicates='drop').astype(str)
        color_arg = binned_col_name
    
    title = f'Scatter Plot: {y_col} vs. {x_col}' + (f' (Colored by {color_col})' if color_col != "None" else '')
    fig = px.scatter(data_to_plot, x=x_col, y=y_col, color=color_arg, title=title, template="plotly_dark")

    if fit_5pl:
        fit_info = fit_5pl_and_plot(fig, data_to_plot, x_col, y_col)
        fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=fit_info, showarrow=False,
                           font=dict(size=12, color="white"), align="left", bgcolor="rgba(0,0,0,0.5)")
    return fig, f"Displaying: {title}"

def generate_automated_eda(data, color_col, fit_5pl, progress=gr.Progress()):
    if data is None: return [], "Upload data to generate plots."
    
    plot_dir = "plots_gallery"
    if os.path.exists(plot_dir): shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    
    data_to_plot = data.copy()
    numeric_cols = data_to_plot.select_dtypes(include=np.number).columns
    categorical_cols = data_to_plot.select_dtypes(include=['object', 'category']).columns
    color_arg = color_col if color_col != "None" else None

    if color_arg and pd.api.types.is_numeric_dtype(data_to_plot[color_arg]) and data_to_plot[color_arg].nunique() > 10:
        binned_col_name = f"{color_arg} (binned)"
        data_to_plot[binned_col_name] = pd.qcut(data_to_plot[color_arg], q=5, duplicates='drop').astype(str)
        color_arg = binned_col_name

    plot_files = []
    total_plots = len(list(combinations(numeric_cols, 2))) + len(numeric_cols) * len(categorical_cols)
    plots_done = 0

    if len(numeric_cols) >= 2:
        for x_col, y_col in combinations(numeric_cols, 2):
            progress(plots_done / total_plots if total_plots > 0 else 0, desc=f"Scatter: {y_col} vs. {x_col}")
            title = f'{y_col} vs. {x_col}'
            fig = px.scatter(data_to_plot, x=x_col, y=y_col, color=color_arg, title=title, template="plotly_dark")
            if fit_5pl:
                fit_info = fit_5pl_and_plot(fig, data_to_plot, x_col, y_col)
                fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=fit_info, showarrow=False, font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)")
            
            safe_x = sanitize_filename(x_col)
            safe_y = sanitize_filename(y_col)
            filepath = os.path.join(plot_dir, f'scatter_{safe_y}_vs_{safe_x}.png')
            fig.write_image(filepath)
            plot_files.append(filepath)
            plots_done += 1

    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        for x_col, y_col in [(cat, num) for cat in categorical_cols for num in numeric_cols]:
            progress(plots_done / total_plots if total_plots > 0 else 0, desc=f"Box Plot: {y_col} vs. {x_col}")
            title = f'Distribution of {y_col} by {x_col}'
            fig = px.box(data_to_plot, x=x_col, y=y_col, color=x_col, title=title, template="plotly_dark")
            
            safe_x = sanitize_filename(x_col)
            safe_y = sanitize_filename(y_col)
            filepath = os.path.join(plot_dir, f'box_{safe_y}_vs_{safe_x}.png')
            fig.write_image(filepath)
            plot_files.append(filepath)
            plots_done += 1

    if not plot_files: return [], "No suitable column pairs found for automated plotting."
    progress(1, desc="Done.")
    return plot_files, f"{len(plot_files)} plots generated for automated EDA."

def run_automl(data, target_variable, numeric_features, categorical_features):
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        try:
            if data is None or target_variable is None:
                yield None, "Error: Data or target not provided.", None, None, None, ""
                return

            yield None, "Processing: Cleaning data...", None, None, None, log_stream.getvalue()
            cleaned_data = data.dropna(subset=[target_variable])
            n_samples = len(cleaned_data)
            fold = min(10, n_samples - 1) if n_samples > 20 else 3
            
            yield None, f"Processing: Setting up PyCaret with {fold} folds...", None, None, None, log_stream.getvalue()
            s = setup(data=cleaned_data, target=target_variable, numeric_features=numeric_features,
                      categorical_features=categorical_features, html=False, verbose=False,
                      polynomial_features=True, session_id=123, fold=fold)
            
            yield None, "Processing: Comparing models...", None, None, None, log_stream.getvalue()
            best_model = compare_models()
            
            yield None, "Processing: Finalizing and generating plots...", None, None, None, log_stream.getvalue()
            results = pull()
            
            plot_files = []
            plots_to_generate = ['residuals', 'feature', 'prediction_error', 'cooks', 'learning']
            for plot_name in plots_to_generate:
                try:
                    plot_model(best_model, plot=plot_name, save=True)
                    plot_files.append(f"{plot_name.replace('_', ' ').title()}.png")
                except Exception as e:
                    print(f"\nCould not generate plot: {plot_name}. Error: {e}")
            final_log = log_stream.getvalue()
            yield results, "AutoML complete.", plot_files, s, best_model, final_log

        except Exception as e:
            yield None, f"An error occurred: {e}", None, None, None, log_stream.getvalue() + f"\nERROR: {e}"

def update_feature_lists(data, target_variable):
    if data is not None and isinstance(data, pd.DataFrame) and target_variable in data.columns:
        numeric_features = [c for c in data.select_dtypes(include=np.number).columns if c != target_variable]
        categorical_features = [c for c in data.select_dtypes(include=['object', 'category']).columns if c != target_variable]
        return gr.update(choices=numeric_features, value=numeric_features), gr.update(choices=categorical_features, value=categorical_features)
    return gr.update(choices=[], value=[]), gr.update(choices=[], value=[])

# --- Main App UI ---
css = """
#main_title h1 { color: #FF7C00; text-align: center; display: block; }
.gradio-container { max-width: 100% !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.orange), css=css) as app:
    shared_data = gr.State()
    pycaret_setup_state = gr.State()
    best_model_state = gr.State()

    gr.Markdown("# Eli Health: LFA Data Analysis Platform", elem_id="main_title")

    with gr.Tabs() as tabs:
        with gr.TabItem("1. Data Upload", id=0):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    file_input = gr.File(label="Upload CSV File")
                    upload_button = gr.Button("Upload and Process", variant="primary")
                    upload_status = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column(scale=3):
                    gr.Markdown("### Data Preview")
                    with gr.Tabs():
                        with gr.TabItem("Processed Data"):
                            data_output_preview = gr.DataFrame(label="Processed & Transformed Data Preview", wrap=True)
                        with gr.TabItem("Raw Data"):
                            raw_data_preview = gr.DataFrame(label="Raw Uploaded Data Preview", wrap=True)

        with gr.TabItem("2. Exploratory Data Analysis (EDA)", id=1):
            with gr.Tabs():
                with gr.TabItem("Automated High-Throughput EDA"):
                    ht_eda_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Row():
                        ht_eda_color_col = gr.Dropdown(label="Color By", choices=[], interactive=True)
                        ht_fit_5pl = gr.Checkbox(label="Fit 5PL Curve (Scatter Plots)", value=False)
                    ht_eda_button = gr.Button("Generate Automated Plots", variant="primary")
                    ht_eda_gallery = gr.Gallery(label="Automated EDA Plots", columns=3, object_fit="contain", height="auto")
                
                with gr.TabItem("Manual Plotting"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=300):
                            gr.Markdown("### Manual Scatter Plot")
                            eda_x_axis = gr.Dropdown(label="Select X-Axis", choices=[], interactive=True)
                            eda_y_axis = gr.Dropdown(label="Select Y-Axis", choices=[], interactive=True)
                            eda_color_col = gr.Dropdown(label="Color By", choices=[], interactive=True)
                            eda_fit_5pl = gr.Checkbox(label="Fit 5PL Curve", value=False)
                            eda_plot_button = gr.Button("Generate Plot", variant="secondary")
                        with gr.Column(scale=2):
                            eda_plot_output = gr.Plot(label="EDA Plot")

        with gr.TabItem("3. AutoML Modeling", id=2):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    target_variable_dropdown = gr.Dropdown(label="Select Target Variable", choices=[], interactive=False)
                    numeric_features_checkbox = gr.CheckboxGroup(label="Numeric Features", interactive=True)
                    categorical_features_checkbox = gr.CheckboxGroup(label="Categorical Features", interactive=True)
                    automl_button = gr.Button("Run AutoML", variant="primary", interactive=False)
                    status_box_automl = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column(scale=2):
                    model_leaderboard = gr.DataFrame(label="Model Leaderboard")
                    automl_log_output = gr.Textbox(label="Live AutoML Log", lines=20, interactive=False, max_lines=20)

        with gr.TabItem("4. Model Analysis", id=3):
            gr.Markdown("### View Generated Model Plots")
            model_plot_gallery = gr.Gallery(label="Model Diagnostic Plots", columns=2, object_fit="contain", height="auto")
            
    # --- Event Handlers ---
    
    upload_button.click(
        fn=upload_and_process_data,
        inputs=file_input,
        outputs=[raw_data_preview, data_output_preview, shared_data, upload_status, eda_x_axis, eda_y_axis, eda_color_col, target_variable_dropdown]
    ).then(
        lambda data: gr.update(choices=["None"] + list(data.columns) if data is not None else []),
        inputs=shared_data,
        outputs=ht_eda_color_col
    )
    
    eda_plot_button.click(fn=generate_manual_eda_plot, inputs=[shared_data, eda_x_axis, eda_y_axis, eda_color_col, eda_fit_5pl], outputs=[eda_plot_output, upload_status])
    ht_eda_button.click(fn=generate_automated_eda, inputs=[shared_data, ht_eda_color_col, ht_fit_5pl], outputs=[ht_eda_gallery, ht_eda_status])

    target_variable_dropdown.change(
        fn=update_feature_lists,
        inputs=[shared_data, target_variable_dropdown],
        outputs=[numeric_features_checkbox, categorical_features_checkbox]
    ).then(lambda: gr.update(interactive=True), None, automl_button)

    automl_button.click(
        fn=run_automl,
        inputs=[shared_data, target_variable_dropdown, numeric_features_checkbox, categorical_features_checkbox],
        outputs=[model_leaderboard, status_box_automl, model_plot_gallery, pycaret_setup_state, best_model_state, automl_log_output]
    )

if __name__ == "__main__":
    app.launch(debug=True)

