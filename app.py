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
import warnings

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")
os.environ['PYCARET_CUSTOM_LOGGING'] = 'CRITICAL'
TEMP_PLOT_DIR = "temp_plots"
MAX_DYNAMIC_TEXTBOXES = 10 

# --- FIX: Definitive list of columns for reliable LUMOS data detection ---
LUMOS_EXPECTED_COLS = [
    'date', 'time', 'strip name', 'run count', 'lot ID', 'assay name',
    'overall result', 'experiment id', 'project id', 'operator',
    'temperature (°C)', 'filename', 'exposure(ms)', 'illumination',
    'crop_offset', 'crop_height', 'reader_serial', 'software version',
    'reprocessed_from_filename', 'reprocessed_from_assay',
    'reprocessed_warning', 'line_name_1', 'line_centre_1',
    'line_peak_above_background_1', 'line_background_1', 'line_area_1',
    'line_polarity_1', 'line_offset_1', 'line_background_offset_1',
    'line_name_2', 'line_centre_2', 'line_peak_above_background_2',
    'line_background_2', 'line_area_2', 'line_polarity_2', 'line_offset_2',
    'line_background_offset_2'
]

def setup_temp_dir():
    """Clears and recreates the temporary directory for storing plot images."""
    if os.path.exists(TEMP_PLOT_DIR):
        shutil.rmtree(TEMP_PLOT_DIR)
    os.makedirs(TEMP_PLOT_DIR)

# --- Core Data & Naming Logic ---

def get_backend_name(original_name):
    """Creates a backend-safe identifier from a user-facing column name."""
    name = str(original_name)
    name = name.replace('/', '_div_').replace('-', '_minus_')
    name = re.sub(r'[^A-Za-z0-9_]+', '_', name)
    return 'col_' + name if name.startswith('_') else name

def create_name_map(columns):
    """Creates a mapping from backend-safe names to original, user-friendly names."""
    return {get_backend_name(col): col for col in columns}

def parse_lumos_data(raw_df, new_col_names):
    """Parses 'overall_result' and splits 'strip name' for LUMOS-specific datasets."""
    if raw_df is None: raise ValueError("No raw data to process.")
    processed_df = raw_df.copy()
    
    if 'strip name' in processed_df.columns and new_col_names:
        valid_new_names = [name for name in new_col_names if name]
        if not valid_new_names: raise ValueError("Please provide names for the new columns.")
        split_data = processed_df['strip name'].str.split('-', expand=True)
        if len(valid_new_names) != len(split_data.columns):
            raise ValueError(f"Name count ({len(valid_new_names)}) mismatches split columns ({len(split_data.columns)}).")
        split_data.columns = valid_new_names
        processed_df = pd.concat([processed_df, split_data], axis=1)
    else:
        valid_new_names = []

    if 'overall_result' in processed_df.columns:
        pattern = re.compile(r"Test Line peak: ([\d.]+).*Control Line peak: ([\d.]+)", re.DOTALL)
        extracted_data = processed_df['overall_result'].str.extract(pattern)
        if extracted_data.shape[1] == 2:
            extracted_data.columns = ['T', 'C']
            processed_df = pd.concat([processed_df.reset_index(drop=True), extracted_data.reset_index(drop=True)], axis=1)
        else:
             raise ValueError("Could not find 'Test Line peak' and 'Control Line peak' in 'overall_result' column.")

    if 'T' in processed_df.columns and 'C' in processed_df.columns:
        for col in ['T', 'C']:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df.dropna(subset=['T', 'C'], inplace=True)
        sum_tc = processed_df['T'] + processed_df['C']
        processed_df['T_norm'] = np.divide(processed_df['T'], sum_tc, where=sum_tc!=0, out=0.0)
        processed_df['C_norm'] = np.divide(processed_df['C'], sum_tc, where=sum_tc!=0, out=0.0)
        processed_df['T-C'] = processed_df['T'] - processed_df['C']
        processed_df['C/T'] = np.divide(processed_df['C'], processed_df['T'], where=processed_df['T']!=0, out=np.inf)
        processed_df['T/C'] = np.divide(processed_df['T'], processed_df['C'], where=processed_df['C']!=0, out=np.inf)
        
        final_cols_to_keep = valid_new_names + ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C']
        final_cols = [col for col in final_cols_to_keep if col in processed_df.columns]
        return processed_df[final_cols]
    else:
        raise ValueError("Could not create 'T' and 'C' columns for calculations.")

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

def upload_data(file, progress=gr.Progress()):
    if file is None: 
        return None, "Please upload a CSV file.", gr.update(visible=False), gr.update(visible=False), *([gr.update(visible=False)]*MAX_DYNAMIC_TEXTBOXES)
    try:
        progress(0.5, desc="Reading file...")
        raw_df = pd.read_csv(file.name)
        raw_df.columns = raw_df.columns.str.strip()
        
        if set(LUMOS_EXPECTED_COLS).issubset(raw_df.columns):
            num_new_cols = len(raw_df['strip name'].iloc[0].split('-'))
            dynamic_updates = [gr.update(label=f"Name for Column {i+1}", visible=True) for i in range(num_new_cols)]
            dynamic_updates += [gr.update(visible=False) for _ in range(num_new_cols, MAX_DYNAMIC_TEXTBOXES)]
            status_update = f"LUMOS data detected. Found {num_new_cols} segments in 'strip name'."
            return raw_df, status_update, gr.update(visible=True), gr.update(visible=False), *dynamic_updates
        else:
            return raw_df, "Standard dataset detected. Press 'Analyze Data' to continue.", gr.update(visible=False), gr.update(visible=True), *([gr.update(visible=False)]*MAX_DYNAMIC_TEXTBOXES)
    except Exception as e:
        return None, f"An error occurred: {e}", gr.update(visible=False), gr.update(visible=False), *([gr.update(visible=False)]*MAX_DYNAMIC_TEXTBOXES)

def update_ui_after_processing(processed_df):
    """A helper function to update all EDA and AutoML UI elements after data is processed."""
    name_map = create_name_map(processed_df.columns)
    all_cols = ["None"] + list(processed_df.columns)
    numeric_cols = list(processed_df.select_dtypes(include=np.number).columns)
    return processed_df, name_map, "Data ready for analysis.", gr.update(choices=numeric_cols, value=numeric_cols, interactive=True), gr.update(choices=numeric_cols, value=numeric_cols, interactive=True), gr.update(choices=all_cols, interactive=True), gr.update(choices=list(processed_df.columns), interactive=True), gr.update(interactive=True), gr.update(choices=numeric_cols, interactive=True), gr.update(choices=numeric_cols, interactive=True), gr.update(choices=all_cols, interactive=True), gr.update(interactive=False)

def process_lumos_data_from_ui(raw_data, *new_names):
    try:
        processed_df = parse_lumos_data(raw_data, new_names)
        return update_ui_after_processing(processed_df)
    except Exception as e:
        num_outputs = 12; error_return = [gr.update()] * num_outputs; error_return[2] = f"Processing Error: {e}"; return tuple(error_return)

def process_standard_data_from_ui(raw_data):
    return update_ui_after_processing(raw_data)

# --- FIX: Restored missing EDA functions ---
def generate_manual_eda_plot(data, x_col, y_col, color_col):
    if data is None: return None, "Upload data to begin EDA."
    if not x_col or not y_col: return go.Figure(), "Please select both X and Y axes."
    title = f'Scatter Plot: {y_col} vs. {x_col}' + (f' (Colored by {color_col})' if color_col != "None" else '')
    fig = px.scatter(data, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=title, template="plotly_dark")
    return fig, f"Displaying: {title}"

def generate_automated_eda(data, x_cols, y_cols, color_col, progress=gr.Progress()):
    if data is None: return [], "Upload data to generate plots.", []
    if not x_cols or not y_cols: return [], "Please select variables for both X and Y axes.", []
    plots = []; plot_combinations = list(product(x_cols, y_cols))
    for i, (x_col, y_col) in enumerate(plot_combinations):
        if x_col == y_col: continue
        progress(i / len(plot_combinations), desc=f"Plotting: {y_col} vs. {x_col}")
        title = f'{y_col} vs. {x_col}' + (f' (Colored by {color_col})' if color_col != "None" else '')
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col if color_col != "None" else None, title=title, template="plotly_dark")
        plots.append(fig)
    if not plots: return [], "No plots generated.", []
    plot_files = save_plots_to_temp_dir(plots)
    return plot_files, f"{len(plots)} plots generated.", plots

def run_automl(data, name_map, target_variable, model_type, progress=gr.Progress()):
    summary_log = ""; blank_df = pd.DataFrame()
    blank_outputs = (blank_df, None, gr.update(visible=False), "", [], summary_log, [], blank_df, blank_df, "", "")
    try:
        if data is None or target_variable is None: raise ValueError("Data or target variable not provided.")
        model_data = data.dropna(subset=[target_variable]).copy()
        if len(model_data) < 10: raise ValueError(f"Insufficient data. After removing missing target rows, only {len(model_data)} rows remain. At least 10 are required.")
        pc = reg if model_type == 'Regression' else clf
        optimize_metric = 'R2' if model_type == 'Regression' else 'Accuracy'
        backend_df = model_data.rename(columns={v: k for k, v in name_map.items()})
        backend_target = get_backend_name(target_variable)
        yield blank_df, None, gr.update(visible=True), "Initializing AutoML...", [], summary_log, [], blank_df, blank_df, "", ""
        n_samples = len(backend_df); num_folds = n_samples if n_samples < 50 else 10
        yield blank_df, None, gr.update(visible=True), f"Setting up PyCaret ({num_folds} folds)...", [], summary_log, [], blank_df, blank_df, "", ""
        s = pc.setup(data=backend_df, target=backend_target, html=False, verbose=False, session_id=123, polynomial_features=False, transformation=True, normalize=True, feature_selection=False, fold=num_folds)
        yield blank_df, None, gr.update(visible=True), "Comparing models...", [], summary_log, [], blank_df, blank_df, "", ""
        best_model = pc.compare_models(sort=optimize_metric)
        if best_model is None: raise ValueError("compare_models() did not find a valid model.")
        leaderboard = pc.pull(); best_model_name = leaderboard.index[0]
        summary_log = f"### AutoML Summary\n- **Task:** {model_type}\n- **Best Model Found:** {leaderboard['Model'][0]} (`{best_model_name}`)"
        
        yield leaderboard, None, gr.update(visible=True), "Creating initial model...", [], summary_log, [], blank_df, blank_df, "", ""
        created_model = pc.create_model(best_model, verbose=False)
        cv_results = pc.pull()

        tuned_model = None; tuned_cv_results = pd.DataFrame()
        try:
            yield leaderboard, None, gr.update(visible=True), f"Tuning best model ({optimize_metric})...", [], summary_log, [], cv_results, blank_df, "", ""
            tuned_model = pc.tune_model(best_model, optimize=optimize_metric, verbose=False)
            tuned_cv_results = pc.pull()
        except Exception as e:
            summary_log += f"\n- **Tuning Status:** Failed. The original model will be used. (Error: {e})"
        
        final_model_obj = best_model; final_cv_results = cv_results
        if tuned_model is not None and not tuned_cv_results.empty:
            summary_log += f"\n- **Tuning Status:** Success!"
            if tuned_cv_results[optimize_metric].mean() > cv_results[optimize_metric].mean():
                final_model_obj = tuned_model; final_cv_results = tuned_cv_results
                summary_log += f"\n  - *Tuned model is better and will be used for final analysis.*"
            else:
                summary_log += f"\n  - *Original model performed better and will be used for final analysis.*"
        else:
             summary_log += "\n- **Action:** Using the **original** model for final analysis."
        
        yield leaderboard, None, gr.update(visible=True), "Finalizing model...", [], summary_log, [], cv_results, tuned_cv_results, "", ""
        final_model = pc.finalize_model(final_model_obj)
        
        kpi_html = create_kpi_dashboard(final_cv_results, pc.get_config('pipeline').steps[-1][1].__class__.__name__, model_type)
        final_model_details = str(final_model)
        pipeline_details = str(pc.get_config('pipeline'))
        
        interactive_plots = []
        plot_names = ['feature'] if model_type == 'Regression' else ['auc', 'confusion_matrix', 'feature']
        if model_type == 'Regression': plot_names.insert(0, 'residuals')

        for plot_name in plot_names:
            try:
                fig = pc.plot_model(final_model, plot=plot_name, save=False, verbose=False)
                if fig: fig.update_layout(template='plotly_dark', title_text=f"{plot_name.replace('_', ' ').title()} Plot"); interactive_plots.append(fig)
            except Exception as e:
                print(f"Could not generate plot: {plot_name}. Error: {e}")
        
        plot_files = save_plots_to_temp_dir(interactive_plots)
        yield leaderboard, kpi_html, gr.update(visible=False), "AutoML complete.", plot_files, summary_log, interactive_plots, cv_results, tuned_cv_results, final_model_details, pipeline_details

    except Exception as e:
        yield blank_df, None, gr.update(visible=False), f"An error occurred: {e}", [], str(e), [], blank_df, blank_df, "", ""


def create_kpi_dashboard(results_df, model_full_name, model_type):
    if results_df.empty: return ""
    metric_col = 'R2' if model_type == 'Regression' else 'Accuracy'
    mean_val = results_df[metric_col].mean(); std_val = results_df[metric_col].std()
    
    metric1_name, metric1_val = metric_col, f"{mean_val:.3f} ± {std_val:.3f}"
    metric2_name, metric2_val = 'MAE' if model_type == 'Regression' else 'AUC', f"{results_df.get('MAE', results_df.get('AUC')).mean():.3f}"
    metric3_name, metric3_val = 'RMSE' if model_type == 'Regression' else 'F1', f"{results_df.get('RMSE', results_df.get('F1')).mean():.3f}"
    
    kpi_style = "padding: 10px; margin: 5px; border-radius: 8px; background-color: #2F2F2F; text-align: center; color: white; flex-grow: 1;"
    return f"""<div style="display: flex; justify-content: space-around; width: 100%; gap: 10px;"><div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">Best Model</h3><p style="margin:0;">{model_full_name}</p></div><div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">{metric1_name}</h3><p style="margin:0;">{metric1_val}</p></div><div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">{metric2_name}</h3><p style="margin:0;">{metric2_val}</p></div><div style="{kpi_style}"><h3 style="margin:0; padding:0; color:#FF7C00;">{metric3_name}</h3><p style="margin:0;">{metric3_val}</p></div></div>"""

# --- Main App UI ---
css = """
#main_container { background-color: #121212; } #control_sidebar { background-color: #1E1E1E; padding: 15px; border-radius: 10px; }
.gradio-container { max-width: 100% !important; } .section_accordion .label-wrap { background-color: #2F2F2F !important; color: white !important; }
#automl_summary .prose { color: white !important; }
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
                upload_status = gr.Textbox(label="Status", interactive=False, lines=2)
            
            with gr.Accordion("2. Data Processing", open=False, elem_classes="section_accordion", visible=False) as processing_accordion:
                with gr.Column(visible=False) as lumos_processing_ui:
                    gr.Markdown("LUMOS data detected. Please name the new columns created from 'strip name'.")
                    dynamic_textboxes_ui = [gr.Textbox(label=f"Name for Column {i+1}", visible=False, interactive=True) for i in range(MAX_DYNAMIC_TEXTBOXES)]
                    lumos_process_button = gr.Button("Process LUMOS Data", variant="primary")
                
                with gr.Column(visible=False) as standard_processing_ui:
                    gr.Markdown("Standard data detected.")
                    standard_process_button = gr.Button("Analyze Data", variant="primary")

            with gr.Accordion("3. EDA Setup", open=False, elem_classes="section_accordion") as eda_accordion:
                with gr.Tabs():
                    with gr.TabItem("High-Throughput", id=0):
                        ht_x_cols = gr.CheckboxGroup(label="Select X-Axis Variables", choices=[], value=[]); ht_y_cols = gr.CheckboxGroup(label="Select Y-Axis Variables", choices=[], value=[]); ht_eda_color_col = gr.Dropdown(label="Color By (Categorical)", choices=[]); ht_eda_button = gr.Button("Generate Automated Plots", variant="secondary")
                    with gr.TabItem("Manual Plot", id=1):
                        manual_x_axis = gr.Dropdown(label="Select X-Axis", choices=[]); manual_y_axis = gr.Dropdown(label="Select Y-Axis", choices=[]); manual_color_col = gr.Dropdown(label="Color By", choices=[]); manual_plot_button = gr.Button("Generate Manual Plot", variant="secondary")

            with gr.Accordion("4. AutoML Setup", open=False, elem_classes="section_accordion") as automl_accordion:
                model_type_selector = gr.Radio(choices=['Regression', 'Classification'], label="Select Analysis Type", value='Regression', interactive=False); target_variable_dropdown = gr.Dropdown(label="Select Target Variable", choices=[], interactive=False); automl_button = gr.Button("Run AutoML", variant="primary", interactive=False)

        with gr.Column(scale=3):
            with gr.Accordion("Data Preview", open=True, elem_classes="section_accordion"):
                with gr.Tabs():
                    with gr.TabItem("Processed Data"): data_output_preview = gr.DataFrame(wrap=True)
                    with gr.TabItem("Raw Data"): raw_data_preview = gr.DataFrame(wrap=True)
            with gr.Accordion("Exploratory Analysis", open=True, elem_classes="section_accordion"):
                interactive_eda_plot = gr.Plot(label="Interactive EDA Plot", visible=False); ht_eda_status = gr.Textbox(label="Status", interactive=False, visible=True); gr.Markdown("Click any plot below to view it interactively above."); ht_eda_gallery = gr.Gallery(label="Automated EDA Plots", columns=3, object_fit="contain", height="auto", type="filepath")
            
            with gr.Accordion("AutoML Modeling", open=True, elem_classes="section_accordion"):
                status_box_automl = gr.Textbox(label="Live Status", interactive=False); status_spinner = gr.HTML('<div class="generating"></div>', visible=False); automl_summary_output = gr.Markdown(elem_id="automl_summary"); kpi_dashboard_html = gr.HTML()
                with gr.Tabs() as automl_results_tabs:
                    with gr.TabItem("1. Model Comparison"): model_leaderboard = gr.DataFrame(label="Leaderboard from compare_models()")
                    with gr.TabItem("2. Best Model CV"): cv_results_df = gr.DataFrame(label="Cross-Validation results for the initial best model")
                    with gr.TabItem("3. Tuned Model CV"): tuned_cv_results_df = gr.DataFrame(label="Cross-Validation results for the tuned model")
                    with gr.TabItem("4. Final Model Details"): final_model_details_df = gr.Textbox(label="Parameters of the final model used for analysis", lines=10)
                    with gr.TabItem("5. Pipeline"): pipeline_df = gr.Textbox(label="Full preprocessing and model pipeline", lines=10)
                    with gr.TabItem("6. Diagnostic Plots"):
                        interactive_model_plot = gr.Plot(label="Interactive Model Plot", visible=False); gr.Markdown("Click any plot below to view it interactively above."); model_plot_gallery = gr.Gallery(label="Generated Model Plots", columns=3, object_fit="contain", height="auto")

    # --- Event Handlers ---
    ui_update_outputs = [processed_data, name_map, upload_status, ht_x_cols, ht_y_cols, ht_eda_color_col, target_variable_dropdown, model_type_selector, manual_x_axis, manual_y_axis, manual_color_col, automl_button]
    
    file_input.upload(fn=upload_data, inputs=[file_input], outputs=[raw_data, upload_status, lumos_processing_ui, standard_processing_ui] + dynamic_textboxes_ui, show_progress="full").then(lambda: gr.update(visible=True), None, processing_accordion).then(lambda df: gr.update(value=df), raw_data, raw_data_preview)
    lumos_process_button.click(fn=process_lumos_data_from_ui, inputs=[raw_data] + dynamic_textboxes_ui, outputs=ui_update_outputs, show_progress="full").then(lambda df: gr.update(value=df), processed_data, data_output_preview)
    standard_process_button.click(fn=process_standard_data_from_ui, inputs=[raw_data], outputs=ui_update_outputs, show_progress="full").then(lambda df: gr.update(value=df), processed_data, data_output_preview)
    manual_plot_button.click(fn=generate_manual_eda_plot, inputs=[processed_data, manual_x_axis, manual_y_axis, manual_color_col], outputs=[interactive_eda_plot, ht_eda_status]).then(lambda: gr.update(visible=True), None, interactive_eda_plot)
    ht_eda_button.click(fn=generate_automated_eda, inputs=[processed_data, ht_x_cols, ht_y_cols, ht_eda_color_col], outputs=[ht_eda_gallery, ht_eda_status, eda_interactive_plots])
    target_variable_dropdown.change(fn=lambda target: gr.update(interactive=target is not None), inputs=target_variable_dropdown, outputs=automl_button)
    automl_button.click(fn=run_automl, inputs=[processed_data, name_map, target_variable_dropdown, model_type_selector], outputs=[model_leaderboard, kpi_dashboard_html, status_spinner, status_box_automl, model_plot_gallery, automl_summary_output, automl_interactive_plots, cv_results_df, tuned_cv_results_df, final_model_details_df, pipeline_df])
    def show_interactive_plot(evt: gr.SelectData, plots_list: list):
        if plots_list and evt.index < len(plots_list): return gr.update(value=plots_list[evt.index], visible=True)
        return gr.update(visible=False)
    ht_eda_gallery.select(fn=show_interactive_plot, inputs=[eda_interactive_plots], outputs=[interactive_eda_plot])
    model_plot_gallery.select(fn=show_interactive_plot, inputs=[automl_interactive_plots], outputs=[interactive_model_plot])

if __name__ == "__main__":
    setup_temp_dir()
    app.launch(debug=True, show_error=True)
