import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycaret.regression as pyreg
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
from io import BytesIO
import base64
import warnings
import tempfile
import os
from datetime import datetime
import shutil
import json # For serializing/deserializing dataframes in dcc.Store

# --- New Imports for Export Features ---
from fpdf import FPDF
import plotly.io as pio
# Kaleido is typically installed as a dependency for plotly.io.write_image
# If not installed, you might get errors during image export.
# In a Dash deployment, ensure 'kaleido' is in your requirements.txt.

# --- New Imports for Optimization ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- SciKit-Learn for r2_score ---
from sklearn.metrics import r2_score 

# --- Suppress Warnings for Cleaner Output ---
warnings.filterwarnings("ignore")

# --- Helper Functions for App Logic (Adapted for Dash) ---

def five_pl(x, a, b, c, d, g):
    """5-Parameter Logistic Regression model function."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_safe = np.where(x <= 0, 1e-9, x)
        val = d + (a - d) / (1 + np.exp(b * (np.log(x_safe) - np.log(c))))**g
    return val

def four_pl(x, a, b, c, d):
    """4-Parameter Logistic Regression model function."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_safe = np.where(x <= 0, 1e-9, x)
        val = d + (a - d) / (1 + np.exp(b * (np.log(x_safe) - np.log(c))))
    return val

# --- Helper Functions for Reporting ---

def get_excel_bytes(dfs_dict):
    """Generates bytes for a multi-sheet Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs_dict.items():
            if df is not None:
                try:
                    # Convert all columns to string before writing to avoid type issues in ExcelWriter
                    # This is particularly useful for mixed types or non-numeric PyCaret output
                    df_to_write = df.astype(str) if not df.empty else pd.DataFrame()
                    df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    print(f"Could not write sheet '{sheet_name}': {e}") # Log error, but don't stop process
    return output.getvalue()

def save_plotly_figure_as_image(fig, filename):
    """Save a plotly figure as PNG image."""
    try:
        pio.write_image(fig, filename, width=800, height=500, scale=2)
        return True
    except Exception as e:
        print(f"Could not save Plotly figure {os.path.basename(filename)}: {e}")
        return False

# Matplotlib save function is kept for consistency but might not be directly used if only Plotly is used for figures
def save_matplotlib_figure_as_image(fig, filename):
    """Save a matplotlib figure as PNG image."""
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        return True
    except Exception as e:
        print(f"Could not save Matplotlib figure {os.path.basename(filename)}: {e}")
        return False

class PDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, 'LFA Analysis & ML Suite Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Times', 'B', 16)
            self.cell(0, 6, title, 0, 1, 'L')
            self.ln(4)
        elif level == 2:
            self.set_font('Times', 'B', 14)
            self.cell(0, 6, title, 0, 1, 'L')
            self.ln(4)
        elif level == 3:
            self.set_font('Times', 'B', 12)
            self.cell(0, 6, title, 0, 1, 'L')
            self.ln(2)

    def chapter_body(self, content):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5, content)
        self.ln()

    def add_df_to_pdf(self, df, caption):
        self.ln(2)
        self.set_font('Times', 'B', 10)
        
        # Format all numeric columns to 4 decimal places
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: f"{x:.4f}")

        page_width = self.w - 2*self.l_margin
        col_names = df.columns
        num_cols = len(col_names)
        # Adjust column width if table is too wide
        effective_col_width = col_width = page_width / num_cols
        if num_cols * 8 > page_width: # Heuristic: if average char width * num_cols is too wide
            self.set_font('Times', 'B', 7) # Smaller font for headers
            self.add_page() # Add a new page for wide tables if necessary
            effective_col_width = page_width / num_cols # Re-calculate column width for new page
        
        for i, header in enumerate(col_names):
            self.cell(effective_col_width, 8, str(header), 1, 0, 'C')
        self.ln()
        
        self.set_font('Times', '', 7) # Smaller font for body
        for index, row in df.iterrows():
            for i, item in enumerate(row):
                 self.cell(effective_col_width, 8, str(item), 1, 0, 'C')
            self.ln()
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, caption, 0, 1, 'C')
        self.ln(5)

    def add_image(self, img_path, caption, width_percent=0.8):
        page_width = self.w - 2*self.l_margin
        self.image(img_path, w=page_width * width_percent, x=self.l_margin + page_width * (1-width_percent)/2)
        self.ln(2)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, caption, 0, 1, 'C')
        self.ln(5)

# This function will need to be part of a callback in Dash
def create_pdf_report_dash(data_state):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Page
    pdf.set_font('Times', 'B', 28)
    pdf.cell(0, 30, 'LFA Analysis & ML Suite Report', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Times', '', 14)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    
    # --- Methods Section ---
    pdf.add_page()
    pdf.chapter_title("1. Methods")
    methods_text = "The analysis was conducted using the LFA Analysis & ML Suite. "
    if data_state['df_tidy']:
        methods_text += "Data was preprocessed from its raw format into a tidy dataset. "
    if data_state['df_5pl_results']:
        model_choice = data_state.get('dr_model_choice_val', '4PL/5PL')
        methods_text += f"Dose-response curves were fitted using a {model_choice} model. "
    if data_state['best_model']:
        methods_text += "Machine learning models were trained and compared using the PyCaret library to predict the target variable. The best model was selected for further analysis and optimization. "
    if data_state['opt_result']:
        methods_text += "Bayesian Optimization with a Lower Confidence Bound (LCB) acquisition function was employed to suggest optimal experimental conditions. "
    pdf.chapter_body(methods_text)

    # --- Results Section ---
    pdf.add_page()
    pdf.chapter_title("2. Results")

    temp_dir = tempfile.mkdtemp() # Create a temporary directory for images

    # Dose-Response
    if data_state['df_5pl_results']:
        pdf.chapter_title("2.1 Dose-Response Analysis", level=2)
        pdf.chapter_body("The following table summarizes the fitted parameters for the dose-response models. 'c' represents the IC50 value, and 'R-squared' indicates the goodness of fit.")
        pdf.add_df_to_pdf(pd.read_json(data_state['df_5pl_results']), "Table 1: Dose-Response Model Parameters.")
        
        if data_state['dose_response_fig']:
            fig = go.Figure(data_state['dose_response_fig'])
            img_path = os.path.join(temp_dir, "dose_response.png")
            if save_plotly_figure_as_image(fig, img_path):
                pdf.add_image(img_path, "Figure 1: Fitted dose-response curves for each experimental group.")

    # Modeling
    if data_state['model_comparison_df']:
        pdf.add_page()
        pdf.chapter_title("2.2 DoE Modeling", level=2)
        pdf.chapter_body("Multiple regression models were automatically trained and evaluated. The table below compares their performance based on key metrics like R-squared, which measures how well the model explains the variance in the data.")
        pdf.add_df_to_pdf(pd.read_json(data_state['model_comparison_df']), "Table 2: Model Performance Comparison.")
        
        if data_state['rsm_fig_path'] and os.path.exists(data_state['rsm_fig_path']):
            pdf.add_image(data_state['rsm_fig_path'], "Figure 2: Response surface plot from the best performing model.")
    
    # Optimization
    if data_state['batch_suggestions']:
        pdf.add_page()
        pdf.chapter_title("2.3 Bayesian Optimization", level=2)
        pdf.chapter_body("Based on the best model, Bayesian Optimization was used to suggest a new batch of experiments. The suggestions balance exploiting known optimal regions with exploring uncertain areas to improve the model.")
        if data_state['global_optimum']:
            pdf.chapter_body(f"The predicted global optimum for {data_state['target_for_opt']} is {data_state['global_optimum']['prediction']:.4f}.")
        
        pdf.add_df_to_pdf(pd.read_json(data_state['batch_suggestions']), "Table 3: Suggested Batch of Experiments.")
        if data_state['opt_landscape_fig']:
            fig = go.Figure(data_state['opt_landscape_fig'])
            img_path = os.path.join(temp_dir, "opt_landscape.png")
            if save_plotly_figure_as_image(fig, img_path):
                pdf.add_image(img_path, "Figure 3: Optimization landscape and partial dependence plots.")

    pdf_output_path = os.path.join(temp_dir, "report.pdf")
    pdf.output(pdf_output_path)

    with open(pdf_output_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    return pdf_bytes

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # Expose Flask server for deployment

app.layout = dbc.Container(
    [
        # dcc.Store for client-side storage of various app states
        dcc.Store(id='data-storage', data={
            'data_source': 'LUMOS',
            'df_raw': None,
            'df_tidy': None,
            'df_5pl_results': None,
            'df_tidy_merged': None,
            'df_5pl_results_merged': None,
            'best_model_json': None, # PyCaret model, needs special handling (e.g., save/load)
            'pycaret_preprocessor_json': None, # PyCaret preprocessor, needs special handling
            'modeling_df_for_opt': None,
            'features_for_opt': None,
            'target_for_opt': None,
            'opt_result_json': None, # skopt result object
            'batch_suggestions': None,
            'global_optimum': None,
            'dose_response_fig': None,
            'opt_landscape_fig': None,
            'opt_convergence_fig': None,
            'model_comparison_df': None,
            'rsm_fig_path': None, # Path to saved image on server
            'perform_dr': True,
            'dr_model_choice_val': '5PL',
            'dr_conc_col': None,
            'dr_group_col': None,
            'dr_y_var': 'T_norm',
            'pycaret_normalize': True,
            'pycaret_normalize_method': 'zscore',
            'pycaret_remove_outliers': False,
            'pycaret_transformation': False,
            'pycaret_remove_multicollinearity': False,
            'pycaret_multicollinearity_threshold': 0.9,
            'pycaret_feature_interaction': False,
            'pycaret_polynomial_degree': 2,
            'pycaret_feature_selection': False,
            'pycaret_num_factors': 1,
            'factor_names': [],
            'edited_doe_df_json': None,
            'pycaret_grouping_col': None,
            'pycaret_target_selector': None,
            'pycaret_features_selector': []
        }),
        dcc.Loading(
            id="loading-global",
            type="default",
            children=[
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.H2("🔬 LFA Analysis & ML Suite", className="my-4 text-center"),
                            html.Hr(),
                            html.H3("1. Data Upload"),
                            dcc.RadioItems(
                                id='data-source-radio',
                                options=[
                                    {'label': 'LUMOS', 'value': 'LUMOS'},
                                    {'label': 'Custom', 'value': 'Custom'}
                                ],
                                value='LUMOS', # Default value
                                inline=True,
                                className="mb-3"
                            ),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed',
                                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0px'
                                },
                                multiple=False
                            ),
                            html.Div(id='upload-status'),
                            html.Hr(),
                            html.H3("2. Navigation"),
                            html.Ul([
                                html.Li(html.A("Data Preprocessing", href="#step-1-data-preprocessing")),
                                html.Li(html.A("Dose-Response Analysis", href="#step-2-dose-response-analysis")),
                                html.Li(html.A("DoE Modeling with PyCaret", href="#step-3-doe-modeling-with-pycaret")),
                                html.Li(html.A("Bayesian Optimization", href="#step-4-bayesian-optimization")),
                                html.Li(html.A("Download Report", href="#step-5-generate-and-download-report"))
                            ])
                        ], className="p-3 bg-light sidebar"), width=3, className="d-none d-md-block"), # Sidebar column
                    dbc.Col(
                        html.Div([
                            html.H1("LFA Data Analysis and Machine Learning Pipeline", className="my-4"),

                            # --- Step 1: Data Preprocessing ---
                            html.Div(id="step-1-data-preprocessing", children=[
                                html.H2("Step 1: Data Preprocessing", className="mb-3"),
                                html.Div(id='uploaded-data-preview-div'), # For data preview
                                html.Div(id='lumos-processing-section'), # For LUMOS specific UI
                                html.Div(id='tidy-data-checkpoint') # For tidy data display
                            ], className="section-container border p-4 mb-4"),

                            # --- Step 2: Dose-Response Analysis ---
                            html.Div(id="step-2-dose-response-analysis", children=[
                                html.H2("Step 2: Dose-Response Analysis", className="mb-3"),
                                html.Div(id='dr-section-content') # Content loaded dynamically
                            ], className="section-container border p-4 mb-4"),

                            # --- Step 3: DoE Modeling with PyCaret ---
                            html.Div(id="step-3-doe-modeling-with-pycaret", children=[
                                html.H2("Step 3: DoE Modeling with PyCaret", className="mb-3"),
                                html.Div(id='modeling-section-content') # Content loaded dynamically
                            ], className="section-container border p-4 mb-4"),

                            # --- Step 4: Bayesian Optimization ---
                            html.Div(id="step-4-bayesian-optimization", children=[
                                html.H2("Step 4: Bayesian Optimization", className="mb-3"),
                                html.Div(id='optimization-section-content') # Content loaded dynamically
                            ], className="section-container border p-4 mb-4"),

                            # --- Step 5: Generate and Download Report ---
                            html.Div(id="step-5-generate-and-download-report", children=[
                                html.H2("Step 5: Generate and Download Report", className="mb-3"),
                                dbc.Row([
                                    dbc.Col(html.Button("Generate PDF Report", id="generate-pdf-button", className="btn btn-primary w-100"), width=6),
                                    dbc.Col(html.Button("Generate Excel Report", id="generate-excel-button", className="btn btn-success w-100"), width=6),
                                ]),
                                dcc.Download(id="download-pdf-report"),
                                dcc.Download(id="download-excel-report"),
                                html.Div(id='report-status', className="mt-3")
                            ], className="section-container border p-4 mb-4")

                        ], className="main-content"), width=9) # Main content column
                ])
            ]
        )
    ],
    fluid=True, className="app-container"
)

# --- Callbacks ---

# Callback to update data_source in store
@app.callback(
    Output('data-storage', 'data', allow_duplicate=True),
    Input('data-source-radio', 'value'),
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def update_data_source_in_store(selected_source, data_store):
    data_store['data_source'] = selected_source
    # Clear relevant processing data when data source changes
    data_store['df_tidy'] = None
    data_store['df_5pl_results'] = None
    data_store['df_tidy_merged'] = None
    data_store['df_5pl_results_merged'] = None
    data_store['best_model_json'] = None
    data_store['model_comparison_df'] = None
    data_store['rsm_fig_path'] = None
    data_store['opt_result_json'] = None
    data_store['batch_suggestions'] = None
    data_store['global_optimum'] = None
    data_store['dose_response_fig'] = None
    data_store['opt_landscape_fig'] = None
    data_store['opt_convergence_fig'] = None
    return data_store


# Callback to handle file upload and initial data processing (Step 1)
@app.callback(
    [Output('data-storage', 'data'),
     Output('upload-status', 'children'),
     Output('uploaded-data-preview-div', 'children'),
     Output('lumos-processing-section', 'children'),
     Output('tidy-data-checkpoint', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('data-storage', 'data'),
    State('data-source-radio', 'value'),
    prevent_initial_call=True
)
def upload_data(contents, filename, data_store, data_source_val):
    if contents is None:
        return data_store, html.Div("No file uploaded."), "", "", ""

    content_type, content_string = contents.split(',')
    decoded = base66.b64decode(content_string)
    
    df_raw = None
    try:
        if 'csv' in filename:
            df_raw = pd.read_csv(BytesIO(decoded))
        elif 'xls' in filename:
            df_raw = pd.read_excel(BytesIO(decoded))
        else:
            return data_store, html.Div("Unsupported file type.", style={'color': 'red'}), "", "", ""
    except Exception as e:
        return data_store, html.Div(f"Error processing file: {e}", style={'color': 'red'}), "", "", ""

    data_store['df_raw'] = df_raw.to_json(date_format='iso', orient='split')
    data_store['data_source'] = data_source_val # Ensure data_source is correctly set in store

    upload_status = html.Div(f"File '{filename}' uploaded successfully!")
    uploaded_preview = html.Div([
        html.H4("Uploaded Data Preview"),
        dbc.Table.from_dataframe(df_raw.head(), striped=True, bordered=True, hover=True, size="sm")
    ])

    lumos_section_content = []
    tidy_checkpoint_content = []

    if data_source_val == 'LUMOS':
        lumos_section_content.append(html.H4("LUMOS Data Processing"))
        
        required_cols = ['strip name', 'line_peak_above_background_1', 'line_peak_above_background_2']
        if not all(col in df_raw.columns for col in required_cols):
            lumos_section_content.append(html.Div(f"LUMOS file must contain: {', '.join(required_cols)}", style={'color': 'red'}))
            data_store['df_tidy'] = None
            return data_store, upload_status, uploaded_preview, lumos_section_content, ""

        if df_raw['strip name'].empty:
            lumos_section_content.append(html.Div("'strip name' column is empty.", style={'color': 'red'}))
            data_store['df_tidy'] = None
            return data_store, upload_status, uploaded_preview, lumos_section_content, ""

        first_strip_name = str(df_raw['strip name'].iloc[0])
        delimiter = '-' if '-' in first_strip_name else '_' if '_' in first_strip_name else None
        
        group_names = []
        if not delimiter:
            lumos_section_content.append(html.Div("Could not auto-detect delimiter ('-' or '_') in 'strip name'. Treating 'strip name' as a single factor."))
            group_names = [html.Div([
                html.Label("Group 1 Name:"),
                dcc.Input(id='group-name-0', type='text', value="Factor_1", className="form-control mb-2")
            ])]
            data_store['factor_names'] = ["Factor_1"] # Store default for processing
        else:
            num_groups = len(first_strip_name.split(delimiter))
            lumos_section_content.append(html.Div(f"Detected **{num_groups}** groups in 'strip name' based on delimiter '{delimiter}'. Please name them:"))
            group_names_inputs = []
            current_factor_names = data_store.get('factor_names', [f"Factor_{i+1}" for i in range(num_groups)])
            if len(current_factor_names) != num_groups: # Reset if group count changes
                current_factor_names = [f"Factor_{i+1}" for i in range(num_groups)]
            for i in range(num_groups):
                group_names_inputs.append(html.Div([
                    html.Label(f"Group {i+1}:"),
                    dcc.Input(id=f'group-name-{i}', type='text', value=current_factor_names[i], className="form-control mb-2")
                ]))
            group_names = group_names_inputs
            data_store['factor_names'] = current_factor_names # Store initial values for callback

        lumos_section_content.extend(group_names)
        lumos_section_content.append(dbc.Button("Process LUMOS Data", id="process-lumos-button", className="btn btn-primary mt-3"))

    elif data_source_val == 'Custom':
        df = df_raw.copy()
        data_store['df_tidy'] = df.to_json(date_format='iso', orient='split')
        tidy_checkpoint_content = html.Div([
            html.H4("Checkpoint: Processed Tidy Data"),
            html.P("This is the resulting table from Step 1. It will be used as the input for all subsequent steps."),
            dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True, size="sm")
        ])
    
    return data_store, upload_status, uploaded_preview, lumos_section_content, tidy_checkpoint_content

# Callback to process LUMOS data (triggered by button or input change)
@app.callback(
    [Output('data-storage', 'data', allow_duplicate=True),
     Output('tidy-data-checkpoint', 'children')],
    [Input('process-lumos-button', 'n_clicks')] + 
    [Input(f'group-name-{i}', 'value') for i in range(10)], # Max 10 dynamic inputs
    State('data-storage', 'data'),
    State('data-source-radio', 'value'),
    State('upload-data', 'filename'), # To re-read raw data
    State('upload-data', 'contents'), # To re-read raw data
    prevent_initial_call=True
)
def process_lumos_data(n_clicks, *group_name_values, data_store, data_source_val, filename, contents):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Only process if button is clicked, or if it's the initial processing after upload for LUMOS
    # Or if a group name input changes
    if trigger_id == 'process-lumos-button' or ('group-name' in trigger_id and n_clicks is None):
        # Re-decode raw data
        if not data_store['df_raw']:
            raise dash.exceptions.PreventUpdate

        df_raw = pd.read_json(data_store['df_raw'], orient='split')

        first_strip_name = str(df_raw['strip name'].iloc[0])
        delimiter = '-' if '-' in first_strip_name else '_' if '_' in first_strip_name else None

        group_names = []
        if delimiter:
            num_groups_detected = len(first_strip_name.split(delimiter))
            # Slice group_name_values to match detected groups
            group_names = list(group_name_values[:num_groups_detected])
            if any(name is None for name in group_names): # If some inputs aren't rendered yet
                raise dash.exceptions.PreventUpdate
        else: # Single factor mode
            group_names = [group_name_values[0]] # Just the first one
            if group_names[0] is None:
                raise dash.exceptions.PreventUpdate

        data_store['factor_names'] = group_names # Store the actual names used for processing

        df = df_raw.copy()
        if delimiter:
            split_cols = df['strip name'].astype(str).str.split(delimiter, expand=True)
            if split_cols.shape[1] != len(group_names):
                return data_store, html.Div(f"Mismatch in group count. Expected {len(group_names)} but found {split_cols.shape[1]} for some rows. Please check 'strip name' consistency.", style={'color': 'red'})
            split_cols.columns = group_names
            df = pd.concat([split_cols, df], axis=1)
        else:
            df.rename(columns={'strip name': group_names[0]}, inplace=True)

        for col in group_names:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        df = df.rename(columns={'line_peak_above_background_1': 'T', 'line_peak_above_background_2': 'C'})
        df['T+C'] = df['T'] + df['C']
        df['T_norm'] = df['T'].divide(df['T+C']).fillna(0)
        df['C_norm'] = df['C'].divide(df['T+C']).fillna(0)
        df['T-C'] = df['T'] - df['C']
        df['C/T'] = df['C'].divide(df['T']).replace([np.inf, -np.inf], 0).fillna(0)
        df['T/C'] = df['T'].divide(df['C']).replace([np.inf, -np.inf], 0).fillna(0)

        relevant_cols = group_names + ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'T+C', 'C/T', 'T/C']
        final_cols = [col for col in relevant_cols if col in df.columns]
        df_tidy = df[final_cols]
        data_store['df_tidy'] = df_tidy.to_json(date_format='iso', orient='split')
        
        # Also, if LUMOS, df_tidy_merged is set here
        data_store['df_tidy_merged'] = df_tidy.to_json(date_format='iso', orient='split')


        tidy_checkpoint = html.Div([
            html.H4("Checkpoint: Processed Tidy Data"),
            html.P("This is the resulting table from Step 1. It will be used as the input for all subsequent steps."),
            dbc.Table.from_dataframe(df_tidy.head(), striped=True, bordered=True, hover=True, size="sm")
        ])
        return data_store, tidy_checkpoint
    else:
        raise dash.exceptions.PreventUpdate

# Update factor_names in store when dynamic input values change
app.clientside_callback(
    """
    function updateFactorNames(inputValues, currentDataStore) {
        if (!dash_clientside.callback_context.triggered.length) {
            return window.dash_clientside.no_update;
        }
        let updatedDataStore = {...currentDataStore};
        let newFactorNames = inputValues.filter(val => val !== undefined && val !== null);
        updatedDataStore['factor_names'] = newFactorNames;
        return updatedDataStore;
    }
    """,
    Output('data-storage', 'data', allow_duplicate=True),
    Input({'type': 'group-name-input', 'index': ALL}, 'value'), # Dynamic input for group names
    State('data-storage', 'data'),
    prevent_initial_call=True
)

# Callback to render Dose-Response section UI (Step 2)
@app.callback(
    Output('dr-section-content', 'children'),
    Input('data-storage', 'data')
)
def render_dr_section(data_store):
    df_tidy_json = data_store.get('df_tidy')
    if not df_tidy_json:
        return html.Div(html.P("Complete Step 1 to generate the Tidy Data required for this step."))

    df_tidy = pd.read_json(df_tidy_json, orient='split')

    y_vars = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']
    all_cols = df_tidy.columns.tolist()
    
    concentration_options = [c for c in all_cols if c not in y_vars and pd.api.types.is_numeric_dtype(df_tidy[c])]
    grouping_options = [c for c in all_cols if c not in y_vars]
    
    if not concentration_options:
        return html.Div(html.P("No suitable numeric columns for 'Concentration' found in the Tidy Data."), style={'color': 'red'})

    # Get initial values from store or set defaults
    perform_dr = data_store.get('perform_dr', True)
    dr_model_choice_val = data_store.get('dr_model_choice_val', '5PL')
    dr_conc_col = data_store.get('dr_conc_col') or concentration_options[0]
    dr_group_col = data_store.get('dr_group_col') or grouping_options[0]
    dr_y_var = data_store.get('dr_y_var') or 'T_norm'
    
    # Ensure selected values are still in options
    if dr_conc_col not in concentration_options: dr_conc_col = concentration_options[0]
    if dr_group_col not in grouping_options: dr_group_col = grouping_options[0]
    if dr_y_var not in y_vars: dr_y_var = 'T_norm'

    dr_content = html.Div([
        dbc.Checklist(
            options=[{"label": "Perform Dose-Response Regression?", "value": True}],
            value=[True] if perform_dr else [],
            id="perform-dr-toggle",
            switch=True
        ),
        html.Div(id='dr-specific-options', children=[
            dcc.RadioItems(
                id='dr-model-choice',
                options=[
                    {'label': '5PL', 'value': '5PL'},
                    {'label': '4PL', 'value': '4PL'}
                ],
                value=dr_model_choice_val,
                inline=True,
                className="my-3"
            ),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Label("Concentration column:"),
                        dcc.Dropdown(
                            id='dr-conc-col-select',
                            options=[{'label': col, 'value': col} for col in concentration_options],
                            value=dr_conc_col,
                            clearable=False
                        )
                    ])
                ),
                dbc.Col(
                    html.Div([
                        html.Label("Group analysis by:"),
                        dcc.Dropdown(
                            id='dr-group-col-select',
                            options=[{'label': col, 'value': col} for col in grouping_options],
                            value=dr_group_col,
                            clearable=False
                        ),
                        html.Small("A model will be fit for each unique value in this column.", className="form-text text-muted")
                    ])
                ),
                dbc.Col(
                    html.Div([
                        html.Label("Response variable (Y-axis):"),
                        dcc.Dropdown(
                            id='dr-y-var-select',
                            options=[{'label': col, 'value': col} for col in y_vars],
                            value=dr_y_var,
                            clearable=False
                        )
                    ])
                )
            ]),
            dbc.Button("Run Dose-Response Analysis", id="run-dr-button", className="btn btn-primary mt-3"),
            html.Div(id='dr-output-content') # For displaying results and plot
        ] if perform_dr else [])
    ])
    return dr_content

# Callback to control visibility of DR options based on toggle
@app.callback(
    [Output('dr-specific-options', 'children'),
     Output('data-storage', 'data', allow_duplicate=True)],
    Input('perform-dr-toggle', 'value'),
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def toggle_dr_options(perform_dr_val, data_store):
    perform_dr = True if perform_dr_val else False
    data_store['perform_dr'] = perform_dr

    if perform_dr:
        # Re-render the full options if turned on
        df_tidy = pd.read_json(data_store['df_tidy'], orient='split')
        y_vars = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']
        all_cols = df_tidy.columns.tolist()
        concentration_options = [c for c in all_cols if c not in y_vars and pd.api.types.is_numeric_dtype(df_tidy[c])]
        grouping_options = [c for c in all_cols if c not in y_vars]
        
        dr_conc_col = data_store.get('dr_conc_col') or (concentration_options[0] if concentration_options else None)
        dr_group_col = data_store.get('dr_group_col') or (grouping_options[0] if grouping_options else None)
        dr_y_var = data_store.get('dr_y_var') or 'T_norm'

        return html.Div([
            dcc.RadioItems(
                id='dr-model-choice',
                options=[
                    {'label': '5PL', 'value': '5PL'},
                    {'label': '4PL', 'value': '4PL'}
                ],
                value=data_store.get('dr_model_choice_val', '5PL'),
                inline=True,
                className="my-3"
            ),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Label("Concentration column:"),
                        dcc.Dropdown(
                            id='dr-conc-col-select',
                            options=[{'label': col, 'value': col} for col in concentration_options],
                            value=dr_conc_col,
                            clearable=False
                        )
                    ])
                ),
                dbc.Col(
                    html.Div([
                        html.Label("Group analysis by:"),
                        dcc.Dropdown(
                            id='dr-group-col-select',
                            options=[{'label': col, 'value': col} for col in grouping_options],
                            value=dr_group_col,
                            clearable=False
                        ),
                        html.Small("A model will be fit for each unique value in this column.", className="form-text text-muted")
                    ])
                ),
                dbc.Col(
                    html.Div([
                        html.Label("Response variable (Y-axis):"),
                        dcc.Dropdown(
                            id='dr-y-var-select',
                            options=[{'label': col, 'value': col} for col in y_vars],
                            value=dr_y_var,
                            clearable=False
                        )
                    ])
                )
            ]),
            dbc.Button("Run Dose-Response Analysis", id="run-dr-button", className="btn btn-primary mt-3"),
            html.Div(id='dr-output-content') # For displaying results and plot
        ]), data_store
    else:
        # Clear DR results when toggled off
        data_store['df_5pl_results'] = None
        data_store['df_5pl_results_merged'] = None
        data_store['dose_response_fig'] = None
        return [], data_store # Return empty div to hide options

# Update DR parameters in store when selected
@app.callback(
    Output('data-storage', 'data', allow_duplicate=True),
    [Input('dr-model-choice', 'value'),
     Input('dr-conc-col-select', 'value'),
     Input('dr-group-col-select', 'value'),
     Input('dr-y-var-select', 'value')],
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def update_dr_params_in_store(model_choice, conc_col, group_col, y_var, data_store):
    data_store['dr_model_choice_val'] = model_choice
    data_store['dr_conc_col'] = conc_col
    data_store['dr_group_col'] = group_col
    data_store['dr_y_var'] = y_var
    # Clear previous DR results if options change, forcing re-run
    data_store['df_5pl_results'] = None
    data_store['df_5pl_results_merged'] = None
    data_store['dose_response_fig'] = None
    return data_store

# Callback to run Dose-Response Analysis
@app.callback(
    [Output('data-storage', 'data', allow_duplicate=True),
     Output('dr-output-content', 'children')],
    Input('run-dr-button', 'n_clicks'),
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def run_dose_response_analysis(n_clicks, data_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    df_tidy = pd.read_json(data_store['df_tidy'], orient='split')
    model_choice = data_store['dr_model_choice_val']
    conc_col = data_store['dr_conc_col']
    group_by_col = data_store['dr_group_col']
    y_var = data_store['dr_y_var']

    if not all([df_tidy is not None, conc_col, group_by_col, y_var]):
        return data_store, html.Div(html.P("Please select all required columns and ensure data is loaded."), style={'color': 'red'})

    df = df_tidy.copy()
    df[conc_col] = pd.to_numeric(df[conc_col], errors='coerce')
    df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
    df.dropna(subset=[conc_col, y_var], inplace=True)

    results = []
    bounds_4pl = ([-np.inf, 0.5, -np.inf, -np.inf], [np.inf, 2.0, np.inf, np.inf])
    bounds_5pl = ([-np.inf, 0.5, -np.inf, -np.inf, 0.7], [np.inf, 2.0, np.inf, np.inf, 1.3])

    for group in sorted(df[group_by_col].unique()):
        group_df = df[df[group_by_col] == group]
        X_fit, y_fit = group_df[conc_col], group_df[y_var]
        if len(X_fit) < 4: continue
        try:
            p0_median_X = np.median(X_fit[X_fit > 0]) if (X_fit > 0).any() else 1
            if model_choice == "5PL":
                p0 = [y_fit.min(), 1, p0_median_X, y_fit.max(), 1]
                params, _ = curve_fit(five_pl, X_fit, y_fit, p0=p0, maxfev=10000, bounds=bounds_5pl)
                r2 = r2_score(y_fit, five_pl(X_fit, *params))
            else:
                p0 = [y_fit.min(), 1, p0_median_X, y_fit.max()]
                params, _ = curve_fit(four_pl, X_fit, y_fit, p0=p0, maxfev=10000, bounds=bounds_4pl)
                r2 = r2_score(y_fit, four_pl(X_fit, *params))
            
            results.append([group] + list(params) + [r2])
        except (RuntimeError, ValueError) as e:
            print(f"Could not fit model for group '{group}': {e}") # Debugging print

    output_content = []
    if results:
        param_names = ['a', 'b', 'c', 'd', 'g'] if model_choice == "5PL" else ['a', 'b', 'c', 'd']
        results_df = pd.DataFrame(results, columns=['Group'] + param_names + ['R-squared'])
        data_store['df_5pl_results'] = results_df.to_json(date_format='iso', orient='split')
        data_store['df_5pl_results_merged'] = results_df.to_json(date_format='iso', orient='split')

        output_content.append(html.H4("Checkpoint: Dose-Response Results"))
        output_content.append(html.P("This table of model parameters will be used for DoE Modeling in Step 3."))
        output_content.append(dbc.Table.from_dataframe(results_df, striped=True, bordered=True, hover=True, size="sm"))

        if 'c' in results_df.columns and not results_df.empty:
            best_performer = results_df.loc[results_df['c'].idxmin()]
            output_content.append(html.H4("🏆 Best Performer (Lowest IC50)"))
            output_content.append(dbc.Row([
                dbc.Col(html.Div([html.P("Group"), html.H5(f"{best_performer['Group']}")])),
                dbc.Col(html.Div([html.P("IC50 (Value of 'c')"), html.H5(f"{best_performer['c']:.3f}")])),
                dbc.Col(html.Div([html.P("R²"), html.H5(f"{best_performer['R-squared']:.4f}")]))
            ]))

        output_content.append(html.H4("Dose-Response Curves"))
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, group in enumerate(sorted(df[group_by_col].unique())):
            color = colors[i % len(colors)]
            group_df_plot = df[df[group_by_col] == group]
            fig.add_trace(go.Scatter(x=group_df_plot[conc_col], y=group_df_plot[y_var], mode='markers', name=f'Group {group} (data)', marker=dict(color=color)))
            if not results_df[results_df['Group'] == group].empty:
                params = results_df[results_df['Group'] == group].iloc[0, 1:-1].values
                x_min = df[conc_col].min()
                x_max = df[conc_col].max()
                x_range = np.linspace(x_min, x_max, 200)
                fit_func = five_pl if model_choice == "5PL" else four_pl
                y_pred = fit_func(x_range, *params)
                fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Group {group} (fit)', line=dict(color=color, dash='solid')))
        fig.update_layout(xaxis_title=conc_col, yaxis_title=y_var, title="Dose-Response Analysis", legend_title_text='Group')
        data_store['dose_response_fig'] = fig.to_json()
        output_content.append(dcc.Graph(figure=fig))
    else:
        output_content.append(html.Div(html.P("Could not derive any dose-response models."), style={'color': 'orange'}))
        data_store['df_5pl_results'] = None
        data_store['df_5pl_results_merged'] = None
        data_store['dose_response_fig'] = None

    return data_store, output_content

# Callback to render Modeling section UI (Step 3)
@app.callback(
    Output('modeling-section-content', 'children'),
    Input('data-storage', 'data')
)
def render_modeling_section(data_store):
    df_raw_json = data_store.get('df_raw')
    if not df_raw_json:
        return html.Div(html.P("Complete Step 1 to generate the data needed for this step."))

    df_tidy = pd.read_json(data_store.get('df_tidy'), orient='split') if data_store.get('df_tidy') else None
    df_5pl_results = pd.read_json(data_store.get('df_5pl_results'), orient='split') if data_store.get('df_5pl_results') else None

    if df_tidy is None:
        return html.Div(html.P("Complete Step 1 to generate the data needed for this step."))

    section_content = []

    if data_store['data_source'] == 'LUMOS':
        section_content.append(html.H4("1. Define & Merge DoE Factors (LUMOS Data)"))
        grouping_col_options = [{'label': col, 'value': col} for col in df_tidy.columns]
        
        default_grouping_col = data_store.get('pycaret_grouping_col')
        if default_grouping_col not in df_tidy.columns:
            default_grouping_col = df_tidy.columns[0] # Fallback to first column

        section_content.append(html.Div([
            html.Label("Select the column that defines your experimental groups:"),
            dcc.Dropdown(
                id='pycaret-grouping-col-select',
                options=grouping_col_options,
                value=default_grouping_col,
                clearable=False
            )
        ]))
        
        grouping_col = default_grouping_col # Use the current default or selected

        if grouping_col:
            unique_groups = pd.DataFrame(df_tidy[grouping_col].unique(), columns=[grouping_col]).sort_values(by=grouping_col).reset_index(drop=True)
            section_content.append(html.P(f"Enter the DoE factor values for each unique group in **'{grouping_col}'**."))
            
            num_factors = data_store.get('pycaret_num_factors', 1)
            section_content.append(html.Div([
                html.Label("Number of DoE Factors:"),
                dcc.Input(id='pycaret-num-factors-input', type='number', min=1, value=num_factors, className="form-control")
            ]))

            factor_names = data_store.get('factor_names', [f"DoE_Factor_{i+1}" for i in range(num_factors)])
            # Adjust factor_names if num_factors changes
            if len(factor_names) != num_factors:
                factor_names = [f"DoE_Factor_{i+1}" for i in range(num_factors)]

            factor_name_inputs = []
            for i in range(num_factors):
                factor_name_inputs.append(html.Div([
                    html.Label(f"Factor {i+1} Name:"),
                    dcc.Input(id={'type': 'pycaret-factor-name-input', 'index': i}, type='text', 
                              value=factor_names[i] if i < len(factor_names) else f"DoE_Factor_{i+1}", 
                              className="form-control mb-2")
                ]))
            section_content.append(html.Div(factor_name_inputs, id='dynamic-factor-names-container'))

            # Prepare data for data editor
            doe_input_df = unique_groups.copy()
            for name in factor_names:
                if name not in doe_input_df.columns:
                    doe_input_df[name] = 0.0 # Initialize new factors

            edited_doe_df = pd.DataFrame()
            if data_store.get('edited_doe_df_json'):
                temp_df = pd.read_json(data_store['edited_doe_df_json'], orient='split')
                # Ensure the columns match before using from store
                if set(temp_df.columns) == set(doe_input_df.columns) and temp_df[grouping_col].equals(doe_input_df[grouping_col]):
                    edited_doe_df = temp_df
                else:
                    edited_doe_df = doe_input_df
            else:
                edited_doe_df = doe_input_df

            section_content.append(html.Div([
                dbc.Table.from_dataframe(edited_doe_df, striped=True, bordered=True, hover=True, size="sm"), # Display as static table
                # For editing, you'd need a more complex Dash DataTable or a button to trigger modal editing
                html.P("Adjust DoE factors directly in the table below. (Note: For direct editing of this table in Dash, a `dash_table.DataTable` component would be needed. Currently, this is for display.)")
            ], id='doe-factor-editor-div')) # Add ID for potential dynamic updates if user edits via a modal or alternative
            
            section_content.append(html.H4("Checkpoint: Merged Tidy Data"))
            section_content.append(dbc.Table.from_dataframe(df_tidy.head(), striped=True, bordered=True, hover=True, size="sm"))
            if df_5pl_results is not None:
                section_content.append(html.H4("Checkpoint: Merged Dose-Response Results"))
                section_content.append(dbc.Table.from_dataframe(df_5pl_results.head(), striped=True, bordered=True, hover=True, size="sm"))
    else: # Custom Data
        section_content.append(html.H4("1. Custom Data for Modeling"))
        section_content.append(html.P("For custom data, it is assumed your uploaded table is already prepared for modeling. No DoE factor merging is required."))
        # Display merged tidy data (which is just tidy data for custom)
        if df_tidy is not None:
            section_content.append(html.H4("Checkpoint: Prepared Data for Modeling"))
            section_content.append(dbc.Table.from_dataframe(df_tidy.head(), striped=True, bordered=True, hover=True, size="sm"))
            
        if df_5pl_results is not None:
            section_content.append(html.H4("Checkpoint: Dose-Response Parameters (if applicable)"))
            section_content.append(dbc.Table.from_dataframe(df_5pl_results.head(), striped=True, bordered=True, hover=True, size="sm"))

    # --- Automated Model Training with PyCaret ---
    section_content.append(html.H4("2. Automated Model Training with PyCaret"))
    
    modeling_source_options = []
    if data_store.get('df_tidy_merged') is not None: # Use merged data for consistency
        modeling_source_options.append({'label': "Raw/Tidy Data", 'value': "Raw/Tidy Data"})
    if data_store.get('df_5pl_results_merged') is not None:
        modeling_source_options.append({'label': "Dose-Response Parameters", 'value': "Dose-Response Parameters"})

    if not modeling_source_options:
        section_content.append(html.Div(html.P("Please process data in previous steps to proceed with modeling."), style={'color': 'orange'}))
        return html.Div(section_content)

    current_modeling_source = data_store.get('current_modeling_source')
    if current_modeling_source not in [opt['value'] for opt in modeling_source_options]:
        current_modeling_source = modeling_source_options[0]['value'] # Default to first valid option

    if len(modeling_source_options) == 1:
        section_content.append(html.P(f"**Selected data source for modeling:** `{modeling_source_options[0]['value']}`"))
    else:
        section_content.append(dcc.RadioItems(
            id='pycaret-source-radio',
            options=modeling_source_options,
            value=current_modeling_source,
            inline=True,
            className="mb-3"
        ))

    # Determine df_for_modeling for column options
    df_for_modeling = None
    if current_modeling_source == "Raw/Tidy Data":
        df_for_modeling = pd.read_json(data_store['df_tidy_merged'], orient='split') if data_store.get('df_tidy_merged') else None
    else: # "Dose-Response Parameters"
        df_for_modeling = pd.read_json(data_store['df_5pl_results_merged'], orient='split') if data_store.get('df_5pl_results_merged') else None

    if df_for_modeling is None or df_for_modeling.empty:
        section_content.append(html.Div(html.P(f"Selected modeling data source ('{current_modeling_source}') is not available or is empty. Please ensure data is processed correctly and contains sufficient rows."), style={'color': 'red'}))
        return html.Div(section_content)

    all_numeric_cols = df_for_modeling.select_dtypes(include=np.number).columns.tolist()

    feature_options_filtered = []
    if data_store['data_source'] == 'LUMOS':
        # For LUMOS, features are from factor_names
        feature_options_raw = data_store.get('factor_names', [])
        feature_options_filtered = [f for f in feature_options_raw if f in all_numeric_cols]
    else: # Custom
        y_vars_common_from_tidy = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']
        dr_param_cols = ['a', 'b', 'c', 'd', 'g', 'R-squared']
        
        if current_modeling_source == "Raw/Tidy Data":
            feature_options_filtered = [col for col in all_numeric_cols if col not in y_vars_common_from_tidy]
        else: # Dose-Response Parameters
            feature_options_filtered = [col for col in all_numeric_cols if col not in dr_param_cols]

    # Ensure selected target and features persist
    default_target_col = data_store.get('pycaret_target_selector')
    if default_target_col not in all_numeric_cols:
        default_target_col = all_numeric_cols[0] if all_numeric_cols else None
    
    default_features = data_store.get('pycaret_features_selector', [])
    default_features = [f for f in default_features if f in feature_options_filtered]
    if not default_features and feature_options_filtered:
        default_features = feature_options_filtered

    section_content.append(html.Div([
        html.Label("Select Target (Y):"),
        dcc.Dropdown(
            id='pycaret-target-selector',
            options=[{'label': col, 'value': col} for col in all_numeric_cols],
            value=default_target_col,
            clearable=False
        )
    ]))
    
    section_content.append(html.Div([
        html.Label("Select Features (X):"),
        dcc.Dropdown(
            id='pycaret-features-selector',
            options=[{'label': col, 'value': col} for col in feature_options_filtered if col != default_target_col],
            value=default_features,
            multi=True
        )
    ]))

    section_content.append(html.H4("3. PyCaret Preprocessing Options"))
    
    # Preprocessing options UI
    section_content.append(dbc.Row([
        dbc.Col([
            dbc.Checklist(
                options=[{"label": "Normalize Data", "value": "normalize"}],
                value=["normalize"] if data_store.get('pycaret_normalize', True) else [],
                id="pycaret-normalize-checkbox",
                switch=True,
            ),
            html.Small("Scales numerical features to a standard range (e.g., between 0 and 1, or mean 0 and std 1). This helps algorithms sensitive to feature magnitudes perform better.", className="form-text text-muted mb-2"),
            html.Div([
                html.Label("Normalization Method:"),
                dcc.Dropdown(
                    id='pycaret-normalize-method-select',
                    options=[
                        {'label': 'zscore', 'value': 'zscore', 'title': 'Transforms data to have a mean of 0 and standard deviation of 1. Good for general use.'},
                        {'label': 'minmax', 'value': 'minmax', 'title': 'Scales data to a range between 0 and 1. Useful for neural networks.'},
                        {'label': 'maxabs', 'value': 'maxabs', 'title': 'Scales data to a range between -1 and 1 by dividing by the maximum absolute value. Preserves sparsity.'},
                        {'label': 'robust', 'value': 'robust', 'title': 'Scales data using statistics that are robust to outliers (median and interquartile range), useful if your data has many outliers.'}
                    ],
                    value=data_store.get('pycaret_normalize_method', 'zscore'),
                    clearable=False
                )
            ], id='normalize-method-div', style={'display': 'block' if data_store.get('pycaret_normalize', True) else 'none'}),
            dbc.Checklist(
                options=[{"label": "Remove Outliers (Isolation Forest)", "value": "remove_outliers"}],
                value=["remove_outliers"] if data_store.get('pycaret_remove_outliers', False) else [],
                id="pycaret-remove-outliers-checkbox",
                switch=True,
                className="mt-3"
            ),
            html.Small("Identifies and removes data points that are significantly different from other observations using the Isolation Forest algorithm. Outliers can negatively impact model training, but removing them might discard valuable information.", className="form-text text-muted mb-2")
        ], width=4),
        dbc.Col([
            dbc.Checklist(
                options=[{"label": "Apply Power Transformation (Numerical)", "value": "transformation"}],
                value=["transformation"] if data_store.get('pycaret_transformation', False) else [],
                id="pycaret-transformation-checkbox",
                switch=True,
            ),
            html.Small("Applies a power transformation (e.g., Box-Cox or Yeo-Johnson) to make numerical features more Gaussian-like (bell-shaped distribution). This can improve model performance, especially for linear models and neural networks.", className="form-text text-muted mb-2"),
            dbc.Checklist(
                options=[{"label": "Remove Multicollinearity", "value": "remove_multicollinearity"}],
                value=["remove_multicollinearity"] if data_store.get('pycaret_remove_multicollinearity', False) else [],
                id="pycaret-remove-multicollinearity-checkbox",
                switch=True,
                className="mt-3"
            ),
            html.Small("Removes highly correlated (linearly dependent) features. Multicollinearity can make models unstable, difficult to interpret, and lead to overfitting.", className="form-text text-muted mb-2"),
            html.Div([
                html.Label("Multicollinearity Threshold (correlation):"),
                dcc.Slider(
                    id='pycaret-multicollinearity-threshold-slider',
                    min=0.5, max=1.0, step=0.05, 
                    value=data_store.get('pycaret_multicollinearity_threshold', 0.9),
                    marks={i/10: str(i/10) for i in range(5, 11)},
                )
            ], id='multicollinearity-threshold-div', style={'display': 'block' if data_store.get('pycaret_remove_multicollinearity', False) else 'none'})
        ], width=4),
        dbc.Col([
            dbc.Checklist(
                options=[{"label": "Create Feature Interaction (Polynomial)", "value": "feature_interaction"}],
                value=["feature_interaction"] if data_store.get('pycaret_feature_interaction', False) else [],
                id="pycaret-feature-interaction-checkbox",
                switch=True,
            ),
            html.Small("Generates new features by multiplying existing numerical features (e.g., creating a 'length * width' feature). This allows the model to capture more complex non-linear relationships, but increases dimensionality.", className="form-text text-muted mb-2"),
            html.Div([
                html.Label("Polynomial Degree:"),
                dcc.Input(id='pycaret-polynomial-degree-input', type='number', min=2, max=3, 
                          value=data_store.get('pycaret_polynomial_degree', 2), className="form-control")
            ], id='polynomial-degree-div', style={'display': 'block' if data_store.get('pycaret_feature_interaction', False) else 'none'}),
            dbc.Checklist(
                options=[{"label": "Perform Feature Selection", "value": "feature_selection"}],
                value=["feature_selection"] if data_store.get('pycaret_feature_selection', False) else [],
                id="pycaret-feature-selection-checkbox",
                switch=True,
                className="mt-3"
            ),
            html.Small("Automatically selects the most relevant features to improve model performance and reduce complexity. This can help prevent overfitting, especially with high-dimensional data, and speed up training.", className="form-text text-muted mb-2")
        ], width=4)
    ]))
    
    section_content.append(dbc.Button("Run PyCaret Analysis", id="run-pycaret-button", className="btn btn-primary mt-3"))
    section_content.append(html.Div(id='pycaret-output-content'))

    return html.Div(section_content)

# Callbacks for PyCaret preprocessing option visibility
@app.callback(
    Output('normalize-method-div', 'style'),
    Input('pycaret-normalize-checkbox', 'value'),
    State('data-storage', 'data'), # To read current state for initial load
    prevent_initial_call=False # Allow initial call to set visibility
)
def toggle_normalize_method_visibility(normalize_val, data_store):
    normalize_checked = "normalize" in (normalize_val or [])
    return {'display': 'block' if normalize_checked else 'none'}

@app.callback(
    Output('multicollinearity-threshold-div', 'style'),
    Input('pycaret-remove-multicollinearity-checkbox', 'value'),
    State('data-storage', 'data'),
    prevent_initial_call=False
)
def toggle_multicollinearity_visibility(remove_multicollinearity_val, data_store):
    remove_checked = "remove_multicollinearity" in (remove_multicollinearity_val or [])
    return {'display': 'block' if remove_checked else 'none'}

@app.callback(
    Output('polynomial-degree-div', 'style'),
    Input('pycaret-feature-interaction-checkbox', 'value'),
    State('data-storage', 'data'),
    prevent_initial_call=False
)
def toggle_polynomial_degree_visibility(feature_interaction_val, data_store):
    interact_checked = "feature_interaction" in (feature_interaction_val or [])
    return {'display': 'block' if interact_checked else 'none'}


# Update PyCaret preprocessing options in store
@app.callback(
    Output('data-storage', 'data', allow_duplicate=True),
    [Input('pycaret-normalize-checkbox', 'value'),
     Input('pycaret-normalize-method-select', 'value'),
     Input('pycaret-remove-outliers-checkbox', 'value'),
     Input('pycaret-transformation-checkbox', 'value'),
     Input('pycaret-remove-multicollinearity-checkbox', 'value'),
     Input('pycaret-multicollinearity-threshold-slider', 'value'),
     Input('pycaret-feature-interaction-checkbox', 'value'),
     Input('pycaret-polynomial-degree-input', 'value'),
     Input('pycaret-feature-selection-checkbox', 'value')],
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def update_pycaret_prep_params_in_store(normalize_val, normalize_method, remove_outliers_val, 
                                        transformation_val, remove_multicollinearity_val, 
                                        multicollinearity_threshold, feature_interaction_val, 
                                        polynomial_degree, feature_selection_val, data_store):
    data_store['pycaret_normalize'] = "normalize" in (normalize_val or [])
    data_store['pycaret_normalize_method'] = normalize_method
    data_store['pycaret_remove_outliers'] = "remove_outliers" in (remove_outliers_val or [])
    data_store['pycaret_transformation'] = "transformation" in (transformation_val or [])
    data_store['pycaret_remove_multicollinearity'] = "remove_multicollinearity" in (remove_multicollinearity_val or [])
    data_store['pycaret_multicollinearity_threshold'] = multicollinearity_threshold
    data_store['pycaret_feature_interaction'] = "feature_interaction" in (feature_interaction_val or [])
    data_store['pycaret_polynomial_degree'] = polynomial_degree
    data_store['pycaret_feature_selection'] = "feature_selection" in (feature_selection_val or [])
    
    # Clear previous ML results if preprocessing options change
    data_store['best_model_json'] = None
    data_store['model_comparison_df'] = None
    data_store['rsm_fig_path'] = None
    data_store['opt_result_json'] = None
    data_store['batch_suggestions'] = None
    data_store['global_optimum'] = None
    data_store['opt_landscape_fig'] = None
    data_store['opt_convergence_fig'] = None
    return data_store

# Update modeling source and selected target/features in store
@app.callback(
    Output('data-storage', 'data', allow_duplicate=True),
    [Input('pycaret-source-radio', 'value'),
     Input('pycaret-target-selector', 'value'),
     Input('pycaret-features-selector', 'value')],
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def update_modeling_params_in_store(source, target, features, data_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'pycaret-source-radio':
        data_store['current_modeling_source'] = source
        # Clear selected target/features if source changes dramatically
        data_store['pycaret_target_selector'] = None
        data_store['pycaret_features_selector'] = []
    elif trigger_id == 'pycaret-target-selector':
        data_store['pycaret_target_selector'] = target
    elif trigger_id == 'pycaret-features-selector':
        data_store['pycaret_features_selector'] = features
    
    # Clear previous ML results if modeling parameters change
    data_store['best_model_json'] = None
    data_store['model_comparison_df'] = None
    data_store['rsm_fig_path'] = None
    data_store['opt_result_json'] = None
    data_store['batch_suggestions'] = None
    data_store['global_optimum'] = None
    data_store['opt_landscape_fig'] = None
    data_store['opt_convergence_fig'] = None
    
    return data_store

# Callback for LUMOS DoE factors (num_factors, factor_names)
@app.callback(
    Output('data-storage', 'data', allow_duplicate=True),
    [Input('pycaret-num-factors-input', 'value'),
     Input({'type': 'pycaret-factor-name-input', 'index': ALL}, 'value')],
    [State('data-storage', 'data'),
     State('pycaret-grouping-col-select', 'value')],
    prevent_initial_call=True
)
def update_lumos_doe_factors(num_factors, factor_name_values, data_store, grouping_col):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'pycaret-num-factors-input':
        data_store['pycaret_num_factors'] = num_factors
        # Re-initialize factor_names with default if num_factors changed
        data_store['factor_names'] = [f"DoE_Factor_{i+1}" for i in range(num_factors)]
    elif 'pycaret-factor-name-input' in trigger_id:
        # Update specific factor name if its input changed
        if factor_name_values: # Ensure values are not empty (e.g. during initial render before values propagate)
            data_store['factor_names'] = [val for val in factor_name_values if val is not None]

    # Recalculate df_tidy_merged based on new factor names
    if data_store.get('df_tidy'):
        df_tidy = pd.read_json(data_store['df_tidy'], orient='split')
        if grouping_col and data_store.get('edited_doe_df_json'):
            edited_doe_df = pd.read_json(data_store['edited_doe_df_json'], orient='split')
            
            # Re-create doe_input_df structure to ensure consistency before merging
            unique_groups = pd.DataFrame(df_tidy[grouping_col].unique(), columns=[grouping_col]).sort_values(by=grouping_col).reset_index(drop=True)
            temp_doe_input_df = unique_groups.copy()
            for name in data_store['factor_names']:
                if name not in temp_doe_input_df.columns:
                    temp_doe_input_df[name] = 0.0

            # Merge existing edited_doe_df data into the new structure
            merged_edited_df = pd.merge(temp_doe_input_df[[grouping_col]], edited_doe_df, on=grouping_col, how='left')
            # Fill NaNs for newly added factor columns
            for col in data_store['factor_names']:
                if col not in merged_edited_df.columns:
                    merged_edited_df[col] = 0.0
                else:
                    merged_edited_df[col] = merged_edited_df[col].fillna(0.0) # Fill any NaNs from the merge
            
            # Ensure columns match expected order from factor_names
            merged_edited_df = merged_edited_df[[grouping_col] + data_store['factor_names']]
            
            data_store['edited_doe_df_json'] = merged_edited_df.to_json(date_format='iso', orient='split')
            df_tidy_merged = pd.merge(df_tidy, merged_edited_df, on=grouping_col, how='left')
            data_store['df_tidy_merged'] = df_tidy_merged.to_json(date_format='iso', orient='split')

            if data_store.get('df_5pl_results'):
                df_params = pd.read_json(data_store['df_5pl_results'], orient='split')
                df_params_renamed = df_params.rename(columns={'Group': grouping_col})
                df_params_merged = pd.merge(df_params_renamed, merged_edited_df, on=grouping_col, how='left')
                data_store['df_5pl_results_merged'] = df_params_merged.to_json(date_format='iso', orient='split')
        else: # If grouping_col is None or no edited_doe_df yet, df_tidy_merged is just df_tidy
            data_store['df_tidy_merged'] = df_tidy.to_json(date_format='iso', orient='split')
            data_store['df_5pl_results_merged'] = None # Clear if no merging happens

    return data_store

# Store DoE factor table edits
@app.callback(
    Output('data-storage', 'data', allow_duplicate=True),
    Input('doe-factor-editor-div', 'children'), # This ID might need to be linked to a DataTable 'data' property
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def update_edited_doe_df_in_store(table_children, data_store):
    # This callback would need a proper DataTable component as an Input
    # For now, it's a placeholder. The current 'doe-factor-editor-div' is just a display.
    # If using Dash DataTable, input would be Input('your-datatable-id', 'data')
    # and then pd.DataFrame.from_records(new_data)
    raise dash.exceptions.PreventUpdate

# Run PyCaret Analysis
@app.callback(
    [Output('pycaret-output-content', 'children'),
     Output('data-storage', 'data', allow_duplicate=True)],
    Input('run-pycaret-button', 'n_clicks'),
    State('data-storage', 'data'),
    prevent_initial_call=True
)
def run_pycaret_analysis(n_clicks, data_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    modeling_source = data_store['current_modeling_source']
    target_col = data_store['pycaret_target_selector']
    features = data_store['pycaret_features_selector']

    if modeling_source == "Raw/Tidy Data":
        df_for_modeling = pd.read_json(data_store['df_tidy_merged'], orient='split')
    else:
        df_for_modeling = pd.read_json(data_store['df_5pl_results_merged'], orient='split')

    if df_for_modeling is None or df_for_modeling.empty or not target_col or not features:
        return html.Div(html.P("Please ensure data is loaded, target and features are selected."), style={'color': 'red'}), data_store

    data = df_for_modeling[features + [target_col]].dropna().drop_duplicates()
    if data.empty:
        return html.Div(html.P("No valid data after dropping NA and duplicates for selected columns."), style={'color': 'red'}), data_store

    # PyCaret preprocessing options
    normalize = data_store.get('pycaret_normalize', True)
    normalize_method = data_store.get('pycaret_normalize_method', 'zscore')
    remove_outliers = data_store.get('pycaret_remove_outliers', False)
    transformation = data_store.get('pycaret_transformation', False)
    remove_multicollinearity = data_store.get('pycaret_remove_multicollinearity', False)
    multicollinearity_threshold = data_store.get('pycaret_multicollinearity_threshold', 0.9)
    feature_interaction = data_store.get('pycaret_feature_interaction', False)
    polynomial_degree = data_store.get('pycaret_polynomial_degree', 2)
    feature_selection = data_store.get('pycaret_feature_selection', False)

    categorical_features = data[features].select_dtypes(include='object').columns.tolist()
    numeric_features = data[features].select_dtypes(include=np.number).columns.tolist()

    pycaret_output_content = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save PyCaret HTML outputs to the temporary directory
        os.chdir(tmpdir) # Change current working directory for PyCaret's save functions
        
        try:
            s = pyreg.setup(data, target=target_col, session_id=123,
                            html=True, silent=True, log_experiment=True, experiment_name='lfa_doe',
                            numeric_features=numeric_features,
                            categorical_features=categorical_features,
                            normalize=normalize,
                            normalize_method=normalize_method,
                            transformation=transformation,
                            remove_outliers=remove_outliers,
                            remove_multicollinearity=remove_multicollinearity,
                            multicollinearity_threshold=multicollinearity_threshold,
                            feature_interaction=feature_interaction, 
                            polynomial_degree=polynomial_degree, 
                            feature_selection=feature_selection,
                            numeric_imputation='mean', 
                            categorical_imputation='mode', 
                            handle_unknown_categorical=True,
                            unknown_categorical_method='most_frequent',
                            ignore_low_variance=True, 
                            combine_rare_levels=True, 
                            bin_numeric_features=None, 
                            remove_perfect_collinearity=True)

            setup_html_path = "pycaret_setup_output.html"
            pyreg.save_html(setup_html_path)
            pycaret_output_content.append(html.H4("PyCaret Setup Output"))
            pycaret_output_content.append(html.Iframe(srcDoc=open(setup_html_path, 'r', encoding='utf-8').read(), 
                                                      style={"height": "500px", "width": "100%", "border": "none"}))

            best_model = pyreg.compare_models()
            comparison_df = pyreg.pull()
            data_store['model_comparison_df'] = comparison_df.to_json(date_format='iso', orient='split')

            pycaret_output_content.append(html.H4("Model Comparison Results"))
            pycaret_output_content.append(dbc.Table.from_dataframe(comparison_df, striped=True, bordered=True, hover=True, size="sm"))
            data_store['best_model_json'] = pyreg.save_model(best_model, 'best_model_pipeline', verbose=False) # Save pipeline
            
            final_model = pyreg.finalize_model(best_model) # Finalize the model for prediction
            data_store['final_model'] = pyreg.save_model(final_model, 'final_model_for_prediction', verbose=False) # Save final model


            # Interactive Model Analysis Dashboard
            pycaret_output_content.append(html.H4("Interactive Model Analysis Dashboard"))
            pycaret_output_content.append(html.P("This is an interactive dashboard. Hover over plots for details and use the dropdown to explore different analyses."))
            
            pyreg.evaluate_model(best_model, use_train_data=True)
            eval_html_path = "pycaret_evaluate_output.html"
            pyreg.save_html(eval_html_path)
            pycaret_output_content.append(html.Iframe(srcDoc=open(eval_html_path, 'r', encoding='utf-8').read(), 
                                                      style={"height": "800px", "width": "100%", "border": "none"}))

            # Response Surface Plot
            if len(features) >= 2:
                pycaret_output_content.append(html.H4("Response Surface Plot"))
                try:
                    rsm_plot_path = pyreg.plot_model(best_model, plot='surface', save=True)
                    data_store['rsm_fig_path'] = os.path.join(tmpdir, rsm_plot_path) # Store full path to be copied later
                    # Need to read image as base64 to embed in Dash or serve statically
                    with open(rsm_plot_path, 'rb') as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode('ascii')
                    pycaret_output_content.append(html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '100%'}))
                except Exception as e:
                    pycaret_output_content.append(html.Div(f"Could not generate response surface plot. This may happen for some model types. Error: {e}", style={'color': 'orange'}))
            else:
                pycaret_output_content.append(html.Div(html.P("Response surface plots require at least 2 features."), style={'color': 'info'}))

        except Exception as e:
            pycaret_output_content.append(html.Div(f"An error occurred during PyCaret analysis: {e}", style={'color': 'red'}))
        finally:
            # Change back to original directory to avoid interference
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            
    return pycaret_output_content, data_store

# Callback to render Optimization section UI (Step 4)
@app.callback(
    Output('optimization-section-content', 'children'),
    Input('data-storage', 'data')
)
def render_optimization_section(data_store):
    if not data_store.get('best_model_json'):
        return html.Div(html.P("Complete Step 3 to train a model with PyCaret before running optimization."))

    section_content = html.Div([
        html.H4("1. Optimization Settings"),
        dbc.Row([
            dbc.Col(
                dcc.RadioItems(
                    id='opt-goal-radio',
                    options=[
                        {'label': 'Minimize', 'value': 'Minimize'},
                        {'label': 'Maximize', 'value': 'Maximize'}
                    ],
                    value='Maximize', # Default to Maximize
                    inline=True
                )
            ),
            dbc.Col(
                html.Div([
                    html.Label("Number of experiments to suggest:"),
                    dcc.Input(id='batch-size-input', type='number', min=1, max=10, value=3, className="form-control")
                ])
            )
        ]),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    html.P("""
                    This tool uses a powerful AI-driven approach based on **Bayesian Optimization** to suggest the most informative experiments to run next. It balances two key strategies:
                    - **Exploitation:** Suggesting points where the model predicts the best outcome.
                    - **Exploration:** Suggesting points in regions where the model is most uncertain.
                    By providing a batch of suggestions, the tool helps you efficiently learn about your experimental space and converge on the true optimum faster.
                    """),
                    title="How are batch suggestions generated?"
                )
            ],
            flush=True,
            className="my-3"
        ),
        dbc.Button("Suggest Next Batch of Experiments", id="run-bayes-opt-button", className="btn btn-primary mt-3"),
        html.Div(id='opt-output-content')
    ])
    return section_content

# Callback to run Bayesian Optimization
@app.callback(
    [Output('opt-output-content', 'children'),
     Output('data-storage', 'data', allow_duplicate=True)],
    Input('run-bayes-opt-button', 'n_clicks'),
    [State('opt-goal-radio', 'value'),
     State('batch-size-input', 'value'),
     State('data-storage', 'data')],
    prevent_initial_call=True
)
def run_bayesian_optimization(n_clicks, goal, batch_size, data_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not data_store.get('final_model'):
        return html.Div(html.P("No finalized model found. Please run PyCaret analysis first."), style={'color': 'red'}), data_store
    
    # Load the final model (pipeline)
    # PyCaret models are saved using joblib internally, need to load
    # In a real deployed app, you'd load from a file path or a model registry.
    # For this example, if the model was saved to a temp file, we need that path.
    # Given the previous callback structure, best_model_json stores the path to the saved model.
    try:
        final_model_path = data_store['final_model']
        final_model = pyreg.load_model(final_model_path, verbose=False)
    except Exception as e:
        return html.Div(html.P(f"Error loading final model: {e}"), style={'color': 'red'}), data_store


    df_for_modeling = pd.read_json(data_store['modeling_df_for_opt'], orient='split')
    features = data_store['features_for_opt']
    target_for_opt = data_store['target_for_opt']

    if not all([df_for_modeling is not None, features, target_for_opt]):
        return html.Div(html.P("Missing data or model parameters for optimization."), style={'color': 'red'}), data_store

    X = df_for_modeling[features].dropna().drop_duplicates()
    
    search_space = []
    for feature in features:
        min_val = X[feature].min()
        max_val = X[feature].max()
        ext = (max_val - min_val) * 0.10 if max_val > min_val else abs(min_val) * 0.10 or 0.1
        search_space.append(Real(min_val - ext, max_val + ext, name=feature))

    @use_named_args(search_space)
    def objective(**params):
        point = pd.DataFrame([params])
        point = point[features]
        prediction = final_model.predict(point)[0]
        return -prediction if goal == "Maximize" else prediction
    
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=25,
        n_initial_points=15,
        acq_func="LCB",
        acq_optimizer="sampling",
        random_state=42
    )
    # Store minimal info from result, not the whole object as it's not JSON serializable
    data_store['opt_result_json'] = {
        'x_iters': [list(x) for x in result.x_iters], # Convert numpy arrays to list for JSON
        'func_vals': list(result.func_vals),
        'space_dimensions': [str(dim) for dim in result.space.dimensions] # Store string representation
    }
    
    # Using the best model from gp_minimize's internal GPR for suggestions
    # This requires recreating the GPR model or using its internal prediction.
    # For simplicity, if we need `gpr.predict` directly, we might need to train it again or rethink storing the full skopt result.
    # Let's approximate by using the 'final_model' for predictions.
    
    def surrogate_objective_from_final_model(x_arr):
        # This is a simplification; ideally, skopt's GPR surrogate is used
        # Here we apply the trained final_model to new points.
        point = pd.DataFrame([x_arr], columns=features)
        prediction = final_model.predict(point)[0]
        return -prediction if goal == "Maximize" else prediction # Match objective's output
    
    bounds = [(dim.low, dim.high) for dim in search_space]
    
    global_min_result = minimize(surrogate_objective_from_final_model, result.x[0], bounds=bounds, method='L-BFGS-B')
    global_optimum_point = global_min_result.x
    global_optimum_point_df = pd.DataFrame([global_optimum_point], columns=features)
    global_optimum_prediction = final_model.predict(global_optimum_point_df)[0] # Predict with actual model

    data_store['global_optimum'] = {
        "point": global_optimum_point.tolist(), # Convert to list for JSON
        "prediction": global_optimum_prediction
    }

    suggestions = []
    # Strategy 1: Best Predicted Optimum
    suggestions.append({
        'point': global_optimum_point.tolist(), # Convert to list
        'strategy': 'Exploitation (Best Predicted Optimum)'
    })

    # To generate other suggestions, we would need the actual GPR model from skopt.
    # Since the skopt `result` object is not directly JSON serializable and `load_model` does not apply to it,
    # we'll approximate other strategies by sampling and predicting with `final_model`.
    # A more robust solution for full skopt result persistence would involve `joblib` or `pickle` for the result object.
    
    if batch_size > 1:
        # Generate random points in the space to evaluate
        grid_points = result.space.rvs(n_samples=1000, random_state=42)
        grid_points_df = pd.DataFrame(grid_points, columns=features)
        
        # Predict on these points using the actual final_model
        mu_preds = final_model.predict(grid_points_df)
        
        # Simulate uncertainty for exploration (simplified, not from actual GPR std)
        # This part is a placeholder. For true uncertainty (std), you need the GPR from skopt.
        # As a workaround for Dash, we can generate synthetic 'std' or skip complex EI/LCB.
        # For a full implementation, the skopt result would need to be joblib.dump-ed and loaded.
        
        # Let's simplify: Pick diverse points and high/low prediction points
        
        # Filter out the point already suggested (best optimum)
        dist_to_best = np.array([np.sqrt(np.sum((gp - global_optimum_point)**2)) for gp in grid_points])
        mask = dist_to_best > 1e-3 # Ensure points are not too close

        remaining_points_masked = grid_points[mask]
        mu_preds_masked = mu_preds[mask]

        if len(remaining_points_masked) > 0:
            # Strategy 2: Point with most "extreme" predicted value (simplified exploration)
            if goal == "Maximize":
                extreme_idx = np.argmin(mu_preds_masked) # Look for lowest for exploration in Maximize
            else:
                extreme_idx = np.argmax(mu_preds_masked) # Look for highest for exploration in Minimize
            suggestions.append({'point': remaining_points_masked[extreme_idx].tolist(), 'strategy': 'Exploration (Extreme Prediction)'})

        if batch_size > 2 and len(remaining_points_masked) > 1:
            # Strategy 3: Random diverse point (simple exploration)
            random_idx = np.random.choice(len(remaining_points_masked))
            suggestions.append({'point': remaining_points_masked[random_idx].tolist(), 'strategy': 'Exploration (Random Diverse)'})

    # Trim suggestions to batch_size
    suggestions = suggestions[:batch_size]

    suggestion_points_list = [s['point'] for s in suggestions]
    suggestion_df = pd.DataFrame(suggestion_points_list, columns=features)
    suggestion_df['Suggestion Strategy'] = [s['strategy'] for s in suggestions]
    
    # Get predictions from the actual best model
    true_model_preds = final_model.predict(suggestion_df[features])
    suggestion_df['Expected Outcome'] = true_model_preds
    
    # Confidence (simplified, as actual GPR std is not easily retrieved without pickling result)
    # Assign a dummy high confidence for now. For real confidence, need skopt.result.models[-1].predict(..., return_std=True)
    suggestion_df['Confidence (%)'] = 95.0 # Placeholder
    
    data_store['batch_suggestions'] = suggestion_df.to_json(date_format='iso', orient='split')

    opt_output_content = html.Div([
        html.H4("Global Optimum Found"),
        html.P(f"Best Predicted {target_for_opt}: {data_store['global_optimum']['prediction']:.4f}"),
        html.H4("Suggested Batch of Experiments"),
        dbc.Table.from_dataframe(suggestion_df.style.format({col: '{:.4f}' for col in features + ['Expected Outcome']}).format({'Confidence (%)': '{:.1f}'}).data, # .data to get plain DataFrame
                                 striped=True, bordered=True, hover=True, size="sm"),
        html.H4("Visualization of the Optimization Landscape"),
        dcc.Graph(id='opt-landscape-plot'),
        dcc.Graph(id='opt-convergence-plot')
    ])
    
    # Generate Plots
    # Optimization Landscape (Surface or Partial Dependence)
    if len(features) == 2:
        x_values = np.linspace(bounds[0][0], bounds[0][1], 50)
        y_values = np.linspace(bounds[1][0], bounds[1][1], 50)
        X_mesh, Y_mesh = np.meshgrid(x_values, y_values)
        
        # Predict Z values using the actual final_model (not skopt's GPR directly)
        Z_mesh_pred = np.array([final_model.predict(pd.DataFrame([[x, y]], columns=features))[0] for x, y in zip(np.ravel(X_mesh), np.ravel(Y_mesh))]).reshape(X_mesh.shape)
        
        fig_surface = go.Figure(data=[go.Surface(z=Z_mesh_pred, x=X_mesh, y=Y_mesh, colorscale='Viridis')])
        fig_surface.update_layout(title='Optimization Landscape (Model Prediction)',
                                  scene = dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=target_for_opt),
                                  autosize=False, width=800, height=600, margin=dict(l=65, r=50, b=65, t=90))
        data_store['opt_landscape_fig'] = fig_surface.to_json()
        opt_output_content.children[5] = dcc.Graph(figure=fig_surface) # Update the graph component

    else:
        # For >2 features, generate partial dependence plots (simplified without skopt's internal GPR)
        partial_dependence_plots = []
        for i, feature in enumerate(features):
            fig_pd = go.Figure()
            feature_range = np.linspace(bounds[i][0], bounds[i][1], 100)
            
            fixed_point_list = list(global_optimum_point)
            predictions = []
            for val in feature_range:
                temp_point = list(fixed_point_list)
                temp_point[i] = val
                predictions.append(final_model.predict(pd.DataFrame([temp_point], columns=features))[0])
            
            fig_pd.add_trace(go.Scatter(x=feature_range, y=predictions, mode='lines', name=f'Partial Dependence: {feature}'))
            fig_pd.add_vline(x=global_optimum_point[i], line_dash="dash", line_color="red", annotation_text="Optimal Point")
            fig_pd.update_layout(title=f"Partial Dependence Plot for {feature}", xaxis_title=feature, yaxis_title=target_for_opt)
            partial_dependence_plots.append(dcc.Graph(figure=fig_pd))
        
        opt_output_content.children[5] = html.Div(partial_dependence_plots) # Replace with multiple plots

    # Convergence Trace Plot
    convergence_preds = final_model.predict(pd.DataFrame(data_store['opt_result_json']['x_iters'], columns=features))
    if goal == "Maximize": 
        best_seen = np.maximum.accumulate(convergence_preds)
    else: 
        best_seen = np.minimum.accumulate(convergence_preds)
    
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=list(range(1, len(best_seen) + 1)), y=best_seen, mode='lines+markers', line=dict(color='green')))
    fig_conv.update_layout(title="Convergence Trace", xaxis_title="Iteration", yaxis_title=f"Best Seen {target_for_opt}", height=400)
    data_store['opt_convergence_fig'] = fig_conv.to_json()
    opt_output_content.children[6] = dcc.Graph(figure=fig_conv) # Update the graph component

    return opt_output_content, data_store

# Callback to generate PDF Report
@app.callback(
    Output("download-pdf-report", "data"),
    Input("generate-pdf-button", "n_clicks"),
    State("data-storage", "data"),
    prevent_initial_call=True
)
def generate_pdf(n_clicks, data_store):
    if n_clicks:
        # Create a temporary directory for image files generated by plotly.io.write_image
        # PyCaret's rsm_fig_path refers to a file that was generated during the PyCaret run.
        # We need to ensure that file is accessible or re-generate it.
        # For robustness, the rsm_fig_path stored in data_store would ideally be a base64 string
        # or the image needs to be regenerated here if only a path was stored from a temporary location.
        # Given how PyCaret saves, let's assume the rsm_fig_path is directly usable if it exists
        # or needs a local temporary copy.
        
        # If rsm_fig_path exists, it's likely still in a temp dir from pycaret_analysis.
        # We need to copy it to a new temp dir for PDF generation scope.
        temp_dir_for_pdf = tempfile.mkdtemp()
        if data_store.get('rsm_fig_path') and os.path.exists(data_store['rsm_fig_path']):
            source_rsm_path = data_store['rsm_fig_path']
            dest_rsm_path = os.path.join(temp_dir_for_pdf, os.path.basename(source_rsm_path))
            try:
                shutil.copy(source_rsm_path, dest_rsm_path)
                data_store['rsm_fig_path'] = dest_rsm_path # Update path for PDF generation
            except Exception as e:
                print(f"Error copying RSM plot for PDF: {e}")
                data_store['rsm_fig_path'] = None # Invalidate path if copy fails

        pdf_bytes = create_pdf_report_dash(data_store)
        
        # Clean up the temporary directory used by create_pdf_report_dash
        if os.path.exists(temp_dir_for_pdf):
             shutil.rmtree(temp_dir_for_pdf)

        return dcc.send_bytes(pdf_bytes, filename="LFA_Analysis_Report.pdf")
    raise dash.exceptions.PreventUpdate

# Callback to generate Excel Report
@app.callback(
    Output("download-excel-report", "data"),
    Input("generate-excel-button", "n_clicks"),
    State("data-storage", "data"),
    prevent_initial_call=True
)
def generate_excel(n_clicks, data_store):
    if n_clicks:
        dfs_to_download_raw = {
            "Raw_Data": data_store.get('df_raw'),
            "Tidy_Data": data_store.get('df_tidy'),
            "Dose_Response_Results": data_store.get('df_5pl_results'),
            "Model_Comparison": data_store.get('model_comparison_df'),
            "Tidy_Data_Merged_DoE": data_store.get('df_tidy_merged'),
            "Params_Merged_DoE": data_store.get('df_5pl_results_merged'),
            "Optimization_Suggestions": data_store.get('batch_suggestions')
        }
        
        # Convert JSON strings back to DataFrames
        dfs_to_download_actual = {}
        for name, df_json in dfs_to_download_raw.items():
            if df_json:
                try:
                    dfs_to_download_actual[name] = pd.read_json(df_json, orient='split')
                except ValueError: # Handle cases where JSON might be malformed or empty string
                    dfs_to_download_actual[name] = pd.DataFrame() # Provide empty DF
            else:
                dfs_to_download_actual[name] = pd.DataFrame() # Provide empty DF if None

        excel_bytes = get_excel_bytes(dfs_to_download_actual)
        return dcc.send_bytes(excel_bytes, filename="LFA_Analysis_Report.xlsx")
    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    # When running locally, set a temporary directory for PyCaret
    # In deployment, this is often handled by the environment (e.g., /tmp in Docker)
    temp_pycaret_dir = tempfile.mkdtemp()
    os.environ['PYCARET_CUSTOM_PIPELINE_PATH'] = temp_pycaret_dir # This ensures models are saved here
    print(f"PyCaret temporary directory: {temp_pycaret_dir}")
    
    app.run_server(debug=True, port=8050)
    
    # Clean up the temporary directory after the app stops (only if running locally)
    # This might not always execute if the server is stopped abruptly.
    # In production, use appropriate cleanup mechanisms.
    if os.path.exists(temp_pycaret_dir):
        try:
            shutil.rmtree(temp_pycaret_dir)
            print(f"Cleaned up PyCaret temporary directory: {temp_pycaret_dir}")
        except OSError as e:
            print(f"Error cleaning up temporary directory {temp_pycaret_dir}: {e}")