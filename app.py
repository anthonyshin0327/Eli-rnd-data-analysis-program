import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycaret.regression as pyreg
import streamlit.components.v1 as components
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
from io import BytesIO
import base64
import warnings
import itertools
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime
import shutil

# --- New Imports for Export Features ---
from fpdf import FPDF
import plotly.io as pio
try:
    import kaleido
except ImportError:
    st.warning("Kaleido package not found. Static image export for Plotly figures will not be available. Please install it using 'pip install kaleido'")

# --- New Imports for Optimization ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- !!! ADDED FOR TROUBLESHOOTING !!! ---
from sklearn.metrics import r2_score 
# --- !!! END ADDITION !!! ---

# --- Page Configuration ---
st.set_page_config(
    page_title="LFA Analysis & ML Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Suppress Warnings for Cleaner Output ---
warnings.filterwarnings("ignore")

# --- Helper Functions for App Logic---

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

def get_excel_download_link(dfs_dict, filename):
    """Generates a link to download a multi-sheet Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs_dict.items():
            if df is not None:
                try:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    st.warning(f"Could not write sheet '{sheet_name}': {e}")
    excel_data = output.getvalue()
    # Fixed: Changed base66 to base64
    b64 = base64.b64encode(excel_data).decode() 
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Full Report (XLSX)</a>'
    return href

def save_plotly_figure_as_image(fig, filename):
    """Save a plotly figure as PNG image."""
    try:
        pio.write_image(fig, filename, width=800, height=500, scale=2)
        return True
    except Exception as e:
        st.warning(f"Could not save Plotly figure {os.path.basename(filename)}: {e}")
        return False

def save_matplotlib_figure_as_image(fig, filename):
    """Save a matplotlib figure as PNG image."""
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        return True
    except Exception as e:
        st.warning(f"Could not save Matplotlib figure {os.path.basename(filename)}: {e}")
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
        col_width = page_width / num_cols
        
        for i, header in enumerate(col_names):
            self.cell(col_width, 8, str(header), 1, 0, 'C')
        self.ln()
        
        self.set_font('Times', '', 9)
        for index, row in df.iterrows():
            for i, item in enumerate(row):
                 self.cell(col_width, 8, str(item), 1, 0, 'C')
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


def create_pdf_report(temp_dir):
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
    if st.session_state.get('df_tidy') is not None:
        methods_text += "Data was preprocessed from its raw format into a tidy dataset. "
    if st.session_state.get('df_5pl_results') is not None:
        model_choice = st.session_state.get('dr_model_choice', '4PL/5PL')
        methods_text += f"Dose-response curves were fitted using a {model_choice} model. "
    if st.session_state.get('best_model') is not None:
        methods_text += "Machine learning models were trained and compared using the PyCaret library to predict the target variable. The best model was selected for further analysis and optimization. "
    if st.session_state.get('opt_result') is not None:
        methods_text += "Bayesian Optimization with a Lower Confidence Bound (LCB) acquisition function was employed to suggest optimal experimental conditions. "
    pdf.chapter_body(methods_text)

    # --- Results Section ---
    pdf.add_page()
    pdf.chapter_title("2. Results")

    # Dose-Response
    if 'df_5pl_results' in st.session_state and st.session_state.df_5pl_results is not None:
        pdf.chapter_title("2.1 Dose-Response Analysis", level=2)
        pdf.chapter_body("The following table summarizes the fitted parameters for the dose-response models. 'c' represents the IC50 value, and 'R-squared' indicates the goodness of fit.")
        pdf.add_df_to_pdf(st.session_state.df_5pl_results.copy(), "Table 1: Dose-Response Model Parameters.")
        if 'dose_response_fig' in st.session_state and st.session_state.dose_response_fig:
            img_path = os.path.join(temp_dir, "dose_response.png")
            if save_plotly_figure_as_image(st.session_state.dose_response_fig, img_path):
                pdf.add_image(img_path, "Figure 1: Fitted dose-response curves for each experimental group.")

    # Modeling
    if 'model_comparison_df' in st.session_state and st.session_state.model_comparison_df is not None:
        pdf.add_page()
        pdf.chapter_title("2.2 DoE Modeling", level=2)
        pdf.chapter_body("Multiple regression models were automatically trained and evaluated. The table below compares their performance based on key metrics like R-squared, which measures how well the model explains the variance in the data.")
        pdf.add_df_to_pdf(st.session_state.model_comparison_df.copy(), "Table 2: Model Performance Comparison.")
        
        if 'rsm_fig_path' in st.session_state and st.session_state.rsm_fig_path:
            pdf.add_image(st.session_state.rsm_fig_path, "Figure 2: Response surface plot from the best performing model.")
    
    # Optimization
    if 'batch_suggestions' in st.session_state and st.session_state.batch_suggestions is not None:
        pdf.add_page()
        pdf.chapter_title("2.3 Bayesian Optimization", level=2)
        pdf.chapter_body("Based on the best model, Bayesian Optimization was used to suggest a new batch of experiments. The suggestions balance exploiting known optimal regions with exploring uncertain areas to improve the model.")
        if st.session_state.get('global_optimum') is not None:
            pdf.chapter_body(f"The predicted global optimum for {st.session_state.target_for_opt} is {st.session_state.global_optimum['prediction']:.4f}.")
        
        pdf.add_df_to_pdf(st.session_state.batch_suggestions.copy(), "Table 3: Suggested Batch of Experiments.")
        if 'opt_landscape_fig' in st.session_state and st.session_state.opt_landscape_fig:
            img_path = os.path.join(temp_dir, "opt_landscape.png")
            if save_plotly_figure_as_image(st.session_state.opt_landscape_fig, img_path):
                pdf.add_image(img_path, "Figure 3: Optimization landscape and partial dependence plots.")

    pdf_path = os.path.join(temp_dir, "report.pdf")
    pdf.output(pdf_path)
    return pdf_path

def get_pdf_download_link(pdf_file_path, filename):
    """Generate download link for PDF file."""
    with open(pdf_file_path, "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report (.pdf)</a>'
    return href

# --- Main App Body ---
def main():
    """Main function to run the Streamlit application."""
    with st.sidebar:
        st.title("🔬 LFA Analysis & ML Suite")
        st.markdown("---")
        
        st.header("1. Data Upload")
        # Initialize session state keys
        keys_to_init = [
            'data_source', 'df_raw', 'df_tidy', 'df_5pl_results',
            'df_doe_factors', 'df_tidy_doe', 'df_5pl_results_doe',
            'final_doe_df', 'df_tidy_merged', 'df_5pl_results_merged',
            'best_model', 'final_model', 'pycaret_preprocessor',
            'modeling_df_for_opt', 'features_for_opt', 'target_for_opt',
            'opt_result', 'batch_suggestions', 'global_optimum', 
            'dose_response_fig', 'opt_landscape_fig', 'opt_convergence_fig',
            'model_comparison_df', 'rsm_fig_path', 'perform_dr', # Added 'perform_dr' to session state
            'current_modeling_source' # Added to persist radio button selection
        ]
        for key in keys_to_init:
            if key not in st.session_state:
                st.session_state[key] = None

        st.radio(
            "Choose data source", ('LUMOS', 'Custom'),
            key='data_source',
            help="Select 'LUMOS' for standard files or 'Custom' for other formats."
        )
        uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx"])

        if uploaded_file:
            # Clear state on new upload
            # Preserve 'data_source' and 'perform_dr' to maintain user choices
            for key in keys_to_init:
                if key in ['data_source', 'perform_dr', 'current_modeling_source']: # Also preserve current_modeling_source
                    continue
                st.session_state[key] = None
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df_raw = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df_raw = pd.read_excel(uploaded_file)
                st.sidebar.success("File uploaded successfully!")

                if st.session_state.data_source == 'Custom':
                    st.session_state.df_tidy = st.session_state.df_raw.copy()
                # Ensure df_tidy_merged is cleared/re-initialized on new upload
                st.session_state.df_tidy_merged = None
                st.session_state.df_5pl_results_merged = None
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.df_raw = None
        
        st.markdown("---")
        st.header("2. Navigation")
        st.markdown("""
        * [Data Preprocessing](#step-1-data-preprocessing)
        * [Dose-Response Analysis](#step-2-dose-response-analysis)
        * [DoE Modeling with PyCaret](#step-3-doe-modeling-with-pycaret)
        * [Bayesian Optimization](#step-4-bayesian-optimization)
        * [Download Report](#step-5-generate-and-download-report)
        """)

    st.title("LFA Data Analysis and Machine Learning Pipeline")

    if st.session_state.df_raw is not None:
        handle_lumos_processing()
        handle_dose_response_regression()
        handle_pycaret_modeling()
        handle_bayesian_optimization()
        handle_reporting()
    else:
        st.info("Awaiting data file upload in the sidebar...")

def handle_lumos_processing():
    """Handles the preprocessing of LUMOS-specific data."""
    with st.container(border=True):
        st.header("Step 1: Data Preprocessing", anchor="step-1-data-preprocessing")
        
        if st.session_state.df_raw is None:
            return

        st.subheader("Uploaded Data Preview")
        st.dataframe(st.session_state.df_raw.head())
        
        if st.session_state.data_source == 'LUMOS':
            st.subheader("LUMOS Data Processing")
            df_raw = st.session_state.df_raw
            required_cols = ['strip name', 'line_peak_above_background_1', 'line_peak_above_background_2']
            if not all(col in df_raw.columns for col in required_cols):
                st.error(f"LUMOS file must contain: {', '.join(required_cols)}")
                return

            if df_raw['strip name'].empty:
                st.error("'strip name' column is empty.")
                return

            first_strip_name = str(df_raw['strip name'].iloc[0])
            delimiter = '-' if '-' in first_strip_name else '_' if '_' in first_strip_name else None
            
            if not delimiter:
                st.warning("Could not auto-detect delimiter ('-' or '_') in 'strip name'. Treating 'strip name' as a single factor.")
                num_groups = 1
                group_names = [st.text_input("Group 1 Name", "Factor_1", key="group_name_0")]
            else:
                num_groups = len(first_strip_name.split(delimiter))
                st.markdown(f"Detected **{num_groups}** groups in 'strip name' based on delimiter '{delimiter}'. Please name them:")
                group_names = [st.text_input(f"Group {i+1}", f"Factor_{i+1}", key=f"group_name_{i}") for i in range(num_groups)]

            with st.spinner("Processing..."):
                df = df_raw.copy()
                if delimiter:
                    split_cols = df['strip name'].astype(str).str.split(delimiter, expand=True)
                    if split_cols.shape[1] != len(group_names):
                        st.error(f"Mismatch in group count. Expected {len(group_names)} but found {split_cols.shape[1]} for some rows. Please check 'strip name' consistency.")
                        return
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
                st.session_state.df_tidy = df[final_cols]
        
        if st.session_state.get('df_tidy') is not None:
            st.subheader("Checkpoint: Processed Tidy Data")
            st.markdown("This is the resulting table from Step 1. It will be used as the input for all subsequent steps.")
            st.dataframe(st.session_state.df_tidy)
        
def handle_dose_response_regression():
    """Handles fitting 4PL/5PL models to the data."""
    with st.container(border=True):
        st.header("Step 2: Dose-Response Analysis", anchor="step-2-dose-response-analysis")
        if st.session_state.get('df_tidy') is None:
            st.info("Complete Step 1 to generate the Tidy Data required for this step.")
            return

        # Use a consistent key for toggles across app reruns
        perform_dr = st.toggle("Perform Dose-Response Regression?", 
                                value=st.session_state.get('perform_dr', True), # Default to True if not set
                                key="perform_dr_toggle") # Renamed key to avoid conflict with st.session_state.perform_dr assignment
        
        # The toggle automatically updates st.session_state['perform_dr_toggle'].
        # We need to map this back to st.session_state.perform_dr for consistent logic.
        st.session_state.perform_dr = perform_dr 

        if not perform_dr:
            # If DR is skipped, the modeling step should use df_tidy directly
            st.session_state.df_tidy_merged = st.session_state.df_tidy.copy() 
            st.session_state.df_5pl_results_merged = None # Explicitly None if DR is skipped
            st.session_state.df_5pl_results = None # Explicitly None if DR is skipped
            st.info("Skipping dose-response. The Tidy Data from Step 1 will be passed directly to the modeling step.")
            return
        
        # Ensure model_choice is persistent
        model_choice = st.radio("Select Model", ("5PL", "4PL"), 
                                index=st.session_state.get('dr_model_choice_idx', 0), # Default to 5PL
                                key="dr_model_choice", horizontal=True)
        st.session_state.dr_model_choice_idx = ["5PL", "4PL"].index(model_choice) # Store chosen index

        y_vars = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']
        all_cols = st.session_state.df_tidy.columns.tolist()
        
        concentration_options = [c for c in all_cols if c not in y_vars and pd.api.types.is_numeric_dtype(st.session_state.df_tidy[c])]
        grouping_options = [c for c in all_cols if c not in y_vars]
        
        if not concentration_options:
            st.error("No suitable numeric columns for 'Concentration' found in the Tidy Data.")
            return

        col1, col2, col3 = st.columns(3)
        conc_col = col1.selectbox("Concentration column:", 
                                  concentration_options, 
                                  # Use session state to retrieve the default value if available, else first option
                                  index=concentration_options.index(st.session_state.get('dr_conc_col', concentration_options[0])) if 'dr_conc_col' in st.session_state and st.session_state.dr_conc_col in concentration_options else 0,
                                  key="dr_conc_col")
        # No explicit st.session_state assignment here; selectbox handles it internally.
        
        group_by_col = col2.selectbox(
            "Group analysis by:", 
            grouping_options, 
            # Use session state to retrieve the default value if available, else first option
            index=grouping_options.index(st.session_state.get('dr_group_col', grouping_options[0])) if 'dr_group_col' in st.session_state and st.session_state.dr_group_col in grouping_options else 0,
            key="dr_group_col", 
            help="A model will be fit for each unique value in this column."
        )
        # No explicit st.session_state assignment here; selectbox handles it internally.

        y_var = col3.selectbox("Response variable (Y-axis):", 
                               y_vars, 
                               # Use session state to retrieve the default value if available, else first option (T_norm)
                               index=y_vars.index(st.session_state.get('dr_y_var', 'T_norm')) if 'dr_y_var' in st.session_state and st.session_state.dr_y_var in y_vars else y_vars.index('T_norm'),
                               key="dr_y_var")
        # No explicit st.session_state assignment here; selectbox handles it internally.
        
        if st.button("Run Dose-Response Analysis", key="run_dr_button"):
            with st.spinner("Fitting models..."):
                df = st.session_state.df_tidy.copy()
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
                        st.warning(f"Could not fit model for group '{group}': {e}")

                if results:
                    param_names = ['a', 'b', 'c', 'd', 'g'] if model_choice == "5PL" else ['a', 'b', 'c', 'd']
                    results_df = pd.DataFrame(results, columns=['Group'] + param_names + ['R-squared'])
                    st.session_state.df_5pl_results = results_df
                    # Also set the merged results here for the modeling step
                    st.session_state.df_5pl_results_merged = results_df.copy()
                    # Merged tidy data is just the tidy data if not combining with DoE factors later
                    st.session_state.df_tidy_merged = st.session_state.df_tidy.copy()
                else:
                    st.warning("Could not derive any dose-response models.")
                    st.session_state.df_5pl_results = None
                    st.session_state.df_5pl_results_merged = None # Ensure it's None if no results
                    st.session_state.df_tidy_merged = st.session_state.df_tidy.copy() # Still keep tidy data for other steps
        
        # Only show results if they exist in session state (after running DR or if already there from previous run)
        if st.session_state.get('df_5pl_results') is not None:
            results_df = st.session_state.df_5pl_results
            st.subheader("Checkpoint: Dose-Response Results")
            st.markdown("This table of model parameters will be used for DoE Modeling in Step 3.")
            st.dataframe(results_df)

            if 'c' in results_df.columns and not results_df.empty:
                best_performer = results_df.loc[results_df['c'].idxmin()]
                st.subheader("🏆 Best Performer (Lowest IC50)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Group", f"{best_performer['Group']}")
                col2.metric("IC50 (Value of 'c')", f"{best_performer['c']:.3f}")
                col3.metric("R²", f"{best_performer['R-squared']:.4f}")

            st.subheader("Dose-Response Curves")
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            df = st.session_state.df_tidy.copy()
            
            # Ensure conc_col, group_by_col, and y_var are available before plotting
            # Retrieve values directly from session_state as they are automatically managed by the selectbox keys
            if 'dr_conc_col' in st.session_state and 'dr_group_col' in st.session_state and 'dr_y_var' in st.session_state:
                conc_col_plot = st.session_state.dr_conc_col
                group_by_col_plot = st.session_state.dr_group_col
                y_var_plot = st.session_state.dr_y_var
                
                for i, group in enumerate(sorted(df[group_by_col_plot].unique())):
                    color = colors[i % len(colors)]
                    group_df_plot = df[df[group_by_col_plot] == group]
                    fig.add_trace(go.Scatter(x=group_df_plot[conc_col_plot], y=group_df_plot[y_var_plot], mode='markers', name=f'Group {group} (data)', marker=dict(color=color)))
                    if not results_df[results_df['Group'] == group].empty:
                        params = results_df[results_df['Group'] == group].iloc[0, 1:-1].values
                        x_min = 0 
                        x_max = group_df_plot[conc_col_plot].max()
                        x_range = np.linspace(x_min, x_max, 200)
                        fit_func = five_pl if model_choice == "5PL" else four_pl
                        y_pred = fit_func(x_range, *params)
                        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Group {group} (fit)', line=dict(color=color, dash='solid')))
                fig.update_layout(xaxis_title=conc_col_plot, yaxis_title=y_var_plot, title="Dose-Response Analysis", legend_title_text='Group')
                st.session_state.dose_response_fig = fig
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select concentration, grouping, and response variables to display dose-response curves.")

def handle_pycaret_modeling():
    """Handles DoE modeling using the PyCaret library."""
    with st.container(border=True):
        st.header("Step 3: DoE Modeling with PyCaret", anchor="step-3-doe-modeling-with-pycaret")
        
        base_df_tidy = st.session_state.get('df_tidy')
        base_df_params = st.session_state.get('df_5pl_results')

        if base_df_tidy is None:
            st.info("Complete Step 1 to generate the data needed for this step.")
            return

        feature_options = [] # Initialize feature_options here
        if st.session_state.data_source == 'LUMOS':
            st.subheader("1. Define & Merge DoE Factors (LUMOS Data)")
            
            grouping_col_options = base_df_tidy.columns.tolist()
            default_grouping_idx = grouping_col_options.index(st.session_state.get('pycaret_grouping_col', grouping_col_options[0])) if grouping_col_options else 0
            grouping_col = st.selectbox("Select the column that defines your experimental groups:", options=grouping_col_options, index=default_grouping_idx, key="pycaret_grouping_col")
            st.session_state.pycaret_grouping_col = grouping_col


            if grouping_col:
                unique_groups = pd.DataFrame(base_df_tidy[grouping_col].unique(), columns=[grouping_col]).sort_values(by=grouping_col).reset_index(drop=True)
                st.write(f"Enter the DoE factor values for each unique group in **'{grouping_col}'**.")
                
                num_factors = st.number_input("Number of DoE Factors", min_value=1, value=st.session_state.get('pycaret_num_factors', 1), key="pycaret_num_factors")
                st.session_state.pycaret_num_factors = num_factors # Store num_factors
                
                # Persist factor_names in session_state, initialize if number of factors changes
                if 'factor_names' not in st.session_state or len(st.session_state.factor_names) != num_factors:
                    st.session_state.factor_names = [f"DoE_Factor_{i+1}" for i in range(num_factors)]
                
                # Use current session state values for text_inputs
                factor_names = [st.text_input(f"Factor {i+1} Name", st.session_state.factor_names[i], key=f"pycaret_factor_name_{i}") for i in range(num_factors)]
                st.session_state.factor_names = factor_names # Update session state after user input

                doe_input_df = unique_groups.copy()
                for name in factor_names:
                    doe_input_df[name] = 0.0
                
                # Retrieve and update edited_doe_df from session state
                # Only update if the base structure (columns, unique groups) changes
                if ('edited_doe_df' not in st.session_state or 
                    not st.session_state.edited_doe_df.columns.equals(doe_input_df.columns) or
                    not st.session_state.edited_doe_df[grouping_col].equals(doe_input_df[grouping_col])):
                    st.session_state.edited_doe_df = doe_input_df
                
                edited_doe_df = st.data_editor(st.session_state.edited_doe_df, key="pycaret_doe_factor_editor", use_container_width=True)
                st.session_state.edited_doe_df = edited_doe_df # Store edited state back

                df_tidy_merged = pd.merge(base_df_tidy, edited_doe_df, on=grouping_col, how='left')
                st.session_state.df_tidy_merged = df_tidy_merged
                
                if base_df_params is not None:
                    base_df_params_renamed = base_df_params.rename(columns={'Group': grouping_col})
                    df_params_merged = pd.merge(base_df_params_renamed, edited_doe_df, on=grouping_col, how='left')
                    st.session_state.df_5pl_results_merged = df_params_merged
                
                st.subheader("Checkpoint: Merged Tidy Data")
                st.dataframe(st.session_state.df_tidy_merged)
                if st.session_state.get('df_5pl_results_merged') is not None:
                    st.subheader("Checkpoint: Merged Dose-Response Results")
                    st.dataframe(st.session_state.df_5pl_results_merged)
                
                feature_options = [col for col in factor_names if col in df_tidy_merged.columns] # Use merged features
        else: # Custom Data
            st.subheader("1. Custom Data for Modeling")
            st.info("For custom data, it is assumed your uploaded table is already prepared for modeling. No DoE factor merging is required.")
            st.session_state.df_tidy_merged = base_df_tidy.copy()
            # If DR was performed on custom data, use its results; otherwise, it's None and we'll use tidy data directly.
            if st.session_state.get('df_5pl_results') is not None:
                st.session_state.df_5pl_results_merged = st.session_state.df_5pl_results.copy()
            else:
                st.session_state.df_5pl_results_merged = None # No dose-response results to merge
            
            # For custom data, all numeric columns (excluding common response variables) are potential features
            if st.session_state.df_tidy_merged is not None:
                feature_options = st.session_state.df_tidy_merged.select_dtypes(include=np.number).columns.tolist()
                # Remove common response variables from initial feature options if present
                y_vars_common_from_tidy = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']
                feature_options = [f for f in feature_options if f not in y_vars_common_from_tidy]
                
        st.subheader("2. Automated Model Training with PyCaret")
        
        modeling_source_options = []
        # Decide which dataframes are available for modeling
        if st.session_state.get('df_tidy_merged') is not None and not st.session_state.df_tidy_merged.empty:
            modeling_source_options.append("Raw/Tidy Data")
        # The key to fixing the issue here: Check if df_5pl_results_merged *exists and is not empty*
        if st.session_state.get('df_5pl_results_merged') is not None and not st.session_state.df_5pl_results_merged.empty:
            modeling_source_options.append("Dose-Response Parameters")

        if not modeling_source_options:
            st.info("Please process data in previous steps to proceed with modeling.")
            return

        # Determine the index for the default selected option
        default_source_index = 0
        if 'current_modeling_source' in st.session_state and st.session_state.current_modeling_source in modeling_source_options:
            default_source_index = modeling_source_options.index(st.session_state.current_modeling_source)
        elif 'df_5pl_results_merged' in st.session_state and st.session_state.df_5pl_results_merged is not None and not st.session_state.df_5pl_results_merged.empty and "Dose-Response Parameters" in modeling_source_options:
            # If DR results are available, default to them, otherwise default to "Raw/Tidy Data"
            default_source_index = modeling_source_options.index("Dose-Response Parameters")
        else: # Default to 'Raw/Tidy Data' if no DR results or if DR was skipped
            if "Raw/Tidy Data" in modeling_source_options:
                default_source_index = modeling_source_options.index("Raw/Tidy Data")
            else:
                default_source_index = 0 # Fallback, should ideally not happen if options list is non-empty


        # UI improvement: If only one option, display it as text instead of a radio button
        if len(modeling_source_options) == 1:
            modeling_source = modeling_source_options[0]
            st.markdown(f"**Selected data source for modeling:** `{modeling_source}`")
            st.session_state.current_modeling_source = modeling_source 
            if 'pycaret_source_radio' in st.session_state: # Clear previous radio button state if it existed
                del st.session_state['pycaret_source_radio']
        else:
            modeling_source = st.radio(
                "Select data source for modeling:",
                options=modeling_source_options,
                horizontal=True,
                index=default_source_index,
                key="pycaret_source_radio"
            )
            st.session_state.current_modeling_source = modeling_source # Store selected source

        df_for_modeling = None
        # Retrieve df_for_modeling based on the current_modeling_source
        if st.session_state.current_modeling_source == "Raw/Tidy Data":
            df_for_modeling = st.session_state.get('df_tidy_merged')
        else: # "Dose-Response Parameters"
            df_for_modeling = st.session_state.get('df_5pl_results_merged')

        if df_for_modeling is None or df_for_modeling.empty:
            st.error(f"Selected modeling data source ('{st.session_state.current_modeling_source}') is not available or is empty. Please ensure data is processed correctly and contains sufficient rows.")
            return

        all_numeric_cols = df_for_modeling.select_dtypes(include=np.number).columns.tolist()
        
        # Refine feature_options based on the selected modeling_source and available columns
        if modeling_source == "Raw/Tidy Data":
            if st.session_state.data_source == 'LUMOS':
                # feature_options already derived from factor_names for LUMOS in this block
                # Ensure only numeric features are included
                feature_options = [f for f in feature_options if f in all_numeric_cols]
            else: # Custom
                # For custom, all numeric columns except the target are potential features
                feature_options = [col for col in all_numeric_cols if col not in ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']]
        else: # Dose-Response Parameters
            # For DR Parameters, all numeric columns (except R-squared and DR parameters a,b,c,d,g if they are not features) are potential features
            dr_param_cols = ['a', 'b', 'c', 'd', 'g'] # DR parameters are usually not features themselves
            feature_options = [col for col in all_numeric_cols if col not in dr_param_cols and col != 'R-squared'] # R-squared is often a target
        
        # Filter feature options to only include those present in the actual dataframe
        feature_options = [f for f in feature_options if f in df_for_modeling.columns]

        # Ensure that selected target is a valid numeric column
        if 'pycaret_target_selector' in st.session_state and st.session_state.pycaret_target_selector in all_numeric_cols:
            default_target_index = all_numeric_cols.index(st.session_state.pycaret_target_selector)
        elif len(all_numeric_cols) > 0:
            default_target_index = 0 # Default to first numeric column
        else:
            default_target_index = None # No numeric columns available

        target_col = st.selectbox(
            "Select Target (Y):", 
            options=all_numeric_cols, 
            index=default_target_index, 
            key="pycaret_target_selector"
        )
        # No explicit st.session_state assignment here; selectbox handles it internally.


        # Ensure target is not in feature options
        current_feature_options_filtered = [f for f in feature_options if f != target_col]
        
        # Preserve selected features if they are still valid options
        default_features = st.session_state.get('pycaret_features_selector', current_feature_options_filtered)
        default_features = [f for f in default_features if f in current_feature_options_filtered]
        if not default_features and current_feature_options_filtered: # If no default features but options exist
            default_features = current_feature_options_filtered

        features = st.multiselect(
            "Select Features (X):", 
            current_feature_options_filtered, 
            default=default_features, 
            key="pycaret_features_selector"
        )
        # No explicit st.session_state assignment here; multiselect handles it internally.
        
        st.session_state.modeling_df_for_opt = df_for_modeling
        st.session_state.features_for_opt = features
        st.session_state.target_for_opt = target_col

        if not target_col or not features:
            st.info("Select a target and at least one feature to begin analysis.")
            return

        st.subheader("3. PyCaret Preprocessing Options")
        col_prep1, col_prep2, col_prep3 = st.columns(3)

        with col_prep1:
            normalize = st.checkbox(
                "Normalize Data", 
                value=st.session_state.get('pycaret_normalize', True), # Default True
                key="pycaret_normalize",
                help="Scales numerical features to a standard range (e.g., between 0 and 1, or mean 0 and std 1). This helps algorithms sensitive to feature magnitudes perform better."
            )
            if normalize:
                normalize_method_options = ['zscore', 'minmax', 'maxabs', 'robust']
                normalize_method = st.selectbox(
                    "Normalization Method:",
                    options=normalize_method_options,
                    index=normalize_method_options.index(st.session_state.get('pycaret_normalize_method', 'zscore')), # Default to zscore
                    key="pycaret_normalize_method_selection", # Use a different key for selectbox itself
                    help="""
                    - **zscore**: Transforms data to have a mean of 0 and standard deviation of 1. Good for general use.
                    - **minmax**: Scales data to a range between 0 and 1. Useful for neural networks.
                    - **maxabs**: Scales data to a range between -1 and 1 by dividing by the maximum absolute value. Preserves sparsity.
                    - **robust**: Scales data using statistics that are robust to outliers (median and interquartile range), useful if your data has many outliers.
                    """
                )
                st.session_state.pycaret_normalize_method = normalize_method # Store the actual value
            else:
                normalize_method = None
                if 'pycaret_normalize_method' in st.session_state: del st.session_state['pycaret_normalize_method']
            
            remove_outliers = st.checkbox(
                "Remove Outliers (Isolation Forest)", 
                value=st.session_state.get('pycaret_remove_outliers', False), # Default False
                key="pycaret_remove_outliers",
                help="Identifies and removes data points that are significantly different from other observations using the Isolation Forest algorithm. Outliers can negatively impact model training, but removing them might discard valuable information."
            )
            
        with col_prep2:
            transformation = st.checkbox(
                "Apply Power Transformation (Numerical)", 
                value=st.session_state.get('pycaret_transformation', False), # Default False
                key="pycaret_transformation",
                help="Applies a power transformation (e.g., Box-Cox or Yeo-Johnson) to make numerical features more Gaussian-like (bell-shaped distribution). This can improve model performance, especially for linear models and neural networks."
            )
            
            remove_multicollinearity = st.checkbox(
                "Remove Multicollinearity", 
                value=st.session_state.get('pycaret_remove_multicollinearity', False), # Default False
                key="pycaret_remove_multicollinearity",
                help="Removes highly correlated (linearly dependent) features. Multicollinearity can make models unstable, difficult to interpret, and lead to overfitting."
            )
            if remove_multicollinearity:
                multicollinearity_threshold = st.slider(
                    "Multicollinearity Threshold (correlation)",
                    min_value=0.5, max_value=1.0, value=st.session_state.get('pycaret_multicollinearity_threshold', 0.9), step=0.05,
                    key="pycaret_multicollinearity_threshold",
                    help="The maximum allowed absolute correlation between features. Features with a correlation above this threshold will be considered for removal to reduce multicollinearity."
                )
            else:
                multicollinearity_threshold = None
                if 'pycaret_multicollinearity_threshold' in st.session_state: del st.session_state['pycaret_multicollinearity_threshold']


        with col_prep3:
            feature_interaction = st.checkbox(
                "Create Feature Interaction (Polynomial)", 
                value=st.session_state.get('pycaret_feature_interaction', False), # Default False
                key="pycaret_feature_interaction",
                help="Generates new features by multiplying existing numerical features (e.g., creating a 'length * width' feature). This allows the model to capture more complex non-linear relationships, but increases dimensionality."
            )
            if feature_interaction:
                polynomial_degree = st.number_input(
                    "Polynomial Degree:", 
                    min_value=2, max_value=3, value=st.session_state.get('pycaret_polynomial_degree', 2), 
                    key="pycaret_polynomial_degree",
                    help="The maximum power to which features will be raised and combined. A degree of 2 includes terms like $X_1^2$, $X_2^2$, $X_1X_2$. Higher degrees create more complex interactions."
                )
            else:
                polynomial_degree = None
                if 'pycaret_polynomial_degree' in st.session_state: del st.session_state['pycaret_polynomial_degree']
            
            feature_selection = st.checkbox(
                "Perform Feature Selection", 
                value=st.session_state.get('pycaret_feature_selection', False), # Default False
                key="pycaret_feature_selection",
                help="Automatically selects the most relevant features to improve model performance and reduce complexity. This can help prevent overfitting, especially with high-dimensional data, and speed up training."
            )

        if st.button("Run PyCaret Analysis", key="run_pycaret_button"):
            with st.spinner("Setting up PyCaret environment and comparing models..."):
                data = df_for_modeling[features + [target_col]].dropna().drop_duplicates()
                
                # Identify categorical features if any exist in the selected features
                # PyCaret's setup automatically handles numeric/categorical if not explicitly provided,
                # but explicit definition is robust.
                categorical_features = data[features].select_dtypes(include='object').columns.tolist()
                numeric_features = data[features].select_dtypes(include=np.number).columns.tolist()

                # Setup PyCaret
                s = pyreg.setup(data, target=target_col, session_id=123,
                                html=True, silent=True, log_experiment=True, experiment_name='lfa_doe',
                                # Preprocessing options
                                numeric_features=numeric_features,
                                categorical_features=categorical_features,
                                normalize=normalize,
                                normalize_method=normalize_method,
                                transformation=transformation,
                                remove_outliers=remove_outliers,
                                remove_multicollinearity=remove_multicollinearity,
                                multicollinearity_threshold=multicollinearity_threshold,
                                feature_interaction=feature_interaction, # Corrected: passed as boolean
                                # Removed polynomial_features argument as it's redundant when feature_interaction is used
                                polynomial_degree=polynomial_degree, # Corrected: passed as numerical
                                feature_selection=feature_selection,
                                # Common imputation strategies
                                numeric_imputation='mean', 
                                categorical_imputation='mode', 
                                handle_unknown_categorical=True,
                                unknown_categorical_method='most_frequent',
                                # Other useful parameters to consider enabling by default or with user options
                                ignore_low_variance=True, # Removes features with low variance (e.g., almost constant values)
                                combine_rare_levels=True, # Combines rare levels in categorical features
                                bin_numeric_features=None, # Can be ['col1', 'col2'] to bin numerical features
                                remove_perfect_collinearity=True, # Remove columns with perfect collinearity
                                )

                # Show setup output as HTML
                setup_html_path = "pycaret_setup_output.html"
                pyreg.save_html(setup_html_path)
                with open(setup_html_path, 'r', encoding='utf-8') as f:
                    html_code = f.read()
                st.subheader("PyCaret Setup Output")
                components.html(html_code, height=500, scrolling=True)

                # Compare models
                best_model = pyreg.compare_models()
                comparison_df = pyreg.pull()
                st.session_state.model_comparison_df = comparison_df

                st.subheader("Model Comparison Results")
                st.dataframe(comparison_df)
                st.session_state.best_model = best_model
            
                # Finalize model for prediction
                final_model = pyreg.finalize_model(best_model)
                st.session_state.final_model = final_model
                st.session_state.pycaret_preprocessor = pyreg.get_config('prep_pipe')

        if st.session_state.get('best_model') is not None:
            st.subheader("Interactive Model Analysis Dashboard")
            st.info("This is an interactive dashboard. Hover over plots for details and use the dropdown to explore different analyses.")
            
            with st.spinner("Generating interactive dashboard..."):
                # Generate evaluation HTML
                pyreg.evaluate_model(st.session_state.best_model, use_train_data=True)
                eval_html_path = "pycaret_evaluate_output.html"
                pyreg.save_html(eval_html_path)

                with open(eval_html_path, 'r', encoding='utf-8') as f:
                     html_code = f.read()
                components.html(html_code, height=800, scrolling=True)

            st.subheader("Response Surface Plot")
            if len(features) >= 2:
                with st.spinner("Generating response surface plot..."):
                    try:
                        # PyCaret's plot_model returns a path to the saved plot
                        plot_path = pyreg.plot_model(st.session_state.best_model, plot='surface', save=True)
                        st.session_state.rsm_fig_path = plot_path
                        st.image(plot_path)
                    except Exception as e:
                        st.warning(f"Could not generate response surface plot. This may happen for some model types. Error: {e}")
            else:
                st.info("Response surface plots require at least 2 features.")

def handle_bayesian_optimization():
    """Performs Bayesian Optimization to find optimal experimental conditions."""
    with st.container(border=True):
        st.header("Step 4: Bayesian Optimization", anchor="step-4-bayesian-optimization")

        if 'final_model' not in st.session_state or st.session_state.final_model is None:
            st.info("Complete Step 3 to train a model with PyCaret before running optimization.")
            return
        
        st.subheader("1. Optimization Settings")
        c1, c2 = st.columns(2)
        goal = c1.radio("Optimization Goal:", ("Minimize", "Maximize"), horizontal=True, key="opt_goal")
        batch_size = c2.number_input("Number of experiments to suggest:", min_value=1, max_value=10, value=3, key="batch_size")

        with st.expander("How are batch suggestions generated?"):
            st.markdown("""
            This tool uses a powerful AI-driven approach based on **Bayesian Optimization** to suggest the most informative experiments to run next. It balances two key strategies:
            - **Exploitation:** Suggesting points where the model predicts the best outcome.
            - **Exploration:** Suggesting points in regions where the model is most uncertain.
            By providing a batch of suggestions, the tool helps you efficiently learn about your experimental space and converge on the true optimum faster.
            """)

        if st.button("Suggest Next Batch of Experiments", key="run_bayes_opt"):
            with st.spinner("Searching for optimal conditions..."):
                final_model = st.session_state.final_model
                df_for_modeling = st.session_state.modeling_df_for_opt
                features = st.session_state.features_for_opt
                
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
                st.session_state.opt_result = result
                
                # Using the best model from gp_minimize's internal GPR for suggestions
                gpr = result.models[-1]
                
                def surrogate_objective(x):
                    return gpr.predict(np.array(x).reshape(1, -1))[0]
                
                bounds = [(dim.low, dim.high) for dim in result.space.dimensions]
                
                global_min_result = minimize(surrogate_objective, result.x, bounds=bounds, method='L-BFGS-B')
                global_optimum_point = global_min_result.x
                global_optimum_point_df = pd.DataFrame([global_optimum_point], columns=features)
                global_optimum_prediction = final_model.predict(global_optimum_point_df)[0]

                st.session_state.global_optimum = {
                    "point": global_optimum_point,
                    "prediction": global_optimum_prediction
                }

                suggestions = []
                # Strategy 1: Best Predicted Optimum
                suggestions.append({
                    'point': global_optimum_point,
                    'strategy': 'Exploitation (Best Predicted Optimum)'
                })

                if batch_size > 1:
                    # Generate random points in the space to evaluate for other strategies
                    grid_points = np.array(result.space.rvs(n_samples=1000, random_state=42))
                    mu, std = gpr.predict(grid_points, return_std=True)
                    y_best = result.fun 
                    
                    with np.errstate(divide='warn', invalid='ignore'):
                        imp = y_best - mu
                        Z = imp / std
                        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
                        ei[std == 0.0] = 0.0

                    # Filter out the point already suggested
                    dist_to_best = np.sqrt(np.sum((grid_points - global_optimum_point)**2, axis=1))
                    mask = dist_to_best > 0.01
                    
                    std_rem = std[mask]
                    ei_rem = ei[mask]
                    remaining_points = grid_points[mask]

                    # Strategy 2: Highest Uncertainty
                    if len(remaining_points) > 0:
                        uncertainty_idx = np.argmax(std_rem)
                        suggestions.append({'point': remaining_points[uncertainty_idx], 'strategy': 'Exploration (Highest Uncertainty)'})

                    # Strategy 3: High Expected Improvement
                    if batch_size > 2 and len(remaining_points) > 1:
                        current_suggestions_pts = np.array([s['point'] for s in suggestions])
                        ei_sorted_indices = np.argsort(ei_rem)[::-1]
                        for idx in ei_sorted_indices:
                            if len(suggestions) >= batch_size: break
                            point_to_add = remaining_points[idx]
                            
                            is_close = any(np.allclose(sug_pt, point_to_add, atol=0.01) for sug_pt in current_suggestions_pts)
                            if not is_close:
                                suggestions.append({'point': point_to_add, 'strategy': 'Balanced (High EI)'})
                
                suggestion_points = np.array([s['point'] for s in suggestions])
                suggestion_df = pd.DataFrame(suggestion_points, columns=features)
                suggestion_df['Suggestion Strategy'] = [s['strategy'] for s in suggestions]
                
                # Get predictions from the actual best model, not the GPR surrogate
                true_model_preds = final_model.predict(suggestion_df[features])
                suggestion_df['Expected Outcome'] = true_model_preds
                
                # Confidence is based on the GPR surrogate's uncertainty
                _, suggestion_std = gpr.predict(suggestion_points, return_std=True)
                max_std_overall = np.max(std) if len(std) > 0 else 0
                suggestion_df['Confidence (%)'] = (1 - (suggestion_std / max_std_overall)) * 100 if max_std_overall > 0 else 100

                st.session_state.batch_suggestions = suggestion_df


        if 'global_optimum' in st.session_state and st.session_state.global_optimum is not None:
             st.subheader(f"Global Optimum Found")
             st.metric(f"Best Predicted {st.session_state.target_for_opt}", f"{st.session_state.global_optimum['prediction']:.4f}")
        
        if 'batch_suggestions' in st.session_state and st.session_state.batch_suggestions is not None:
            st.subheader("2. Suggested Batch of Experiments")
            df_to_display = st.session_state.batch_suggestions.copy()
            st.dataframe(df_to_display.style.format({col: '{:.4f}' for col in features + ['Expected Outcome']}).format({'Confidence (%)': '{:.1f}'}))
        
        if 'opt_result' in st.session_state and st.session_state.opt_result is not None:
            st.subheader("3. Visualization of the Optimization Landscape")
            result = st.session_state.opt_result
            features = st.session_state.features_for_opt
            goal = st.session_state.get('opt_goal', 'Minimize')
            
            with st.expander("Show Diagnostic Plots", expanded=True):
                try:
                    with st.spinner("Generating diagnostic plots..."):
                        # Plot based on the GPR surrogate from gp_minimize
                        gpr = result.models[-1]
                        optimal_point = st.session_state.global_optimum['point'] 
                        n_features = len(features)

                        # Logic for 2D and >2D plots remains the same, using the GPR surrogate
                        if n_features == 2:
                             # Create a meshgrid for surface plotting
                            x_values = np.linspace(result.space.dimensions[0].low, result.space.dimensions[0].high, 50)
                            y_values = np.linspace(result.space.dimensions[1].low, result.space.dimensions[1].high, 50)
                            X_mesh, Y_mesh = np.meshgrid(x_values, y_values)
                            
                            # Predict Z values using the surrogate model
                            Z_mesh_pred = np.array([gpr.predict(np.array([x, y]).reshape(1, -1))[0] for x, y in zip(np.ravel(X_mesh), np.ravel(Y_mesh))]).reshape(X_mesh.shape)
                            
                            fig_surface = go.Figure(data=[go.Surface(z=Z_mesh_pred, x=X_mesh, y=Y_mesh, colorscale='Viridis')])
                            fig_surface.update_layout(title='Optimization Landscape (Surrogate Model)',
                                                      scene = dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=st.session_state.target_for_opt),
                                                      autosize=False,
                                                      width=800, height=600,
                                                      margin=dict(l=65, r=50, b=65, t=90))
                            st.session_state.opt_landscape_fig = fig_surface
                            st.plotly_chart(fig_surface, use_container_width=True) # Added use_container_width

                            # Partial Dependence Plots (using PyCaret's plot_model functionality if applicable, or custom)
                            # PyCaret's plot_model for 'residuals', 'error', 'boundary', 'rfe', etc. might be used here.
                            # For partial dependence, skopt provides plot_objective, but it uses matplotlib.
                            # Let's create a custom Plotly version for consistent output.
                            
                            st.write("#### Partial Dependence Plots")
                            for i, feature in enumerate(features):
                                fig_pd = go.Figure()
                                # Generate a range of values for the current feature
                                feature_range = np.linspace(result.space.dimensions[i].low, result.space.dimensions[i].high, 100)
                                
                                # Create input points by varying one feature and holding others at the optimal point
                                fixed_point = list(optimal_point)
                                predictions = []
                                for val in feature_range:
                                    temp_point = list(fixed_point)
                                    temp_point[i] = val
                                    predictions.append(gpr.predict(np.array(temp_point).reshape(1, -1))[0])
                                
                                fig_pd.add_trace(go.Scatter(x=feature_range, y=predictions, mode='lines', name=f'Partial Dependence: {feature}'))
                                fig_pd.add_vline(x=optimal_point[i], line_dash="dash", line_color="red", annotation_text="Optimal Point")
                                fig_pd.update_layout(title=f"Partial Dependence Plot for {feature}", xaxis_title=feature, yaxis_title=st.session_state.target_for_opt)
                                st.plotly_chart(fig_pd, use_container_width=True)

                        else: # For >2 features, only show partial dependence plots
                            st.write("#### Partial Dependence Plots")
                            for i, feature in enumerate(features):
                                fig_pd = go.Figure()
                                feature_range = np.linspace(result.space.dimensions[i].low, result.space.dimensions[i].high, 100)
                                
                                fixed_point = list(optimal_point)
                                predictions = []
                                for val in feature_range:
                                    temp_point = list(fixed_point)
                                    temp_point[i] = val
                                    predictions.append(gpr.predict(np.array(temp_point).reshape(1, -1))[0])
                                
                                fig_pd.add_trace(go.Scatter(x=feature_range, y=predictions, mode='lines', name=f'Partial Dependence: {feature}'))
                                fig_pd.add_vline(x=optimal_point[i], line_dash="dash", line_color="red", annotation_text="Optimal Point")
                                fig_pd.update_layout(title=f"Partial Dependence Plot for {feature}", xaxis_title=feature, yaxis_title=st.session_state.target_for_opt)
                                st.plotly_chart(fig_pd, use_container_width=True)

                        st.write("#### Convergence Trace")
                        final_model = st.session_state.final_model
                        convergence_preds = final_model.predict(pd.DataFrame(result.x_iters, columns=features))
                        if goal == "Maximize": 
                            best_seen = np.maximum.accumulate(convergence_preds)
                        else: 
                            best_seen = np.minimum.accumulate(convergence_preds)
                        
                        fig_conv = go.Figure()
                        fig_conv.add_trace(go.Scatter(x=list(range(1, len(best_seen) + 1)), y=best_seen, mode='lines+markers', line=dict(color='green')))
                        fig_conv.update_layout(title="Convergence Trace", xaxis_title="Iteration", yaxis_title=f"Best Seen {st.session_state.target_for_opt}", height=400)
                        st.session_state.opt_convergence_fig = fig_conv
                        st.plotly_chart(fig_conv, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not generate diagnostic plots: {e}")

def handle_reporting():
    """Handles the generation of a downloadable Excel report."""
    with st.container(border=True):
        st.header("Step 5: Generate and Download Report", anchor="step-5-generate-and-download-report")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate PDF Report"):
                with st.spinner("Creating PDF report..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Ensure the RSM plot path is handled correctly if it exists
                        if st.session_state.get('rsm_fig_path'):
                             shutil.copy(st.session_state.rsm_fig_path, os.path.join(temp_dir, "rsm_plot.png"))
                             st.session_state.rsm_fig_path = os.path.join(temp_dir, "rsm_plot.png")

                        pdf_path = create_pdf_report(temp_dir)
                        if pdf_path:
                            st.markdown(get_pdf_download_link(pdf_path, "LFA_Analysis_Report.pdf"), unsafe_allow_html=True)
        with c2:
            if st.button("Generate Excel Report"):
                 dfs_to_download = {
                    "Raw_Data": st.session_state.get('df_raw'),
                    "Tidy_Data": st.session_state.get('df_tidy'),
                    "Dose_Response_Results": st.session_state.get('df_5pl_results'),
                    "Model_Comparison": st.session_state.get('model_comparison_df'),
                    "Tidy_Data_Merged_DoE": st.session_state.get('df_tidy_merged'),
                    "Params_Merged_DoE": st.session_state.get('df_5pl_results_merged')
                }
                 if st.session_state.get('batch_suggestions') is not None:
                     dfs_to_download["Optimization_Suggestions"] = st.session_state.batch_suggestions

                 st.markdown(get_excel_download_link(dfs_to_download, "LFA_Analysis_Report.xlsx"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
