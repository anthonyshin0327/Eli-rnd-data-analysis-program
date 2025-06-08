import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from io import BytesIO
import base64
import warnings
import itertools
import matplotlib.pyplot as plt
import shap

# --- Model Imports ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- Page Configuration ---
st.set_page_config(
    page_title="LFA Analysis & ML Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Suppress Warnings for Cleaner Output ---
warnings.filterwarnings("ignore")

# --- Helper Functions ---

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
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Full Report (XLSX)</a>'
    return href


# --- Main App ---
def main():
    """Main function to run the Streamlit application."""
    with st.sidebar:
        st.title("🔬 LFA Analysis & ML Suite")
        st.markdown("---")
        
        st.header("1. Data Upload")
        keys_to_init = [
            'data_source', 'df_raw', 'df_tidy', 'df_5pl_results',
            'df_doe_factors', 'df_tidy_doe', 'df_5pl_results_doe',
            'gpr_model', 'rsm_model', 'rf_model', 'doe_features', 'doe_model_r2',
            'y_scaler', 'x_preprocessor', 'target_col', 'performance_metrics',
            'X_train', 'y_train', 'final_doe_df', 'df_tidy_merged', 'df_5pl_results_merged'
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
            try:
                for key in keys_to_init:
                    if key == 'data_source':
                        continue
                    st.session_state[key] = None
                
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df_raw = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df_raw = pd.read_excel(uploaded_file)
                st.sidebar.success("File uploaded successfully!")

                if st.session_state.data_source == 'Custom':
                    st.session_state.df_tidy = st.session_state.df_raw.copy()
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.df_raw = None
        
        st.markdown("---")
        st.header("2. Navigation")
        st.markdown("""
        * [Data Preprocessing](#step-1-data-preprocessing)
        * [Dose-Response Analysis](#step-2-dose-response-analysis)
        * [DoE Modeling & Visualization](#step-3-doe-modeling-visualization)
        * [Bayesian Optimization](#step-4-bayesian-optimization)
        * [Download Report](#step-5-generate-and-download-report)
        """)

    st.title("LFA Data Analysis and Machine Learning Pipeline")

    if st.session_state.df_raw is not None:
        handle_lumos_processing()
        handle_dose_response_regression()
        handle_doe_modeling()
        handle_bayesian_optimization()
        handle_reporting()
    else:
        st.info("Awaiting data file upload in the sidebar...")

def handle_lumos_processing():
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
    with st.container(border=True):
        st.header("Step 2: Dose-Response Analysis", anchor="step-2-dose-response-analysis")
        if st.session_state.get('df_tidy') is None:
            st.info("Complete Step 1 to generate the Tidy Data required for this step.")
            return

        perform_dr = st.toggle("Perform Dose-Response Regression?", value=True, key="perform_dr")
        
        if not perform_dr:
            st.session_state.df_5pl_results_doe = st.session_state.df_tidy.copy()
            st.info("Skipping dose-response. The Tidy Data from Step 1 will be passed directly to the modeling step.")
            st.session_state.df_5pl_results = None
            return
        
        model_choice = st.radio("Select Model", ("5PL", "4PL"), key="dr_model_choice", horizontal=True)
        
        y_vars = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T', 'T/C', 'T+C']
        all_cols = st.session_state.df_tidy.columns
        group_cols = [c for c in all_cols if c not in y_vars and pd.api.types.is_numeric_dtype(st.session_state.df_tidy[c])]
        
        if not group_cols:
            st.error("No suitable numeric grouping or concentration columns found in the Tidy Data.")
            return

        col1, col2, col3 = st.columns(3)
        conc_col = col1.selectbox("Concentration column:", group_cols, key="dr_conc_col")
        group_by_col = col2.selectbox("Group analysis by:", group_cols, key="dr_group_col", help="A model will be fit for each unique value in this column.")
        y_var = col3.selectbox("Response variable (Y-axis):", y_vars, key="dr_y_var", index=2)
        
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
                st.session_state.df_5pl_results_doe = results_df.copy()
            else:
                st.warning("Could not derive any dose-response models.")
                st.session_state.df_5pl_results = None
                st.session_state.df_5pl_results_doe = None

        if st.session_state.get('df_5pl_results') is not None:
            results_df = st.session_state.df_5pl_results
            st.subheader("Checkpoint: Dose-Response Results")
            st.markdown("This table of model parameters will be used for DoE Modeling in Step 3.")
            st.dataframe(results_df)

            if 'c' in results_df.columns and not results_df.empty:
                best_performer = results_df.loc[results_df['c'].idxmin()]
                st.markdown("""
                <style>
                div[data-testid="stMetric"] {
                    background-color: #e8f5e9;
                    border: 1px solid #4CAF50;
                    padding: 1rem;
                    border-radius: 0.5rem;
                }
                </style>""", unsafe_allow_html=True)
                st.subheader("🏆 Best Performer (Lowest IC50)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Group", f"{best_performer['Group']}")
                col2.metric("IC50 (Value of 'c')", f"{best_performer['c']:.3f}")
                col3.metric("R²", f"{best_performer['R-squared']:.4f}")

            st.subheader("Dose-Response Curves")
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, group in enumerate(sorted(df[group_by_col].unique())):
                color = colors[i % len(colors)]
                group_df_plot = df[df[group_by_col] == group]
                fig.add_trace(go.Scatter(x=group_df_plot[conc_col], y=group_df_plot[y_var], mode='markers', name=f'Group {group} (data)', marker=dict(color=color)))
                if not results_df[results_df['Group'] == group].empty:
                    params = results_df[results_df['Group'] == group].iloc[0, 1:-1].values
                    x_min = 0 
                    x_max = group_df_plot[conc_col].max()
                    x_range = np.linspace(x_min, x_max, 200)
                    fit_func = five_pl if model_choice == "5PL" else four_pl
                    y_pred = fit_func(x_range, *params)
                    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Group {group} (fit)', line=dict(color=color, dash='solid')))
            fig.update_layout(xaxis_title=conc_col, yaxis_title=y_var, title="Dose-Response Analysis", legend_title_text='Group')
            st.plotly_chart(fig, use_container_width=True)

def handle_doe_modeling():
    with st.container(border=True):
        st.header("Step 3: DoE Modeling & Visualization", anchor="step-3-doe-modeling-visualization")
        
        base_df_tidy = st.session_state.get('df_tidy')
        base_df_params = st.session_state.get('df_5pl_results')

        if base_df_tidy is None:
            st.info("Complete Step 1 to generate the data needed for this step.")
            return

        st.subheader("1. Define & Merge DoE Factors")
        
        grouping_col = st.selectbox("Select the column that defines your experimental groups:", options=base_df_tidy.columns, index=0)

        if grouping_col:
            unique_groups = pd.DataFrame(base_df_tidy[grouping_col].unique(), columns=[grouping_col]).sort_values(by=grouping_col).reset_index(drop=True)
            st.write(f"Enter the DoE factor values for each unique group in **'{grouping_col}'**.")
            
            num_factors = st.number_input("Number of DoE Factors", min_value=1, value=1, key="ml_num_factors")
            factor_names = [st.text_input(f"Factor {i+1} Name", f"DoE_Factor_{i+1}", key=f"ml_factor_name_{i}") for i in range(num_factors)]

            doe_input_df = unique_groups.copy()
            for name in factor_names:
                doe_input_df[name] = 0.0
            
            edited_doe_df = st.data_editor(doe_input_df, key="doe_factor_editor", use_container_width=True)
            
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

        st.subheader("2. Model Training & Interpretation")
        
        df_for_modeling = st.session_state.get('df_5pl_results_merged') if st.session_state.get('df_5pl_results_merged') is not None and not st.session_state.get('df_5pl_results_merged').empty else st.session_state.get('df_tidy_merged')

        if df_for_modeling is None:
            st.info("Please define and merge DoE factors above to proceed with modeling.")
            return
            
        all_numeric_cols = df_for_modeling.select_dtypes(include=np.number).columns.tolist()
        
        target_col = st.selectbox("Select Target (Y):", all_numeric_cols, index=len(all_numeric_cols)-2 if 'R-squared' in all_numeric_cols else 0, key="target_col_selector")
        features = st.multiselect("Select Features (X):", [col for col in factor_names if col in df_for_modeling.columns], default=[col for col in factor_names if col in df_for_modeling.columns], key="features_selector")

        if not target_col or not features:
            st.info("Select a target and at least one feature to begin analysis.")
            return

        with st.spinner("Training models and generating analyses..."):
            X = df_for_modeling[features].dropna().drop_duplicates()
            if X.empty:
                st.warning("No data available for modeling after removing duplicates and missing values.")
                return
                
            y = df_for_modeling.loc[X.index, target_col]
            
            poly = PolynomialFeatures(degree=len(features) if len(features) > 1 else 1, interaction_only=True, include_bias=False)
            scaler = StandardScaler()
            preprocessor = Pipeline([('poly', poly), ('scaler', scaler)])
            X_processed = preprocessor.fit_transform(X)
            poly_feature_names = preprocessor.named_steps['poly'].get_feature_names_out(features)

            models = {
                "Linear Model (RSM)": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gaussian Process (GPR)": GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0) + WhiteKernel(0.1), random_state=42, n_restarts_optimizer=10)
            }
            
            results = {}
            # --- CHANGE: Add robust check for cross-validation ---
            min_cv_samples = 5 
            can_cv = len(X) >= min_cv_samples

            for name, model in models.items():
                model.fit(X_processed, y)
                y_pred_full = model.predict(X_processed)
                full_fit_r2 = r2_score(y, y_pred_full)
                
                if can_cv:
                    cv_scores = cross_val_score(model, X_processed, y, cv=min_cv_samples, scoring='r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean, cv_std = np.nan, np.nan

                results[name] = {"model": model, "full_fit_r2": full_fit_r2, "cv_mean_r2": cv_mean, "cv_std_r2": cv_std}
            
            st.subheader("Model Performance Comparison")
            # --- CHANGE: Add explainer for CV ---
            with st.expander("A Note on Model Validation (Full Fit vs. Cross-Validation)"):
                st.markdown("""
                In machine learning, we want to know how well our model will perform on *new, unseen data*.
                
                * **Full Fit R²**: This score shows how well the model fits the data it was trained on. A high score means the model learned the patterns in your current data very well. For sparse lab data, this is often the primary metric of interest to understand the relationships in the experiment.
                * **Cross-Validated (CV) R²**: This is a more robust estimate of how the model will perform on *new* data. It works by splitting your data into several "folds" (e.g., 5), training the model on 4 folds, and testing it on the 1 fold it hasn't seen. This process is repeated 5 times, and the scores are averaged.
                
                **Why does CV sometimes fail or give a low score?**
                With very small datasets (like in many DoE studies), there isn't enough data for this splitting process. The "test" set in each fold might be just one or two points, making the score unreliable. If CV was skipped, it's because the dataset was too small for a meaningful validation.
                """)

            if not can_cv:
                st.warning(f"Cross-validation was skipped because the number of unique data points ({len(X)}) is less than the required minimum of {min_cv_samples}.")

            perf_df = pd.DataFrame({"Model": results.keys(), "Full Fit R²": [r['full_fit_r2'] for r in results.values()], "CV Mean R²": [r['cv_mean_r2'] for r in results.values()], "CV R² Std Dev": [r['cv_std_r2'] for r in results.values()]}).set_index("Model")
            st.dataframe(perf_df.style.format("{:.4f}", na_rep="N/A"))

            # --- CHANGE: Base best model on Full Fit R² ---
            best_model_name = perf_df['Full Fit R²'].idxmax()
            st.success(f"🏆 **{best_model_name}** is the best model based on **Full Fit R²**.")
            best_model = results[best_model_name]['model']

            st.subheader("Feature Importance (SHAP)")
            try:
                with st.spinner("Calculating SHAP values..."):
                    # For GPR, KernelExplainer can be slow. A sample can speed it up.
                    X_summary = shap.sample(X_processed, 100) if len(X_processed) > 100 else X_processed
                    explainer = shap.KernelExplainer(best_model.predict, X_summary)
                    shap_values = explainer.shap_values(X_processed)
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                    shap.summary_plot(shap_values, features=X_processed, feature_names=poly_feature_names, show=False, plot_size=None)
                    plt.tight_layout()
                    st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"Could not generate SHAP plot for {best_model_name}: {e}")

            st.subheader("Interpretable Decision Tree Surrogate")
            with st.spinner("Training and plotting surrogate decision tree..."):
                surrogate_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
                y_pred_best_model = best_model.predict(X_processed)
                surrogate_tree.fit(X_processed, y_pred_best_model)
                fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                plot_tree(surrogate_tree, feature_names=poly_feature_names, filled=True, rounded=True, ax=ax_tree, fontsize=10)
                plt.title(f"Decision Tree Approximating the '{best_model_name}' Model")
                st.pyplot(fig_tree)

def handle_bayesian_optimization():
    with st.container(border=True):
        st.header("Step 4: Bayesian Optimization", anchor="step-4-bayesian-optimization")
        st.info("This step is under construction.")
        
def handle_reporting():
    with st.container(border=True):
        st.header("Step 5: Generate and Download Report", anchor="step-5-generate-and-download-report")
        
        if st.button("Generate Download Link"):
            dfs_to_download = {
                "Raw_Data": st.session_state.get('df_raw'),
                "Tidy_Data": st.session_state.get('df_tidy'),
                "Dose_Response_Results": st.session_state.get('df_5pl_results'),
                "Tidy_Data_Merged_DoE": st.session_state.get('df_tidy_merged'),
                "Params_Merged_DoE": st.session_state.get('df_5pl_results_merged')
            }
            st.markdown(get_excel_download_link(dfs_to_download, "LFA_Analysis_Report.xlsx"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
