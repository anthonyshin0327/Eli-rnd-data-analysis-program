import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from scipy.optimize import curve_fit
from io import BytesIO
import base64
import re
import warnings

# --- Model Imports for Deep Dive ---
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


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
        val = d + (a - d) / (1 + np.exp(b * (np.log(x) - np.log(c))))**g
    return val

def four_pl(x, a, b, c, d):
    """4-Parameter Logistic Regression model function."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = d + (a - d) / (1 + np.exp(b * (np.log(x) - np.log(c))))
    return val


def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_excel_download_link(dfs_dict, filename):
    """Generates a link to download a multi-sheet Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs_dict.items():
            if df is not None:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
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
        st.header("Workflow Control")

        # Initialize session state
        for key in ['data_source', 'df_raw', 'df_tidy', 'df_5pl_results', 'df_doe_factors', 'df_tidy_doe', 'df_5pl_results_doe', 'automl_models']:
            if key not in st.session_state:
                st.session_state[key] = None

        st.session_state.data_source = st.radio(
            "1. Choose your data source", ('LUMOS', 'Custom'),
            help="Select 'LUMOS' for standard files or 'Custom' for other formats."
        )
        uploaded_file = st.file_uploader("2. Upload your data file", type=["csv"])

        if uploaded_file:
            try:
                st.session_state.df_raw = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.df_raw = None
        
        st.markdown("---")
        st.info("This app provides a streamlined workflow for analyzing Lateral Flow Assay (LFA) data, from initial data processing to advanced machine learning.")

    st.title("LFA Data Analysis and Machine Learning Pipeline")
    st.markdown("Follow the steps below to analyze your data. Options will appear as you complete each stage.")

    if st.session_state.df_raw is not None:
        st.success("File uploaded successfully!")
        with st.expander("Show Uploaded Data Preview"):
            st.dataframe(st.session_state.df_raw.head())

        st.header("Step 1: Data Preprocessing")
        if st.session_state.data_source == 'LUMOS':
            handle_lumos_processing()
        else:
            st.info("Custom data loaded. Proceed to Exploratory Data Analysis.")
            st.session_state.df_tidy = st.session_state.df_raw.copy()
            st.dataframe(st.session_state.df_tidy.head())

        if st.session_state.df_tidy is not None:
            if st.session_state.data_source == 'LUMOS':
                with st.expander("Step 2: Dose-Response Analysis (Optional)"):
                    handle_dose_response_regression()
            with st.expander("Step 3: Exploratory Data Analysis (EDA)", expanded=True):
                handle_eda()
            with st.expander("Step 4: Automated Machine Learning & Optimization", expanded=True):
                handle_ml()
            with st.expander("Step 5: AI-Generated Conclusion"):
                 st.warning("AI-Generated Conclusion feature is under development.")
            with st.expander("Step 6: Generate and Download Report"):
                handle_reporting()
    else:
        st.info("Awaiting data file upload...")

def handle_lumos_processing():
    """Manages the processing pipeline for LUMOS data."""
    st.subheader("LUMOS Data Processing")
    df_raw = st.session_state.df_raw
    required_cols = ['strip name', 'line_peak_above_background_1', 'line_peak_above_background_2']
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"LUMOS file must contain the following columns: {', '.join(required_cols)}")
        return

    first_strip_name = df_raw['strip name'].iloc[0]
    delimiter = '-' if '-' in first_strip_name else '_' if '_' in first_strip_name else None
    if not delimiter:
        st.error("Could not auto-detect delimiter ('-' or '_') in 'strip name' column.")
        return

    st.success(f"Detected delimiter: '{delimiter}'")
    num_groups = len(first_strip_name.split(delimiter))
    st.markdown(f"Detected **{num_groups}** groups in 'strip name'. Please name them:")
    group_names = [st.text_input(f"Group {i+1} Name", f"Factor_{i+1}") for i in range(num_groups)]

    if st.button("Process LUMOS Data"):
        df = df_raw.copy()
        try:
            split_cols = df['strip name'].str.split(delimiter, expand=True)
            if split_cols.shape[1] == len(group_names):
                split_cols.columns = group_names
                df = pd.concat([split_cols, df], axis=1)
                for col in group_names:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            else:
                st.error("Mismatch between detected groups and provided names.")
                return
        except Exception as e:
            st.error(f"Error splitting columns: {e}")
            return

        df = df.rename(columns={'line_peak_above_background_1': 'T', 'line_peak_above_background_2': 'C'})
        df['T+C'] = df['T'] + df['C']
        df['T_norm'] = df['T'].divide(df['T+C']).fillna(0)
        df['C_norm'] = df['C'].divide(df['T+C']).fillna(0)
        df['T-C'] = df['T'] - df['C']
        df['C/T'] = df['C'].divide(df['T']).fillna(0).replace([np.inf, -np.inf], 0)
        final_cols = group_names + ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T']
        st.session_state.df_tidy = df[final_cols]
        st.subheader("Processed Tidy Data")
        st.dataframe(st.session_state.df_tidy)
        st.success("LUMOS data processed successfully!")

def handle_dose_response_regression():
    """Manages the 4PL/5PL regression analysis section."""
    if st.session_state.df_tidy is None:
        st.info("Process LUMOS data first to enable this section.")
        return

    df = st.session_state.df_tidy.copy()
    if not st.checkbox("Perform Dose-Response Regression?"):
        return

    model_choice = st.radio("Select Regression Model", ("5PL", "4PL"), help="5PL includes an asymmetry parameter 'g'; 4PL is symmetrical (g=1).")
    
    y_vars = ['T', 'C', 'T_norm', 'C_norm', 'T-C', 'C/T']
    group_cols = [col for col in df.columns if col not in y_vars]
    conc_col = st.selectbox("Select the analyte concentration column:", group_cols)
    group_by_col = st.selectbox("Select a column to group the analysis by:", group_cols) # Made mandatory for DoE
    y_var_to_plot = st.selectbox("Select the response variable (Y-axis) to analyze:", y_vars)
    
    df[conc_col] = pd.to_numeric(df[conc_col], errors='coerce')
    df.dropna(subset=[conc_col, y_var_to_plot], inplace=True)
    if df.empty:
        st.warning(f"No valid data in '{conc_col}' or '{y_var_to_plot}' columns.")
        return

    st.markdown(f"### {model_choice} Regression Results")
    results = []
    groups_to_iterate = df[group_by_col].unique()
    overfitting_warning = False

    for group in groups_to_iterate:
        group_df = df[df[group_by_col] == group]
        X_orig, y = group_df[conc_col], group_df[y_var_to_plot]
        if len(X_orig) < 5: continue

        X_fit = X_orig.copy().replace(0, 1e-9)
        try:
            non_zero_X = X_orig[X_orig > 0]
            median_X = np.median(non_zero_X) if not non_zero_X.empty else 1.0
            
            if model_choice == "5PL":
                p0 = [np.min(y), 1, median_X, np.max(y), 1]
                params, _ = curve_fit(five_pl, X_fit, y, p0=p0, maxfev=10000, bounds=(-np.inf, np.inf))
                if 0.95 < params[4] < 1.05: # Check if g is close to 1
                    overfitting_warning = True
            else: # 4PL
                p0 = [np.min(y), 1, median_X, np.max(y)]
                params, _ = curve_fit(four_pl, X_fit, y, p0=p0, maxfev=10000, bounds=(-np.inf, np.inf))

            target_func = five_pl if model_choice == "5PL" else four_pl
            y_pred = target_func(X_fit, *params)
            r2 = r2_score(y, y_pred)
            results.append([group] + list(params) + [r2])

        except (RuntimeError, ValueError) as e:
            st.warning(f"Could not fit {model_choice} model for group '{group}': {e}")
    
    if overfitting_warning:
        st.info("💡 Suggestion: The asymmetry parameter 'g' in one or more of your 5PL fits is close to 1. A 4PL model might provide a simpler and equally effective fit.")

    if not results:
        st.warning("No models were successfully fitted.")
        return

    # Define columns based on model choice
    if model_choice == "5PL":
        res_cols = ['Group', 'a (Min Y)', 'b (Steepness)', 'c (IC50)', 'd (Max Y)', 'g (Asymmetry)', 'R-squared']
    else:
        res_cols = ['Group', 'a (Min Y)', 'b (Steepness)', 'c (IC50)', 'd (Max Y)', 'R-squared']

    results_df = pd.DataFrame(results, columns=res_cols)
    st.session_state.df_5pl_results = results_df.copy()
    st.subheader("Fitted Model Parameters")
    st.dataframe(results_df)

    use_log_scale = st.checkbox("Use Logarithmic Scale for X-axis", True)
    fig = go.Figure()
    for group_name in groups_to_iterate:
        group_df = df[df[group_by_col] == group_name]
        X_plot, y_plot = group_df[conc_col], group_df[y_var_to_plot]
        fig.add_trace(go.Scatter(x=X_plot, y=y_plot, mode='markers', name=f'Raw - {group_name}'))
        
        if group_name in results_df['Group'].values:
            fit_params = results_df[results_df['Group'] == group_name].iloc[0, 1:-1].values
            X_fit_plot = X_plot.copy().replace(0, 1e-9)
            if not X_fit_plot.empty:
                x_min_log, x_max_log = np.log10(X_fit_plot.min()), np.log10(X_fit_plot.max())
                x_range = np.logspace(x_min_log, x_max_log, 100)
                
                plot_func = five_pl if model_choice == '5PL' else four_pl
                y_fit = plot_func(x_range, *fit_params)
                fig.add_trace(go.Scatter(x=x_range, y=y_fit, mode='lines', name=f'Fit - {group_name}'))
    
    fig.update_layout(title=f"{model_choice} Fit for {y_var_to_plot} vs {conc_col}", xaxis_title=conc_col, yaxis_title=y_var_to_plot, xaxis_type="log" if use_log_scale else "linear")
    st.plotly_chart(fig, use_container_width=True)

    # --- DoE Factor Definition ---
    st.subheader("Define DoE Factors for Analysis")
    st.info("Define the experimental factors corresponding to each group. This creates an enriched dataset for ML.")
    
    num_doe_factors = st.number_input("How many DoE factors to define?", min_value=1, value=2, step=1)
    doe_factor_names = [st.text_input(f"Name for DoE Factor {i+1}", f"DoE_Factor_{i+1}") for i in range(num_doe_factors)]
    
    if 'df_5pl_results' in st.session_state and st.session_state.df_5pl_results is not None:
        doe_df_template = pd.DataFrame(columns=doe_factor_names, index=st.session_state.df_5pl_results['Group'].unique())
        st.write("Enter the values for each factor for each group:")
        edited_doe_df = st.data_editor(doe_df_template)
        
        if st.button("Merge DoE Factors into Datasets"):
            st.session_state.df_doe_factors = edited_doe_df.copy()
            st.session_state.df_doe_factors['Group'] = st.session_state.df_doe_factors.index
            
            st.session_state.df_5pl_results_doe = pd.merge(st.session_state.df_5pl_results, st.session_state.df_doe_factors, on='Group', how='left')
            st.subheader("Dose-Response Results with DoE Factors")
            st.dataframe(st.session_state.df_5pl_results_doe)

            st.session_state.df_tidy_doe = pd.merge(st.session_state.df_tidy, st.session_state.df_doe_factors, left_on=group_by_col, right_on='Group', how='left')
            st.subheader("Tidy Data with DoE Factors")
            st.dataframe(st.session_state.df_tidy_doe.head())

            st.success("DoE factors successfully merged!")


def handle_eda():
    """Manages the exploratory data analysis section."""
    st.subheader("Descriptive Statistics")
    st.dataframe(st.session_state.df_tidy.describe(include='all'))
    st.subheader("Visualizations")
    all_cols = st.session_state.df_tidy.columns.tolist()
    numerical_cols = st.session_state.df_tidy.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = st.session_state.df_tidy.select_dtypes(include=['object', 'category']).columns.tolist()
    if not numerical_cols:
        st.warning("No numerical data to visualize.")
        return
    
    plot_options = ["Histogram", "Box Plot", "Heatmap", "3D Scatter Plot", "Violin Plot", "Scatter Matrix", "Density Contour", "Sunburst Chart"]
    plot_type = st.selectbox("Choose a plot type:", plot_options)

    if plot_type == "Histogram":
        col_to_plot = st.selectbox("Select numerical column:", numerical_cols)
        color_col = st.selectbox("Color by (optional):", [None] + categorical_cols, key='hist_color')
        st.plotly_chart(px.histogram(st.session_state.df_tidy, x=col_to_plot, color=color_col, title=f"Histogram of {col_to_plot}"), use_container_width=True)
    elif plot_type == "Box Plot":
        y_col, x_col = st.selectbox("Y-axis (numerical):", numerical_cols, key='box_y'), st.selectbox("X-axis:", all_cols, key='box_x')
        color_col = st.selectbox("Color by (optional):", [None] + categorical_cols, key='box_color')
        st.plotly_chart(px.box(st.session_state.df_tidy, x=x_col, y=y_col, color=color_col, title=f"Box Plot of {y_col} by {x_col}"), use_container_width=True)
    elif plot_type == "Heatmap":
        st.plotly_chart(px.imshow(st.session_state.df_tidy[numerical_cols].corr(), text_auto=True, title="Correlation Heatmap"), use_container_width=True)
    # ... other plots follow a similar pattern ...

def handle_ml():
    """Manages the machine learning modeling section."""
    st.subheader("Automated Model Benchmarking")
    st.info("This module uses LazyPredict to quickly compare dozens of models.")
    
    # --- Data Selection for ML ---
    ml_options = ["Measurement Data (Tidy)"]
    if hasattr(st.session_state, 'df_tidy_doe') and st.session_state.df_tidy_doe is not None: ml_options.append("Measurement Data with DoE Factors")
    if st.session_state.df_5pl_results is not None: ml_options.append("Dose-Response Results (Grouped)")
    if hasattr(st.session_state, 'df_5pl_results_doe') and st.session_state.df_5pl_results_doe is not None: ml_options.append("Dose-Response Results with DoE Factors")
    
    ml_choice = st.radio("Choose data for ML analysis:", ml_options)

    analysis_df = None
    if ml_choice == "Measurement Data (Tidy)": analysis_df = st.session_state.df_tidy
    elif ml_choice == "Measurement Data with DoE Factors": analysis_df = st.session_state.df_tidy_doe
    elif ml_choice == "Dose-Response Results (Grouped)": analysis_df = st.session_state.df_5pl_results
    else: analysis_df = st.session_state.df_5pl_results_doe
    
    if analysis_df is None:
        st.warning("Selected dataset is not available yet. Please complete the required steps.")
        return

    st.dataframe(analysis_df.head())
    all_cols = analysis_df.columns.tolist()
    target_col = st.selectbox("Select the target (output) variable (Y):", all_cols)
    
    if not target_col: return
        
    feature_cols = st.multiselect("Select the feature (input) variables (X):", [c for c in all_cols if c != target_col], default=[c for c in all_cols if c != target_col and analysis_df[c].dtype != 'object'])

    if not feature_cols:
        st.warning("Please select at least one feature variable.")
        return

    X = analysis_df[feature_cols]
    y = analysis_df[target_col]
    
    is_classification = pd.api.types.is_string_dtype(y) or pd.api.types.is_categorical_dtype(y) or (pd.api.types.is_integer_dtype(y) and y.nunique() < 25)
    st.write(f"**Analysis Mode:** {'Classification' if is_classification else 'Regression'}")
    
    # Define preprocessor outside the button so it's available for the deep dive
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])

    if st.button("Run AutoML Analysis"):
        with st.spinner("Training and evaluating models..."):
            test_size_adj = 0.2
            if len(X) < 10:
                st.warning("Warning: The dataset is very small. Test set size adjusted.")
                test_size_adj = 2 if len(X) > 2 else 1
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_adj, random_state=42)
            
            if is_classification:
                clf = LazyClassifier(verbose=0, ignore_warnings=True, preprocessor=preprocessor)
                models, _ = clf.fit(X_train, X_test, y_train, y_test)
                st.subheader("LazyPredict Classification Results")
            else:
                X_train_processed, X_test_processed = preprocessor.fit_transform(X_train), preprocessor.transform(X_test)
                reg = LazyRegressor(verbose=0, ignore_warnings=True)
                models, _ = reg.fit(X_train_processed, X_test_processed, y_train, y_test)
                st.subheader("LazyPredict Regression Results")

            st.session_state.automl_models = models
            st.dataframe(models)
            st.success("AutoML analysis complete!")
            st.balloons()
    
    # --- Model Deep Dive & Optimization ---
    if st.session_state.automl_models is not None:
        st.markdown("---")
        st.subheader("Model Deep Dive & Optimization")
        
        MODEL_MAP = {
            "AdaBoostRegressor": AdaBoostRegressor, "BaggingRegressor": BaggingRegressor, "DecisionTreeRegressor": DecisionTreeRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor, "GaussianProcessRegressor": GaussianProcessRegressor, 
            "GradientBoostingRegressor": GradientBoostingRegressor, "KNeighborsRegressor": KNeighborsRegressor, 
            "LGBMRegressor": LGBMRegressor, "RandomForestRegressor": RandomForestRegressor, "XGBRegressor": XGBRegressor,
            "BayesianRidge": BayesianRidge, "ElasticNet": ElasticNet, "Lasso": Lasso, "Ridge": Ridge, "SGDRegressor": SGDRegressor,
        }
        
        model_name = st.selectbox("Select a model for deep dive:", st.session_state.automl_models.index)
        model_class = MODEL_MAP.get(model_name)

        if model_class:
            st.markdown("#### Predict Optimal Conditions")
            optimization_goal = st.radio("Optimization Goal", ("Minimize", "Maximize"), horizontal=True)

            search_space = {}
            for col in numerical_features:
                min_val, max_val = float(X[col].min()), float(X[col].max())
                search_space[col] = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
            
            for col in categorical_features:
                search_space[col] = st.multiselect(f"Values for {col}", options=X[col].unique(), default=X[col].unique())
            
            if st.button("Find Optimum"):
                with st.spinner("Searching for optimal combination..."):
                    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_class())])
                    full_pipeline.fit(X, y)

                    grid_values = [np.linspace(search_space[c][0], search_space[c][1], 10) if c in numerical_features else search_space[c] for c in X.columns]
                    
                    grid = pd.DataFrame(np.array(np.meshgrid(*grid_values)).T.reshape(-1, len(X.columns)), columns=X.columns)
                    
                    predictions = full_pipeline.predict(grid)
                    grid['predicted_target'] = predictions

                    optimum_idx = grid['predicted_target'].idxmax() if optimization_goal == "Maximize" else grid['predicted_target'].idxmin()
                    optimum_combination = grid.loc[optimum_idx]

                    st.metric(label=f"Optimal Predicted {target_col}", value=f"{optimum_combination['predicted_target']:.4f}")
                    st.write("Using combination:")
                    st.json(optimum_combination.drop('predicted_target').to_dict())

                    fig = px.parallel_coordinates(grid, color="predicted_target", 
                                                  color_continuous_scale=px.colors.sequential.Viridis,
                                                  title="Optimization Search Results")
                    st.plotly_chart(fig)
        else:
            st.warning("Selected model is not yet available for a deep dive.")

def handle_reporting():
    """Manages the report generation and download section."""
    st.subheader("Download Center")
    st.info("Download your processed data and results as a multi-sheet Excel file.")
    
    dfs_to_download = {
        "Raw_Data": st.session_state.df_raw,
        "Tidy_Data": st.session_state.df_tidy,
        "Dose_Response_Results": st.session_state.df_5pl_results,
        "Tidy_Data_with_DoE": st.session_state.df_tidy_doe,
        "Dose_Response_with_DoE": st.session_state.df_5pl_results_doe,
        "AutoML_Results": st.session_state.automl_models
    }
    
    st.markdown(get_excel_download_link(dfs_to_download, "LFA_Analysis_Report.xlsx"), unsafe_allow_html=True)
    st.warning("PDF and PowerPoint downloads are under development.")


if __name__ == "__main__":
    main()
