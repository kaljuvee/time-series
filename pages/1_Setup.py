import streamlit as st
import pandas as pd
import numpy as np
from pycaret.datasets import get_data
from pycaret.time_series import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Step 1: Setup & Data Loading",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .step-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="step-header">🔧 Step 1: Setup & Data Loading</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
    <h3>🎯 What you'll learn in this step:</h3>
    <ul>
        <li>Load and explore the sample dataset</li>
        <li>Initialize PyCaret's time series environment</li>
        <li>Understand the setup configuration</li>
        <li>Check statistical properties of the data</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Setup Configuration")

# Data selection
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["airline", "airline_passengers", "boston", "energy", "weather"],
    help="Choose a sample dataset to work with"
)

# Forecast horizon
fh = st.sidebar.slider(
    "Forecast Horizon (periods)",
    min_value=1,
    max_value=24,
    value=3,
    help="Number of periods to forecast into the future"
)

# Session ID
session_id = st.sidebar.number_input(
    "Session ID",
    min_value=1,
    max_value=9999,
    value=123,
    help="Random seed for reproducibility"
)

# Fold strategy
fold_strategy = st.sidebar.selectbox(
    "Fold Strategy",
    ["expanding", "sliding", "temporal"],
    help="Cross-validation strategy for time series"
)

# Main content
try:
    # Load data
    st.markdown("### 📊 Loading Dataset")
    with st.spinner("Loading dataset..."):
        data = get_data(dataset_option)
    
    st.success(f"✅ Successfully loaded {dataset_option} dataset!")
    
    # Display dataset info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Dataset Preview:**")
        st.dataframe(data.head(10), use_container_width=True)
    
    with col2:
        st.markdown("**Dataset Information:**")
        st.write(f"**Shape:** {data.shape}")
        
        # Handle both Series and DataFrame cases
        if isinstance(data, pd.Series):
            st.write(f"**Variable:** {data.name if data.name else 'Unnamed'}")
            st.write(f"**Type:** Time Series (pandas Series)")
        else:
            st.write(f"**Columns:** {list(data.columns)}")
            st.write(f"**Type:** Time Series (pandas DataFrame)")
            
        st.write(f"**Date Range:** {data.index.min()} to {data.index.max()}")
        st.write(f"**Frequency:** {data.index.freq if hasattr(data.index, 'freq') else 'Not specified'}")
    
    # Plot the dataset
    st.markdown("### 📈 Dataset Visualization")
    
    fig = go.Figure()
    
    # Handle both Series and DataFrame cases
    if isinstance(data, pd.Series):
        y_values = data.values
        series_name = data.name if data.name else 'Value'
    else:
        y_values = data.iloc[:, 0].values
        series_name = data.columns[0]
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=y_values,
        mode='lines+markers',
        name=series_name,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f"{dataset_option.title()} Dataset Time Series",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    # Setup PyCaret
    st.markdown("### ⚙️ Initializing PyCaret Setup")
    
    if st.button("🚀 Initialize PyCaret Setup", type="primary"):
        with st.spinner("Setting up PyCaret environment..."):
            try:
                # Initialize setup
                s = setup(
                    data, 
                    fh=fh, 
                    session_id=session_id,
                    fold_strategy=fold_strategy,
                    verbose=False
                )
                
                st.success("✅ PyCaret setup completed successfully!")
                
                # Display setup information
                st.markdown("### 📋 Setup Configuration")
                
                # Get configuration details
                config_info = {
                    "Session ID": get_config('seed'),
                    "Forecast Horizon": fh,
                    "Fold Strategy": fold_strategy,
                    "Original Data Shape": get_config('data_before_preprocess').shape,
                    "Transformed Train Set Shape": get_config('y_train_transformed').shape,
                    "Transformed Test Set Shape": get_config('y_test_transformed').shape,
                    "Target Variable": get_config('target'),
                    "Approach": "Univariate" if len(data.columns) == 1 else "Multivariate"
                }
                
                # Display config in a nice format
                for key, value in config_info.items():
                    st.markdown(f"**{key}:** {value}")
                
                # Store setup in session state
                st.session_state['setup_completed'] = True
                st.session_state['data'] = data
                st.session_state['fh'] = fh
                st.session_state['session_id'] = session_id
                
                st.markdown("""
                <div class="success-box">
                    <h4>🎉 Setup Complete!</h4>
                    <p>Your PyCaret environment is now ready. You can proceed to the next step to compare models.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during setup: {str(e)}")
                st.info("Please check your data format and try again.")
    
    # Check stats functionality
    if st.session_state.get('setup_completed', False):
        st.markdown("### 🔍 Statistical Analysis")
        
        if st.button("📊 Run Statistical Tests"):
            with st.spinner("Running statistical tests..."):
                try:
                    # This would normally call check_stats(), but we'll create a simplified version
                    st.markdown("**Statistical Tests Results:**")
                    
                    # Basic statistical tests
                    from scipy import stats
                    
                    # Get the values for statistical tests (handle both Series and DataFrame)
                    if isinstance(data, pd.Series):
                        test_values = data.values
                    else:
                        test_values = data.iloc[:, 0].values
                    
                    # Normality test
                    stat, p_value = stats.normaltest(test_values)
                    st.write(f"**Normality Test (D'Agostino K^2):**")
                    st.write(f"  - Statistic: {stat:.4f}")
                    st.write(f"  - P-value: {p_value:.4f}")
                    st.write(f"  - Normal distribution: {'Yes' if p_value > 0.05 else 'No'}")
                    
                    # Stationarity test (ADF)
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(test_values)
                    st.write(f"**Augmented Dickey-Fuller Test:**")
                    st.write(f"  - ADF Statistic: {adf_result[0]:.4f}")
                    st.write(f"  - P-value: {adf_result[1]:.4f}")
                    st.write(f"  - Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")
                    
                except Exception as e:
                    st.error(f"Error running statistical tests: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.get('setup_completed', False):
            st.markdown("""
            <div style="text-align: center;">
                <h4>✅ Ready for the next step!</h4>
                <p>Setup is complete. You can now proceed to model comparison.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please complete the setup to proceed to the next step.")

except Exception as e:
    st.error(f"❌ Error loading dataset: {str(e)}")
    st.info("Please make sure PyCaret is properly installed: `pip install pycaret[full]`") 