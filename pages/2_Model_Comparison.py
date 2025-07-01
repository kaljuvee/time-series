import streamlit as st
import pandas as pd
import numpy as np
from pycaret.time_series import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Step 2: Model Comparison",
    page_icon="🏆",
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
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="step-header">🏆 Step 2: Model Comparison</h1>', unsafe_allow_html=True)

# Check if setup is completed
if not st.session_state.get('setup_completed', False):
    st.error("❌ Please complete Step 1 (Setup) first!")
    st.info("Go back to Step 1 and initialize the PyCaret setup before proceeding.")
    st.stop()

# Introduction
st.markdown("""
<div class="info-box">
    <h3>🎯 What you'll learn in this step:</h3>
    <ul>
        <li>Compare multiple forecasting models</li>
        <li>Understand different model performance metrics</li>
        <li>Select the best performing model</li>
        <li>Explore model-specific parameters</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Model Comparison Settings")

# Model selection
st.sidebar.markdown("### 📊 Model Selection")
include_all = st.sidebar.checkbox("Include All Models", value=True, help="Compare all available models")

if not include_all:
    available_models = [
        'ets', 'arima', 'theta', 'naive', 'snaive', 'grand_means', 'polytrend',
        'dt_cds_dt', 'dt_cds_dt_cds', 'dt_cds_dt_cds_cds', 'dt_cds_dt_cds_cds_cds'
    ]
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        available_models,
        default=['ets', 'arima', 'theta', 'naive', 'snaive'],
        help="Choose specific models to compare"
    )
else:
    selected_models = None

# Number of top models to select
n_select = st.sidebar.slider(
    "Number of Top Models",
    min_value=1,
    max_value=5,
    value=1,
    help="Number of best models to return"
)

# Sort metric
sort_metric = st.sidebar.selectbox(
    "Sort by Metric",
    ["MAE", "MSE", "RMSE", "MAPE", "MASE", "R2"],
    help="Metric to use for ranking models"
)

# Cross-validation settings
fold = st.sidebar.slider(
    "Cross-validation Folds",
    min_value=2,
    max_value=10,
    value=3,
    help="Number of CV folds"
)

# Main content
st.markdown("### 🚀 Model Comparison")

# Available models info
st.markdown("#### 📋 Available Models")
st.markdown("""
PyCaret's time series module includes over 30 algorithms comprising statistical/time-series methods 
as well as machine learning based models. Here are some key categories:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="model-card">
        <h4>📈 Statistical Models</h4>
        <ul>
            <li>ETS (Error, Trend, Seasonality)</li>
            <li>ARIMA</li>
            <li>Theta</li>
            <li>Naive</li>
            <li>Seasonal Naive</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <h4>🤖 Machine Learning</h4>
        <ul>
            <li>Decision Trees</li>
            <li>Random Forest</li>
            <li>XGBoost</li>
            <li>LightGBM</li>
            <li>Neural Networks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="model-card">
        <h4>📊 Baseline Models</h4>
        <ul>
            <li>Grand Means</li>
            <li>Polynomial Trend</li>
            <li>Moving Average</li>
            <li>Exponential Smoothing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Model comparison button
if st.button("🏁 Start Model Comparison", type="primary"):
    with st.spinner("Comparing models... This may take a few minutes."):
        try:
            # Compare models
            if selected_models:
                best_models = compare_models(
                    include=selected_models,
                    n_select=n_select,
                    sort=sort_metric,
                    fold=fold
                )
            else:
                best_models = compare_models(
                    n_select=n_select,
                    sort=sort_metric,
                    fold=fold
                )
            
            st.success("✅ Model comparison completed successfully!")
            
            # Store results in session state
            st.session_state['best_models'] = best_models
            st.session_state['comparison_completed'] = True
            
            # Display results
            st.markdown("### 📊 Comparison Results")
            
            # Get the comparison results
            comparison_results = pull()
            st.dataframe(comparison_results, use_container_width=True)
            
            # Show best model info
            if isinstance(best_models, list):
                st.markdown(f"### 🏆 Top {len(best_models)} Models Selected")
                for i, model in enumerate(best_models, 1):
                    st.markdown(f"**{i}. {type(model).__name__}**")
                    st.write(f"Model parameters: {model}")
            else:
                st.markdown("### 🏆 Best Model Selected")
                st.markdown(f"**Model:** {type(best_models).__name__}")
                st.write(f"Model parameters: {best_models}")
            
            # Performance visualization
            st.markdown("### 📈 Performance Comparison")
            
            # Create performance comparison chart
            if not comparison_results.empty:
                # Select key metrics for visualization
                metrics_to_plot = ['MAE', 'MSE', 'RMSE', 'MAPE']
                available_metrics = [m for m in metrics_to_plot if m in comparison_results.columns]
                
                if available_metrics:
                    # Prepare data for plotting
                    plot_data = comparison_results[available_metrics].reset_index()
                    plot_data = plot_data.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
                    
                    # Create bar chart
                    fig = px.bar(
                        plot_data,
                        x='Model',
                        y='Score',
                        color='Metric',
                        title='Model Performance Comparison',
                        barmode='group'
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="success-box">
                <h4>🎉 Comparison Complete!</h4>
                <p>You can now proceed to analyze the selected model(s) in detail.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error during model comparison: {str(e)}")
            st.info("Please check your setup and try again.")

# Individual model creation
if st.session_state.get('comparison_completed', False):
    st.markdown("### 🔧 Individual Model Creation")
    
    st.markdown("""
    You can also create and train individual models with specific parameters:
    """)
    
    # Model selection for individual creation
    individual_model = st.selectbox(
        "Select Model to Create",
        ['ets', 'arima', 'theta', 'naive', 'snaive', 'grand_means', 'polytrend'],
        help="Choose a specific model to create and train"
    )
    
    if st.button(f"🔨 Create {individual_model.upper()} Model"):
        with st.spinner(f"Creating {individual_model} model..."):
            try:
                # Create individual model
                individual_model_obj = create_model(individual_model, fold=fold)
                
                st.success(f"✅ {individual_model.upper()} model created successfully!")
                
                # Display model results
                model_results = pull()
                st.dataframe(model_results, use_container_width=True)
                
                # Store in session state
                st.session_state[f'{individual_model}_model'] = individual_model_obj
                
            except Exception as e:
                st.error(f"❌ Error creating {individual_model} model: {str(e)}")

# Model tuning section
if st.session_state.get('comparison_completed', False):
    st.markdown("### 🎛️ Model Tuning")
    
    st.markdown("""
    You can tune the hyperparameters of your models for better performance:
    """)
    
    # Select model to tune
    models_to_tune = []
    if 'best_models' in st.session_state:
        if isinstance(st.session_state['best_models'], list):
            models_to_tune = st.session_state['best_models']
        else:
            models_to_tune = [st.session_state['best_models']]
    
    if models_to_tune:
        tune_model_choice = st.selectbox(
            "Select Model to Tune",
            [f"Model {i+1}" for i in range(len(models_to_tune))],
            help="Choose which model to tune"
        )
        
        if st.button("🎛️ Tune Selected Model"):
            with st.spinner("Tuning model hyperparameters..."):
                try:
                    model_index = int(tune_model_choice.split()[-1]) - 1
                    model_to_tune = models_to_tune[model_index]
                    
                    # Tune the model
                    tuned_model = tune_model(model_to_tune, optimize=sort_metric)
                    
                    st.success("✅ Model tuning completed!")
                    st.write(f"Tuned model: {tuned_model}")
                    
                    # Store tuned model
                    st.session_state['tuned_model'] = tuned_model
                    
                except Exception as e:
                    st.error(f"❌ Error tuning model: {str(e)}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.session_state.get('comparison_completed', False):
        st.markdown("""
        <div style="text-align: center;">
            <h4>✅ Ready for the next step!</h4>
            <p>Model comparison is complete. You can now proceed to model analysis and visualization.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Please complete the model comparison to proceed to the next step.") 