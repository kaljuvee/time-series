import streamlit as st
import pandas as pd
import numpy as np
from pycaret.time_series import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Step 3: Model Analysis & Visualization",
    page_icon="📊",
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
    .plot-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="step-header">📊 Step 3: Model Analysis & Visualization</h1>', unsafe_allow_html=True)

# Check if previous steps are completed
if not st.session_state.get('setup_completed', False):
    st.error("❌ Please complete Step 1 (Setup) first!")
    st.info("Go back to Step 1 and initialize the PyCaret setup before proceeding.")
    st.stop()

if not st.session_state.get('comparison_completed', False):
    st.error("❌ Please complete Step 2 (Model Comparison) first!")
    st.info("Go back to Step 2 and complete the model comparison before proceeding.")
    st.stop()

# Introduction
st.markdown("""
<div class="info-box">
    <h3>🎯 What you'll learn in this step:</h3>
    <ul>
        <li>Analyze model performance with various plots</li>
        <li>Understand forecast accuracy and residuals</li>
        <li>Explore time series diagnostics</li>
        <li>Visualize model predictions vs actual values</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Analysis Settings")

# Model selection for analysis
st.sidebar.markdown("### 🎯 Model Selection")
available_models = []

# Get available models from session state
if 'best_models' in st.session_state:
    if isinstance(st.session_state['best_models'], list):
        available_models = st.session_state['best_models']
    else:
        available_models = [st.session_state['best_models']]

if 'tuned_model' in st.session_state:
    available_models.append(st.session_state['tuned_model'])

# Add individual models if they exist
for model_name in ['ets', 'arima', 'theta', 'naive', 'snaive']:
    if f'{model_name}_model' in st.session_state:
        available_models.append(st.session_state[f'{model_name}_model'])

if not available_models:
    st.error("❌ No models available for analysis!")
    st.info("Please go back to Step 2 and create some models first.")
    st.stop()

# Model selection dropdown
model_names = [f"Model {i+1}" for i in range(len(available_models))]
if 'tuned_model' in st.session_state:
    model_names.append("Tuned Model")

selected_model_index = st.sidebar.selectbox(
    "Select Model for Analysis",
    range(len(available_models)),
    format_func=lambda x: model_names[x] if x < len(model_names) else f"Model {x+1}"
)

selected_model = available_models[selected_model_index]

# Forecast horizon for analysis
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon for Analysis",
    min_value=1,
    max_value=36,
    value=12,
    help="Number of periods to forecast for analysis plots"
)

# Main content
st.markdown("### 📈 Model Analysis Dashboard")

# Display selected model info
st.markdown(f"#### 🎯 Analyzing: {type(selected_model).__name__}")
st.write(f"**Model Type:** {type(selected_model).__name__}")
st.write(f"**Model Parameters:** {selected_model}")

# Analysis plots section
st.markdown("### 📊 Analysis Plots")

# Plot selection
plot_options = {
    "Forecast Plot": "forecast",
    "Residuals Plot": "residuals", 
    "Diagnostics Plot": "diagnostics",
    "ACF Plot": "acf",
    "PACF Plot": "pacf",
    "Decomposition Plot": "decomp",
    "Decomp Plot": "decomp_stl",
    "Periodogram": "periodogram",
    "Cross Validation": "cv",
    "Prediction Error": "prediction_error"
}

selected_plots = st.multiselect(
    "Select Plots to Generate",
    list(plot_options.keys()),
    default=["Forecast Plot", "Residuals Plot"],
    help="Choose which analysis plots to generate"
)

# Generate plots
if st.button("📊 Generate Analysis Plots", type="primary"):
    with st.spinner("Generating analysis plots..."):
        try:
            for plot_name in selected_plots:
                plot_type = plot_options[plot_name]
                
                st.markdown(f"#### {plot_name}")
                
                try:
                    # Generate plot
                    if plot_type in ['forecast', 'residuals']:
                        # These plots require a trained model
                        fig = plot_model(selected_model, plot=plot_type, return_fig=True)
                        
                        # Convert matplotlib figure to plotly
                        if hasattr(fig, 'savefig'):
                            # Save to bytes and convert
                            import io
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            
                            # Display as image
                            st.image(buf, use_column_width=True)
                        else:
                            st.pyplot(fig)
                            
                    elif plot_type in ['acf', 'diagnostics']:
                        # These plots don't require a trained model
                        fig = plot_model(plot=plot_type, return_fig=True)
                        
                        if hasattr(fig, 'savefig'):
                            import io
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            st.image(buf, use_column_width=True)
                        else:
                            st.pyplot(fig)
                    
                    st.success(f"✅ {plot_name} generated successfully!")
                    
                except Exception as plot_error:
                    st.error(f"❌ Error generating {plot_name}: {str(plot_error)}")
                    st.info(f"This plot type might not be available for the selected model or data.")
                
                st.markdown("---")
            
        except Exception as e:
            st.error(f"❌ Error during plot generation: {str(e)}")

# Custom forecast visualization
st.markdown("### 🎨 Custom Forecast Visualization")

if st.button("🔮 Generate Custom Forecast", type="secondary"):
    with st.spinner("Generating custom forecast..."):
        try:
            # Generate forecast
            forecast_data = predict_model(selected_model, fh=forecast_horizon)
            
            # Get original data
            data = st.session_state.get('data')
            
            if data is not None and not forecast_data.empty:
                # Create custom forecast plot
                fig = go.Figure()
                
                # Original data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.iloc[:, 0],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                
                # Forecast data
                if 'y_pred' in forecast_data.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data['y_pred'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        marker=dict(size=4)
                    ))
                
                fig.update_layout(
                    title=f'Custom Forecast - {type(selected_model).__name__}',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast data
                st.markdown("#### 📋 Forecast Data")
                st.dataframe(forecast_data, use_container_width=True)
                
                st.success("✅ Custom forecast generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating custom forecast: {str(e)}")

# Model evaluation metrics
st.markdown("### 📊 Model Evaluation Metrics")

if st.button("📈 Calculate Model Metrics", type="secondary"):
    with st.spinner("Calculating model metrics..."):
        try:
            # Get model performance metrics
            from pycaret.time_series import get_metrics
            
            metrics = get_metrics()
            
            if metrics is not None:
                st.markdown("#### 📋 Performance Metrics")
                st.dataframe(metrics, use_container_width=True)
                
                # Create metrics visualization
                if not metrics.empty:
                    # Select numeric columns for visualization
                    numeric_cols = metrics.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 0:
                        # Create radar chart for metrics
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=metrics[numeric_cols].iloc[0].values,
                            theta=numeric_cols,
                            fill='toself',
                            name='Model Performance'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, metrics[numeric_cols].max().max()]
                                )),
                            showlegend=True,
                            title="Model Performance Radar Chart"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Model metrics calculated successfully!")
            else:
                st.info("No metrics available for the selected model.")
                
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")

# Residuals analysis
st.markdown("### 🔍 Residuals Analysis")

if st.button("🔍 Analyze Residuals", type="secondary"):
    with st.spinner("Analyzing residuals..."):
        try:
            # Get residuals
            residuals = get_config('y_test_transformed') - predict_model(selected_model)
            
            if residuals is not None and len(residuals) > 0:
                # Create residuals analysis plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Residuals vs fitted
                axes[0, 0].scatter(predict_model(selected_model), residuals)
                axes[0, 0].axhline(y=0, color='r', linestyle='--')
                axes[0, 0].set_xlabel('Fitted Values')
                axes[0, 0].set_ylabel('Residuals')
                axes[0, 0].set_title('Residuals vs Fitted')
                
                # Residuals histogram
                axes[0, 1].hist(residuals, bins=20, alpha=0.7)
                axes[0, 1].set_xlabel('Residuals')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Residuals Distribution')
                
                # Q-Q plot
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot')
                
                # Residuals over time
                axes[1, 1].plot(residuals.index, residuals.values)
                axes[1, 1].axhline(y=0, color='r', linestyle='--')
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title('Residuals Over Time')
                
                plt.tight_layout()
                
                # Display plot
                st.pyplot(fig)
                
                # Residuals statistics
                st.markdown("#### 📊 Residuals Statistics")
                residuals_stats = {
                    "Mean": np.mean(residuals),
                    "Std": np.std(residuals),
                    "Min": np.min(residuals),
                    "Max": np.max(residuals),
                    "Skewness": stats.skew(residuals),
                    "Kurtosis": stats.kurtosis(residuals)
                }
                
                for stat, value in residuals_stats.items():
                    st.write(f"**{stat}:** {value:.4f}")
                
                st.success("✅ Residuals analysis completed!")
                
        except Exception as e:
            st.error(f"❌ Error analyzing residuals: {str(e)}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <h4>✅ Analysis Complete!</h4>
        <p>You can now proceed to generate predictions and forecasts.</p>
    </div>
    """, unsafe_allow_html=True) 