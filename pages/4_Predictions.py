import streamlit as st
import pandas as pd
import numpy as np
from pycaret.time_series import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Page configuration
st.set_page_config(
    page_title="Step 4: Predictions & Forecasting",
    page_icon="🔮",
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
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="step-header">🔮 Step 4: Predictions & Forecasting</h1>', unsafe_allow_html=True)

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
        <li>Generate predictions on test data</li>
        <li>Create future forecasts</li>
        <li>Visualize predictions with confidence intervals</li>
        <li>Export forecast results</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Prediction Settings")

# Model selection for predictions
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
    st.error("❌ No models available for predictions!")
    st.info("Please go back to Step 2 and create some models first.")
    st.stop()

# Model selection dropdown
model_names = [f"Model {i+1}" for i in range(len(available_models))]
if 'tuned_model' in st.session_state:
    model_names.append("Tuned Model")

selected_model_index = st.sidebar.selectbox(
    "Select Model for Predictions",
    range(len(available_models)),
    format_func=lambda x: model_names[x] if x < len(model_names) else f"Model {x+1}"
)

selected_model = available_models[selected_model_index]

# Forecast horizon
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (periods)",
    min_value=1,
    max_value=60,
    value=12,
    help="Number of periods to forecast into the future"
)

# Confidence interval
confidence_level = st.sidebar.slider(
    "Confidence Level (%)",
    min_value=80,
    max_value=99,
    value=95,
    help="Confidence level for prediction intervals"
)

# Main content
st.markdown("### 📊 Prediction Dashboard")

# Display selected model info
st.markdown(f"#### 🎯 Using Model: {type(selected_model).__name__}")
st.write(f"**Model Type:** {type(selected_model).__name__}")
st.write(f"**Model Parameters:** {selected_model}")

# Test set predictions
st.markdown("### 🎯 Test Set Predictions")

if st.button("📊 Generate Test Predictions", type="primary"):
    with st.spinner("Generating test set predictions..."):
        try:
            # Generate predictions on test set
            test_predictions = predict_model(selected_model)
            
            if not test_predictions.empty:
                st.success("✅ Test predictions generated successfully!")
                
                # Display predictions
                st.markdown("#### 📋 Test Predictions")
                st.dataframe(test_predictions, use_container_width=True)
                
                # Get actual test data
                y_test = get_config('y_test_transformed')
                
                if y_test is not None:
                    # Create comparison plot
                    fig = go.Figure()
                    
                    # Actual values
                    fig.add_trace(go.Scatter(
                        x=y_test.index,
                        y=y_test.values,
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Predicted values
                    if 'y_pred' in test_predictions.columns:
                        fig.add_trace(go.Scatter(
                            x=test_predictions.index,
                            y=test_predictions['y_pred'],
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#ff7f0e', width=2, dash='dash'),
                            marker=dict(size=4)
                        ))
                    
                    fig.update_layout(
                        title='Test Set Predictions vs Actual Values',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate prediction metrics
                    if 'y_pred' in test_predictions.columns:
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        
                        mae = mean_absolute_error(y_test, test_predictions['y_pred'])
                        mse = mean_squared_error(y_test, test_predictions['y_pred'])
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, test_predictions['y_pred'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("MAE", f"{mae:.4f}")
                        with col2:
                            st.metric("MSE", f"{mse:.4f}")
                        with col3:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col4:
                            st.metric("R²", f"{r2:.4f}")
                
                # Store predictions in session state
                st.session_state['test_predictions'] = test_predictions
                
            else:
                st.warning("⚠️ No predictions generated. Check your model and data.")
                
        except Exception as e:
            st.error(f"❌ Error generating test predictions: {str(e)}")

# Future forecasts
st.markdown("### 🔮 Future Forecasts")

if st.button("🔮 Generate Future Forecast", type="primary"):
    with st.spinner(f"Generating {forecast_horizon}-period forecast..."):
        try:
            # Generate future forecast
            future_forecast = predict_model(selected_model, fh=forecast_horizon)
            
            if not future_forecast.empty:
                st.success(f"✅ {forecast_horizon}-period forecast generated successfully!")
                
                # Display forecast
                st.markdown("#### 📋 Future Forecast")
                st.dataframe(future_forecast, use_container_width=True)
                
                # Get original data for context
                data = st.session_state.get('data')
                
                if data is not None:
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data.iloc[:, 0],
                        mode='lines+markers',
                        name='Historical Data',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Forecast data
                    if 'y_pred' in future_forecast.columns:
                        fig.add_trace(go.Scatter(
                            x=future_forecast.index,
                            y=future_forecast['y_pred'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='#ff7f0e', width=2, dash='dash'),
                            marker=dict(size=4)
                        ))
                        
                        # Add confidence intervals if available
                        if 'y_pred_lower' in future_forecast.columns and 'y_pred_upper' in future_forecast.columns:
                            fig.add_trace(go.Scatter(
                                x=future_forecast.index,
                                y=future_forecast['y_pred_upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                fillcolor='rgba(255, 127, 14, 0.2)',
                                fill='tonexty'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=future_forecast.index,
                                y=future_forecast['y_pred_lower'],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(255, 127, 14, 0.2)',
                                fill='tonexty',
                                name=f'{confidence_level}% Confidence Interval'
                            ))
                    
                    fig.update_layout(
                        title=f'{forecast_horizon}-Period Future Forecast',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store forecast in session state
                st.session_state['future_forecast'] = future_forecast
                
            else:
                st.warning("⚠️ No forecast generated. Check your model and forecast horizon.")
                
        except Exception as e:
            st.error(f"❌ Error generating future forecast: {str(e)}")

# Ensemble predictions
st.markdown("### 🤝 Ensemble Predictions")

if st.button("🤝 Generate Ensemble Predictions", type="secondary"):
    with st.spinner("Generating ensemble predictions..."):
        try:
            # Get all available models for ensemble
            ensemble_models = []
            
            if 'best_models' in st.session_state:
                if isinstance(st.session_state['best_models'], list):
                    ensemble_models.extend(st.session_state['best_models'])
                else:
                    ensemble_models.append(st.session_state['best_models'])
            
            if len(ensemble_models) > 1:
                # Create ensemble
                ensemble_model = blend_models(ensemble_models)
                
                # Generate ensemble predictions
                ensemble_predictions = predict_model(ensemble_model, fh=forecast_horizon)
                
                if not ensemble_predictions.empty:
                    st.success("✅ Ensemble predictions generated successfully!")
                    
                    # Display ensemble predictions
                    st.markdown("#### 📋 Ensemble Forecast")
                    st.dataframe(ensemble_predictions, use_container_width=True)
                    
                    # Compare with individual models
                    st.markdown("#### 📊 Ensemble vs Individual Models")
                    
                    # Create comparison plot
                    fig = go.Figure()
                    
                    # Add individual model predictions
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    for i, model in enumerate(ensemble_models[:5]):  # Limit to 5 models for clarity
                        try:
                            individual_pred = predict_model(model, fh=forecast_horizon)
                            if 'y_pred' in individual_pred.columns:
                                fig.add_trace(go.Scatter(
                                    x=individual_pred.index,
                                    y=individual_pred['y_pred'],
                                    mode='lines',
                                    name=f'{type(model).__name__}',
                                    line=dict(color=colors[i % len(colors)], width=1)
                                ))
                        except:
                            continue
                    
                    # Add ensemble prediction
                    if 'y_pred' in ensemble_predictions.columns:
                        fig.add_trace(go.Scatter(
                            x=ensemble_predictions.index,
                            y=ensemble_predictions['y_pred'],
                            mode='lines+markers',
                            name='Ensemble',
                            line=dict(color='#000000', width=3),
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        title='Ensemble vs Individual Model Predictions',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store ensemble model
                    st.session_state['ensemble_model'] = ensemble_model
                    st.session_state['ensemble_predictions'] = ensemble_predictions
                    
                else:
                    st.warning("⚠️ No ensemble predictions generated.")
            else:
                st.info("ℹ️ Need at least 2 models for ensemble predictions.")
                
        except Exception as e:
            st.error(f"❌ Error generating ensemble predictions: {str(e)}")

# Export functionality
st.markdown("### 💾 Export Results")

if st.button("💾 Export Predictions", type="secondary"):
    try:
        # Prepare data for export
        export_data = {}
        
        if 'test_predictions' in st.session_state:
            export_data['test_predictions'] = st.session_state['test_predictions']
        
        if 'future_forecast' in st.session_state:
            export_data['future_forecast'] = st.session_state['future_forecast']
        
        if 'ensemble_predictions' in st.session_state:
            export_data['ensemble_predictions'] = st.session_state['ensemble_predictions']
        
        if export_data:
            # Create combined dataframe
            combined_data = pd.DataFrame()
            
            for name, data in export_data.items():
                if not data.empty:
                    data_copy = data.copy()
                    data_copy['prediction_type'] = name
                    combined_data = pd.concat([combined_data, data_copy], ignore_index=True)
            
            # Convert to CSV
            csv = combined_data.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"time_series_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Export data prepared successfully!")
            
            # Show preview
            st.markdown("#### 📋 Export Preview")
            st.dataframe(combined_data.head(20), use_container_width=True)
            
        else:
            st.info("ℹ️ No predictions available for export. Generate some predictions first.")
            
    except Exception as e:
        st.error(f"❌ Error exporting data: {str(e)}")

# Summary statistics
if st.session_state.get('future_forecast') is not None:
    st.markdown("### 📊 Forecast Summary")
    
    forecast_data = st.session_state['future_forecast']
    
    if 'y_pred' in forecast_data.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Forecast Mean", f"{forecast_data['y_pred'].mean():.4f}")
        with col2:
            st.metric("Forecast Std", f"{forecast_data['y_pred'].std():.4f}")
        with col3:
            st.metric("Forecast Min", f"{forecast_data['y_pred'].min():.4f}")
        with col4:
            st.metric("Forecast Max", f"{forecast_data['y_pred'].max():.4f}")
        
        # Trend analysis
        st.markdown("#### 📈 Trend Analysis")
        
        # Calculate trend
        x = np.arange(len(forecast_data))
        y = forecast_data['y_pred'].values
        
        if len(y) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            
            st.write(f"**Trend Direction:** {trend_direction}")
            st.write(f"**Trend Slope:** {slope:.4f}")
            st.write(f"**Trend Strength:** {abs(slope):.4f}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <h4>✅ Predictions Complete!</h4>
        <p>You can now proceed to save and manage your models.</p>
    </div>
    """, unsafe_allow_html=True) 