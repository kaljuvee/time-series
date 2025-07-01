import streamlit as st
import pandas as pd
import numpy as np
from pycaret.time_series import *
import os
import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Step 5: Model Management",
    page_icon="💾",
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
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="step-header">💾 Step 5: Model Management</h1>', unsafe_allow_html=True)

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
        <li>Save trained models to disk</li>
        <li>Load models from disk</li>
        <li>Deploy models to cloud platforms</li>
        <li>Manage model versions and experiments</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Sidebar for configuration
st.sidebar.header("⚙️ Model Management Settings")

# Model selection for management
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

if 'ensemble_model' in st.session_state:
    available_models.append(st.session_state['ensemble_model'])

# Add individual models if they exist
for model_name in ['ets', 'arima', 'theta', 'naive', 'snaive']:
    if f'{model_name}_model' in st.session_state:
        available_models.append(st.session_state[f'{model_name}_model'])

if not available_models:
    st.error("❌ No models available for management!")
    st.info("Please go back to Step 2 and create some models first.")
    st.stop()

# Model selection dropdown
model_names = [f"Model {i+1}" for i in range(len(available_models))]
if 'tuned_model' in st.session_state:
    model_names.append("Tuned Model")
if 'ensemble_model' in st.session_state:
    model_names.append("Ensemble Model")

selected_model_index = st.sidebar.selectbox(
    "Select Model for Management",
    range(len(available_models)),
    format_func=lambda x: model_names[x] if x < len(model_names) else f"Model {x+1}"
)

selected_model = available_models[selected_model_index]

# Main content
st.markdown("### 📊 Model Management Dashboard")

# Display selected model info
st.markdown(f"#### 🎯 Managing: {type(selected_model).__name__}")
st.write(f"**Model Type:** {type(selected_model).__name__}")
st.write(f"**Model Parameters:** {selected_model}")

# Save Model Section
st.markdown("### 💾 Save Model")

# Model name input
model_name = st.text_input(
    "Model Name",
    value=f"{type(selected_model).__name__}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    help="Enter a name for your model"
)

# Model description
model_description = st.text_area(
    "Model Description",
    placeholder="Enter a description of your model, its performance, and use case...",
    help="Optional description of the model"
)

# Save options
save_options = st.multiselect(
    "Save Options",
    ["Save Model", "Save Experiment", "Save Pipeline"],
    default=["Save Model"],
    help="Choose what to save"
)

if st.button("💾 Save Model", type="primary"):
    with st.spinner("Saving model..."):
        try:
            saved_files = []
            
            if "Save Model" in save_options:
                # Save the model
                model_path = f"models/{model_name}.pkl"
                save_model(selected_model, model_path)
                saved_files.append(model_path)
                
                # Save model metadata
                metadata = {
                    "model_name": model_name,
                    "model_type": type(selected_model).__name__,
                    "description": model_description,
                    "created_date": datetime.datetime.now().isoformat(),
                    "model_parameters": str(selected_model)
                }
                
                metadata_path = f"models/{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(metadata_path)
            
            if "Save Experiment" in save_options:
                # Save experiment
                experiment_path = f"models/{model_name}_experiment"
                save_experiment(experiment_path)
                saved_files.append(experiment_path)
            
            if "Save Pipeline" in save_options:
                # Save pipeline
                pipeline_path = f"models/{model_name}_pipeline.pkl"
                save_model(selected_model, pipeline_path)
                saved_files.append(pipeline_path)
            
            st.success(f"✅ Model saved successfully!")
            st.write("**Saved files:**")
            for file_path in saved_files:
                st.write(f"- {file_path}")
            
            # Store in session state
            st.session_state['saved_models'] = st.session_state.get('saved_models', [])
            st.session_state['saved_models'].append({
                'name': model_name,
                'path': model_path if "Save Model" in save_options else None,
                'description': model_description,
                'date': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")

# Load Model Section
st.markdown("### 📂 Load Model")

# Check for saved models
saved_models = []
if os.path.exists('models'):
    for file in os.listdir('models'):
        if file.endswith('.pkl') and not file.endswith('_pipeline.pkl'):
            model_name_from_file = file.replace('.pkl', '')
            saved_models.append(model_name_from_file)

if saved_models:
    selected_saved_model = st.selectbox(
        "Select Model to Load",
        saved_models,
        help="Choose a saved model to load"
    )
    
    if st.button("📂 Load Selected Model", type="secondary"):
        with st.spinner("Loading model..."):
            try:
                # Load the model
                model_path = f"models/{selected_saved_model}.pkl"
                loaded_model = load_model(model_path)
                
                st.success(f"✅ Model '{selected_saved_model}' loaded successfully!")
                
                # Display loaded model info
                st.markdown("#### 📋 Loaded Model Information")
                st.write(f"**Model Type:** {type(loaded_model).__name__}")
                st.write(f"**Model Parameters:** {loaded_model}")
                
                # Load metadata if available
                metadata_path = f"models/{selected_saved_model}_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    st.markdown("#### 📝 Model Metadata")
                    st.write(f"**Description:** {metadata.get('description', 'No description available')}")
                    st.write(f"**Created Date:** {metadata.get('created_date', 'Unknown')}")
                
                # Store loaded model in session state
                st.session_state['loaded_model'] = loaded_model
                st.session_state['loaded_model_name'] = selected_saved_model
                
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
else:
    st.info("ℹ️ No saved models found. Save some models first.")

# Model Finalization
st.markdown("### 🎯 Finalize Model")

st.markdown("""
Finalizing a model trains it on the entire dataset (including the hold-out set) to create 
the best possible model for production use.
""")

if st.button("🎯 Finalize Model", type="primary"):
    with st.spinner("Finalizing model..."):
        try:
            # Finalize the model
            finalized_model = finalize_model(selected_model)
            
            st.success("✅ Model finalized successfully!")
            
            # Display finalized model info
            st.markdown("#### 📋 Finalized Model Information")
            st.write(f"**Model Type:** {type(finalized_model).__name__}")
            st.write(f"**Model Parameters:** {finalized_model}")
            
            # Store finalized model in session state
            st.session_state['finalized_model'] = finalized_model
            
            st.markdown("""
            <div class="success-box">
                <h4>🎉 Model Ready for Production!</h4>
                <p>Your finalized model is now ready for deployment and production use.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error finalizing model: {str(e)}")

# Model Deployment Section
st.markdown("### 🚀 Model Deployment")

st.markdown("""
PyCaret supports deploying models to various cloud platforms. Here are the available options:
""")

# Deployment options
deployment_platform = st.selectbox(
    "Deployment Platform",
    ["Local", "AWS S3", "Google Cloud Platform", "Microsoft Azure"],
    help="Choose where to deploy your model"
)

if deployment_platform != "Local":
    st.info(f"ℹ️ {deployment_platform} deployment requires additional setup and credentials.")
    
    if deployment_platform == "AWS S3":
        st.markdown("""
        **AWS S3 Setup Requirements:**
        - AWS Access Key ID
        - AWS Secret Access Key
        - Default Region Name
        - S3 Bucket Name
        
        Configure using: `aws configure`
        """)
    
    elif deployment_platform == "Google Cloud Platform":
        st.markdown("""
        **GCP Setup Requirements:**
        - GCP Project ID
        - Service Account Key (JSON file)
        - Storage Bucket Name
        
        Set environment variables for authentication.
        """)
    
    elif deployment_platform == "Microsoft Azure":
        st.markdown("""
        **Azure Setup Requirements:**
        - Azure Storage Account
        - Connection String
        - Container Name
        
        Set AZURE_STORAGE_CONNECTION_STRING environment variable.
        """)

# Local deployment
if deployment_platform == "Local":
    deployment_name = st.text_input(
        "Deployment Name",
        value=f"{model_name}_deployment",
        help="Name for the deployment"
    )
    
    if st.button("🚀 Deploy Model Locally", type="primary"):
        with st.spinner("Deploying model locally..."):
            try:
                # Create deployment directory
                deployment_dir = f"deployments/{deployment_name}"
                os.makedirs(deployment_dir, exist_ok=True)
                
                # Save model for deployment
                deployment_path = f"{deployment_dir}/model.pkl"
                save_model(selected_model, deployment_path)
                
                # Create deployment configuration
                deployment_config = {
                    "deployment_name": deployment_name,
                    "model_type": type(selected_model).__name__,
                    "deployment_date": datetime.datetime.now().isoformat(),
                    "model_path": deployment_path,
                    "description": model_description
                }
                
                config_path = f"{deployment_dir}/deployment_config.json"
                with open(config_path, 'w') as f:
                    json.dump(deployment_config, f, indent=2)
                
                st.success(f"✅ Model deployed locally to {deployment_dir}")
                
                # Create simple prediction script
                prediction_script = f"""
# Prediction script for {deployment_name}
import pandas as pd
from pycaret.time_series import load_model

def predict(data, fh=1):
    '''
    Make predictions using the deployed model
    
    Parameters:
    data: Input time series data
    fh: Forecast horizon
    
    Returns:
    Predictions
    '''
    model = load_model('{deployment_path}')
    predictions = model.predict(fh=fh)
    return predictions

# Example usage:
# predictions = predict(your_data, fh=12)
"""
                
                script_path = f"{deployment_dir}/predict.py"
                with open(script_path, 'w') as f:
                    f.write(prediction_script)
                
                st.write("**Deployment files created:**")
                st.write(f"- {deployment_path}")
                st.write(f"- {config_path}")
                st.write(f"- {script_path}")
                
            except Exception as e:
                st.error(f"❌ Error deploying model: {str(e)}")

# Model Registry
st.markdown("### 📚 Model Registry")

# Display saved models
if st.session_state.get('saved_models'):
    st.markdown("#### 📋 Saved Models")
    
    for i, model_info in enumerate(st.session_state['saved_models']):
        with st.expander(f"📦 {model_info['name']} - {model_info['date'][:10]}"):
            st.write(f"**Description:** {model_info['description']}")
            st.write(f"**Saved Date:** {model_info['date']}")
            st.write(f"**Path:** {model_info['path']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🗑️ Delete {model_info['name']}", key=f"delete_{i}"):
                    try:
                        if model_info['path'] and os.path.exists(model_info['path']):
                            os.remove(model_info['path'])
                        st.success(f"✅ {model_info['name']} deleted successfully!")
                        st.session_state['saved_models'].pop(i)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting model: {str(e)}")
            
            with col2:
                if st.button(f"📂 Load {model_info['name']}", key=f"load_{i}"):
                    try:
                        loaded_model = load_model(model_info['path'])
                        st.session_state['loaded_model'] = loaded_model
                        st.session_state['loaded_model_name'] = model_info['name']
                        st.success(f"✅ {model_info['name']} loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <h4>🎉 Tutorial Complete!</h4>
        <p>Congratulations! You've successfully completed the PyCaret Time Series Forecasting tutorial.</p>
        <p>You now have a complete understanding of time series forecasting with PyCaret.</p>
    </div>
    """, unsafe_allow_html=True) 