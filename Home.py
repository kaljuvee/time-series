import streamlit as st
import pandas as pd
from pycaret.datasets import get_data

# Page configuration
st.set_page_config(
    page_title="PyCaret Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .feature-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 PyCaret Time Series Forecasting Tutorial</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="step-card">
    <h2>🎯 What is PyCaret?</h2>
    <p>PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. 
    It's an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle 
    and makes you more productive.</p>
    
    <p>Compared with other open-source machine learning libraries, PyCaret is an alternate low-code library that can 
    be used to replace hundreds of lines of code with a few lines only. This makes experiments exponentially fast 
    and efficient.</p>
</div>
""", unsafe_allow_html=True)

# Key Features
st.markdown("### 🚀 Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🔧 Low-Code</h4>
        <p>Replace hundreds of lines of code with just a few lines</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>⚡ Fast & Efficient</h4>
        <p>Exponentially speed up your experiment cycle</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>🎯 End-to-End</h4>
        <p>Complete ML pipeline from data prep to deployment</p>
    </div>
    """, unsafe_allow_html=True)

# Workflow Steps
st.markdown("### 📋 Tutorial Workflow")
st.markdown("""
This tutorial will guide you through a complete time series forecasting workflow using PyCaret. 
Follow these steps in order:
""")

# Step cards
steps = [
    {
        "number": "1",
        "title": "Setup & Data Loading",
        "description": "Initialize the training environment and load sample data",
        "page": "pages/1_Setup.py"
    },
    {
        "number": "2", 
        "title": "Model Comparison",
        "description": "Compare multiple forecasting models and select the best one",
        "page": "pages/2_Model_Comparison.py"
    },
    {
        "number": "3",
        "title": "Model Analysis & Visualization", 
        "description": "Analyze model performance with various plots and diagnostics",
        "page": "pages/3_Model_Analysis.py"
    },
    {
        "number": "4",
        "title": "Predictions & Forecasting",
        "description": "Generate predictions and future forecasts",
        "page": "pages/4_Predictions.py"
    },
    {
        "number": "5",
        "title": "Model Management",
        "description": "Save, load, and deploy your trained models",
        "page": "pages/5_Model_Management.py"
    }
]

for step in steps:
    st.markdown(f"""
    <div class="step-card">
        <h3>Step {step['number']}: {step['title']}</h3>
        <p>{step['description']}</p>
        <a href="{step['page']}" target="_self">
            <button style="background-color: #1f77b4; color: white; padding: 0.5rem 1rem; border: none; border-radius: 5px; cursor: pointer;">
                Go to Step {step['number']}
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Sample Data Preview
st.markdown("### 📊 Sample Dataset Preview")
st.markdown("We'll be using the Airline dataset throughout this tutorial. Here's a preview:")

try:
    # Load sample data
    data = get_data('airline')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(data.head(10), use_container_width=True)
    
    with col2:
        st.markdown("**Dataset Info:**")
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Columns:** {list(data.columns)}")
        st.write(f"**Date Range:** {data.index.min()} to {data.index.max()}")
        
        # Basic statistics
        st.markdown("**Basic Statistics:**")
        st.write(data.describe())
        
except Exception as e:
    st.error(f"Error loading sample data: {str(e)}")
    st.info("Please install PyCaret first: `pip install pycaret[full]`")

# Installation Instructions
st.markdown("### 💻 Installation")
st.markdown("""
If you haven't installed PyCaret yet, you can install it using pip:

```bash
pip install pycaret[full]
```

**System Requirements:**
- Python 3.7 – 3.10
- Python 3.9 for Ubuntu only
- Ubuntu 16.04 or later
- Windows 7 or later
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit and PyCaret</p>
    <p>Follow the steps above to get started with time series forecasting!</p>
</div>
""", unsafe_allow_html=True) 