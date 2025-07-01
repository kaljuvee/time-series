# 📈 PyCaret Time Series Forecasting Tutorial

A comprehensive Streamlit application that guides you through the complete time series forecasting workflow using PyCaret. This interactive tutorial covers everything from data setup to model deployment.

## 🎯 What is PyCaret?

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It's an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes you more productive.

### Key Features:
- **🔧 Low-Code**: Replace hundreds of lines of code with just a few lines
- **⚡ Fast & Efficient**: Exponentially speed up your experiment cycle
- **🎯 End-to-End**: Complete ML pipeline from data prep to deployment
- **📊 30+ Algorithms**: Statistical and machine learning models
- **🤖 Automated**: Hyperparameter tuning, ensembling, and model selection

## 🚀 Getting Started

### Prerequisites

- Python 3.7 – 3.10
- Python 3.9 for Ubuntu only
- Ubuntu 16.04 or later / Windows 7 or later

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd time-series
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run Home.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Tutorial Structure

The application is organized into 5 logical steps:

### Step 1: Setup & Data Loading 🔧
- Load and explore sample datasets
- Initialize PyCaret's time series environment
- Understand setup configuration
- Check statistical properties of the data

### Step 2: Model Comparison 🏆
- Compare multiple forecasting models
- Understand different performance metrics
- Select the best performing model
- Explore model-specific parameters

### Step 3: Model Analysis & Visualization 📊
- Analyze model performance with various plots
- Understand forecast accuracy and residuals
- Explore time series diagnostics
- Visualize model predictions vs actual values

### Step 4: Predictions & Forecasting 🔮
- Generate predictions on test data
- Create future forecasts
- Visualize predictions with confidence intervals
- Export forecast results

### Step 5: Model Management 💾
- Save trained models to disk
- Load models from disk
- Deploy models to cloud platforms
- Manage model versions and experiments

## 📊 Available Datasets

The application includes several sample datasets:
- **Airline**: Monthly airline passenger numbers
- **Airline Passengers**: Alternative airline dataset
- **Boston**: Housing data
- **Energy**: Energy consumption data
- **Weather**: Weather time series data

## 🤖 Supported Models

### Statistical Models
- **ETS** (Error, Trend, Seasonality)
- **ARIMA** (AutoRegressive Integrated Moving Average)
- **Theta**
- **Naive**
- **Seasonal Naive**

### Machine Learning Models
- **Decision Trees**
- **Random Forest**
- **XGBoost**
- **LightGBM**
- **Neural Networks**

### Baseline Models
- **Grand Means**
- **Polynomial Trend**
- **Moving Average**
- **Exponential Smoothing**

## 📈 Performance Metrics

The application evaluates models using multiple metrics:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **MASE** (Mean Absolute Scaled Error)
- **R²** (Coefficient of Determination)

## 🎨 Features

### Interactive Dashboard
- **Real-time Model Training**: Train models with custom parameters
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Dynamic Configuration**: Adjust settings via sidebar
- **Progress Tracking**: Visual feedback for all operations

### Model Management
- **Save/Load Models**: Persistent storage of trained models
- **Model Registry**: Track and manage multiple models
- **Metadata Storage**: Store model descriptions and parameters
- **Version Control**: Keep track of model versions

### Export Capabilities
- **CSV Export**: Download predictions and forecasts
- **Model Deployment**: Deploy models locally or to cloud
- **Configuration Export**: Save experiment settings

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web application
- **Backend**: PyCaret time series module
- **Visualization**: Plotly and Matplotlib
- **Data Processing**: Pandas and NumPy

### File Structure
```
time-series/
├── Home.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── pages/
│   ├── 1_Setup.py                  # Step 1: Setup & Data Loading
│   ├── 2_Model_Comparison.py       # Step 2: Model Comparison
│   ├── 3_Model_Analysis.py         # Step 3: Model Analysis
│   ├── 4_Predictions.py            # Step 4: Predictions
│   └── 5_Model_Management.py       # Step 5: Model Management
├── models/                         # Saved models directory
├── deployments/                    # Model deployments
└── notebooks/                      # Jupyter notebooks
```

## 🔧 Configuration

### Environment Variables
- `PYCARET_LOG_LEVEL`: Set logging level for PyCaret
- `STREAMLIT_SERVER_PORT`: Custom port for Streamlit (default: 8501)

### Customization
- **Datasets**: Add your own datasets to the data loading section
- **Models**: Extend with custom model implementations
- **Metrics**: Add custom evaluation metrics
- **Visualizations**: Create custom plot types

## 🚀 Deployment

### Local Deployment
```bash
streamlit run Home.py --server.port 8501
```

### Cloud Deployment
The application can be deployed to:
- **Heroku**: Use the provided `Procfile`
- **AWS**: Deploy using AWS Elastic Beanstalk
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Deploy to Azure App Service

## 📚 Learning Resources

### PyCaret Documentation
- [PyCaret Official Docs](https://pycaret.org/)
- [Time Series Module](https://pycaret.org/time-series/)
- [API Reference](https://pycaret.org/api/)

### Time Series Forecasting
- [Time Series Analysis](https://otexts.com/fpp3/)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Statistical Forecasting](https://www.statsmodels.org/stable/tsa.html)

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyCaret Team**: For the amazing low-code ML library
- **Streamlit Team**: For the interactive web app framework
- **Open Source Community**: For the various dependencies and tools

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Join our [Discord](https://discord.gg/pycaret) community

---

**Happy Forecasting! 🎉**

Built with ❤️ using Streamlit and PyCaret