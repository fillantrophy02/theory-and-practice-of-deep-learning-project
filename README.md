# 🌧️ Rainfall Prediction with Deep Learning

This project benchmarks deep learning models for **daily rainfall prediction** using **meteorological time series data** from [Australian Weather Data](https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data?select=Weather+Training+Data.csv). Implemented in PyTorch, it evaluates standard and custom variants of **LSTM, GRU, and Transformer** architectures.

## 📂 Datasets

The dataset contains sequential weather features from multiple Australian locations, including:

- Temperature  
- Humidity  
- Pressure  
- Wind patterns  

The target is a **binary rainfall occurrence** (rain vs. no rain) for each day.

## 🔧 Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the environment and install dependencies:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Evaluate model with existing weights:
   ```bash
   python main.py
   ```
   
   To switch between models, open config.py:
   ```python
   model = "LSTM"  # Options: "LSTM", "GRU", "Transformer", etc.
   ```
   
4. (Optional) Retrain model from scratch:
   ```python
   use_existing_weights = False
   ```
