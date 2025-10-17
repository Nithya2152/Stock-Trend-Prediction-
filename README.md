# Stock Price Prediction

## Overview
This repository hosts a Flask web application that serves real-time stock closing price predictions powered by a Random Forest regressor trained on historical market data.

## Features
- **Interactive UI**: Bootstrap interface for submitting ticker symbols and visualizing prediction outputs.
- **Real-time data fetch**: Downloads the latest OHLCV data via `yfinance` before every prediction.
- **Model metrics**: Displays MAE, RMSE, and R² to evaluate prediction quality.
- **Charting**: Generates comparison plots and exponential moving average overlays saved under `static/`.
- **Data export**: Provides CSV downloads for the processed feature set and predictions.

## Prerequisites
- **Python**: 3.9 or newer
- **System packages**: Build tooling required for scientific Python wheels (e.g., `gcc`, `build-essential` on Linux)

## Installation
1. Clone the repository.
2. Create a virtual environment.
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies.
   ```bash
   pip install flask pandas numpy scikit-learn yfinance matplotlib joblib
   ```

## Usage
1. Ensure model artifacts (`powergrid_model.pkl`, `scaler.pkl`, `stock_dl_model.h5`, etc.) reside in the project root.
2. Start the server.
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in a browser, enter a valid stock ticker (e.g., `POWERGRID.NS`), and submit to view predictions, metrics, and downloads.

## Training Pipeline
1. Place the raw dataset at `powergrid.csv`.
2. Rebuild the model and scaler.
   ```bash
   python model.py
   ```
3. Updated artifacts will overwrite the previous versions and remain excluded from version control via `.gitignore`.

## Project Structure
- **app.py**: Flask application entry point exposing the prediction interface.
- **model.py**: Training script that fits the Random Forest model and persists artifacts with `joblib`.
- **templates/**: HTML templates for the main dashboard and error view.
- **static/**: Generated plots and CSV exports returned to end users.
- **Stock Price Prediction .ipynb**: Exploratory notebook with supplemental experimentation.

## Data Protection
- **Credentials**: Store API keys or secrets in a `.env` file (already ignored by Git).
- **Artifacts**: Large binaries and datasets (`*.pkl`, `*.h5`, `*.keras`, `*.csv`) are ignored to prevent leaking sensitive or proprietary assets.
