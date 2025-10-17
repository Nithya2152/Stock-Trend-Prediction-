import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import yfinance as yf
import joblib
from flask import Flask, render_template, request, send_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# Load model and scaler safely
MODEL_PATH = "powergrid_model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    raise
except Exception as e:
    print(f"Unexpected error loading model/scaler: {e}")
    raise

def fetch_stock_data(symbol, days=365):
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days)
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"No data found for symbol: {symbol}")
            return pd.DataFrame()
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()

def create_ema_plots(df, symbol):
    if 'Close' not in df.columns:
        print("Error: 'Close' column missing in DataFrame.")
        return None, None

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    if df[['EMA20', 'EMA50', 'EMA100', 'EMA200']].isnull().any().any():
        print("Warning: NaN values encountered after EMA calculation.")

    # Plot 1
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price', alpha=0.7)
    plt.plot(df.index, df['EMA20'], label='EMA 20', alpha=0.7)
    plt.plot(df.index, df['EMA50'], label='EMA 50', alpha=0.7)
    plt.title(f"{symbol} Closing Price vs Time (20 & 50 Days EMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    ema_20_50_path = os.path.join(app.config['STATIC_FOLDER'], f"{symbol}_ema_20_50.png")
    plt.savefig(ema_20_50_path)
    plt.close()

    # Plot 2
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price', alpha=0.7)
    plt.plot(df.index, df['EMA100'], label='EMA 100', alpha=0.7)
    plt.plot(df.index, df['EMA200'], label='EMA 200', alpha=0.7)
    plt.title(f"{symbol} Closing Price vs Time (100 & 200 Days EMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    ema_100_200_path = os.path.join(app.config['STATIC_FOLDER'], f"{symbol}_ema_100_200.png")
    plt.savefig(ema_100_200_path)
    plt.close()

    return ema_20_50_path, ema_100_200_path

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_path_pred = None
    plot_path_ema_20_50 = None
    plot_path_ema_100_200 = None
    csv_path = None
    data_desc_html = None
    error_message = None
    mae = rmse = r2 = None

    if request.method == 'POST':
        stock_symbol = request.form.get('stock', 'POWERGRID.NS')
        print(f"Received stock symbol: {stock_symbol}")

        try:
            df = fetch_stock_data(stock_symbol)
            if df.empty:
                error_message = "Invalid stock symbol or no data available."
                print(error_message)
                return render_template('index.html', error_message=error_message)

            df['Volatility'] = df['High'] - df['Low']
            features = df[["Open", "High", "Low", "Volume"]]
            scaled_features = scaler.transform(features)
            df['Predicted_Close'] = model.predict(scaled_features)

            mae = mean_absolute_error(df['Close'], df['Predicted_Close'])
            rmse = np.sqrt(mean_squared_error(df['Close'], df['Predicted_Close']))
            r2 = r2_score(df['Close'], df['Predicted_Close'])

            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], label='Actual Close Price', alpha=0.7)
            plt.plot(df.index, df['Predicted_Close'], label='Predicted Close Price', alpha=0.7)
            plt.title(f"{stock_symbol} Stock Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plot_path_pred = os.path.join(app.config['STATIC_FOLDER'], f"{stock_symbol}_prediction.png")
            plt.savefig(plot_path_pred)
            plt.close()

            ema_20_50_path, ema_100_200_path = create_ema_plots(df.copy(), stock_symbol)
            plot_path_ema_20_50 = ema_20_50_path
            plot_path_ema_100_200 = ema_100_200_path

            data_desc = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
            data_desc_html = data_desc.to_html(classes='table table-bordered')

            csv_path = os.path.join(app.config['STATIC_FOLDER'], f"{stock_symbol}_data.csv")
            df.to_csv(csv_path)
            print(f"CSV saved to: {csv_path}")

        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            return render_template('index.html', error_message=error_message)

    return render_template('index.html',
                           plot_path_pred=plot_path_pred,
                           plot_path_ema_20_50=plot_path_ema_20_50,
                           plot_path_ema_100_200=plot_path_ema_100_200,
                           data_desc_html=data_desc_html,
                           csv_path=csv_path,
                           error_message=error_message,
                           mae=mae,
                           rmse=rmse,
                           r2=r2)

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['STATIC_FOLDER'], filename)
    print("Trying to send file:", file_path)

    if not os.path.exists(file_path):
        return render_template('error.html', message="File not found!")

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
