import numpy as np
import pandas as pd
import os
import json
import xgboost as xgb
import joblib
import ta
import schedule
import time
import pymongo
from datetime import datetime, timedelta
from dotenv import load_dotenv
import backtrader as bt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, mean_absolute_error,
                             classification_report, mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy.signal import hilbert
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf

# Global variable to store predictions for live plotting
live_predictions = []  # Each entry is a dict with keys: timestamp, actual_close, forecast_close

# CSV file for logging predictions
CSV_FILE = "real_time_forecast_log.csv"
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configuration
model_dir = "./models"
PAST_BARS = 20
PREDICTION_THRESHOLD = 0.001
VALIDATION_WINDOWS = 5
FEATURE_NAMES_FILE = os.path.join(model_dir, "feature_names.json")
SCALER_FILE = os.path.join(model_dir, "scaler.pkl")

os.makedirs(model_dir, exist_ok=True)

# MongoDB setup (optional)
client = pymongo.MongoClient(os.getenv("DATABASE_URL", "mongodb+srv://BaseZero:dWajaoeQDj5sNvuD@skyzero.j95hi.mongodb.net/"))
db = client['skyZero']

class EnhancedFeatureCalculator:
    """Advanced feature engineering for price forecasting"""
    def __init__(self):
        self.prefix = '1m_'
        
    def calculate_core_features(self, df):
        """Robust feature engineering with validation"""
        if len(df) < PAST_BARS*2:
            return pd.DataFrame()
        try:
            features = pd.DataFrame(index=df.index)
        
            # Price transformation features
            features[f'{self.prefix}log_return'] = np.log(df['close'] / df['close'].shift(1))
            features[f'{self.prefix}cumulative_gain'] = df['close'].pct_change().rolling(5).sum()
            features[f'{self.prefix}volatility'] = df['close'].pct_change().rolling(PAST_BARS).std()
        
            # Advanced momentum indicators
            features[f'{self.prefix}RSI'] = ta.momentum.RSIIndicator(df.close).rsi()
            features[f'{self.prefix}TSI'] = ta.momentum.TSIIndicator(df.close).tsi()
            features[f'{self.prefix}KAMA'] = ta.momentum.KAMAIndicator(df.close).kama()
        
            # Cycle detection
            features[f'{self.prefix}Hilbert'] = np.unwrap(np.angle(hilbert(df.close)))
        
            # Advanced volatility
            keltner = ta.volatility.KeltnerChannel(df.high, df.low, df.close)
            features[f'{self.prefix}Keltner_Width'] = ((keltner.keltner_channel_hband() - 
                                                        keltner.keltner_channel_lband()) / df.close)
        
            # Lagged features
            for lag in [1, 3, 5, 8]:
                features[f'{self.prefix}lag_{lag}'] = df['close'].shift(lag)
            
            # Forecast target
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            features['next_close'] = df['close'].shift(-1)  # Store actual next close
            
            return features.dropna()
        except Exception as e:
            print(f"Feature calculation error: {str(e)}")
            return pd.DataFrame()

def load_data():
    """Load historical data from Yahoo Finance for training"""
    symbol = "^NSEBANK"
    interval = "1m"
    period = "8d"  # 8 days of historical data
    
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    return df.ffill().dropna()

def load_data_csv(csv_file_path):
    """
    Load historical data from a CSV file for training.
    The CSV file should contain a 'timestamp' column that can be parsed into datetime,
    and columns: open, high, low, close, volume.
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file '{csv_file_path}' not found.")
    
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index("timestamp", inplace=True)
    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]
    return df.ffill().dropna()

def load_live_data(symbol="^NSEBANK", interval="1m", period="1d"):
    """Fetch real-time 1-minute price data from Yahoo Finance."""
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    return df.ffill().dropna()

def real_time_forecast():
    """Fetches live data and makes real-time predictions."""
    print("\nFetching latest market data...")
    df = load_live_data()

    if df.empty or len(df) < 2:
        print("No new data available.")
        return
    
    # Extract the latest timestamp from yfinance data in 12-hour format
    last_timestamp = df.index[-1]
    timestamp = last_timestamp.strftime("%Y-%m-%d %I:%M:%S %p")
    
    # Load trained model and scaler
    model_path = os.path.join(model_dir, "xgboost_1m_one.model")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scaler = joblib.load(SCALER_FILE)
    with open(FEATURE_NAMES_FILE, 'r') as f:
        feature_names = json.load(f)

    calculator = EnhancedFeatureCalculator()
    features = calculator.calculate_core_features(df)

    if features.empty:
        print("Feature extraction failed.")
        return  

    latest_features = features[feature_names].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_features)

    prob = model.predict_proba(latest_scaled)[0][1]
    predicted_direction = 1 if prob > 0.5 else 0

    # Calculate the next close price
    last_close = df["close"].iloc[-1]
    last_high = df["high"].iloc[-1]
    last_low = df["low"].iloc[-1]

    # Calculate price change using range and confidence
    # price_change = (last_high - last_low) * prob * (1 if predicted_direction else -1)
    # next_close = last_close + price_change

    # Calculate dynamic volatility
    if (last_high - last_low) == 0:
        # Fallback to percentage-based volatility
        base_volatility = last_close * 0.0005  # 0.05%
    else:
        # Use actual range scaled by confidence
        base_volatility = (last_high - last_low) * 0.5 # Reduce impact of full range

    direction_multiplier = 1 if predicted_direction else -1
    price_change = base_volatility * prob * direction_multiplier
    next_close = last_close + price_change

    print(f"{timestamp} | Predicted Direction: {'UP' if predicted_direction else 'DOWN'}, Confidence: {prob:.2%}")
    print(f"Last Close Price: {last_close:.2f}, Predicted Next Close: {next_close:.2f}")

    # Save prediction to CSV file including the yfinance timestamp in 12-hour format
    data = {
        "Timestamp": [timestamp],
        "Last_Close": [last_close],
        "Predicted_Next_Close": [next_close],
        "Direction": ["UP" if predicted_direction else "DOWN"],
        "Confidence": [f"{prob * 100:.2f}%"]
    }

    df_log = pd.DataFrame(data)

    if not os.path.exists(CSV_FILE):
        df_log.to_csv(CSV_FILE, index=False, mode='w')
    else:
        df_log.to_csv(CSV_FILE, index=False, mode='a', header=False)

    # Append the new prediction to the global list for live plotting
    live_predictions.append({
        "timestamp": last_timestamp,
        "actual_close": last_close,
        "forecast_close": next_close
    })

    # Update the live plot (saves to the shared volume directory)
    update_live_plot(output_file="/plots/live_plot.png")

    return predicted_direction, prob, next_close

def update_live_plot(output_file="/plots/live_plot.png"):
    """Update a live Matplotlib plot of actual market close vs. forecast close prices."""
    if not live_predictions:
        return

    # Convert list of dictionaries to a DataFrame
    df_plot = pd.DataFrame(live_predictions)
    
    # Create the plot
    plt.figure("Live Forecast vs Actual", figsize=(10, 6))
    
    # Plot actual close prices
    plt.plot(df_plot['timestamp'], df_plot['actual_close'], label="Actual Close", marker="o")
    # Plot forecast close prices
    plt.plot(df_plot['timestamp'], df_plot['forecast_close'], label="Forecast Close", marker="x")
    
    # Format the x-axis to show time properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title("Live Actual vs. Forecast Close Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a file in the shared volume directory
    plt.savefig(output_file)
    plt.close()  # Close the figure to free up memory

def preprocess_data(df):
    """Feature engineering pipeline"""
    calculator = EnhancedFeatureCalculator()
    features = calculator.calculate_core_features(df)
    
    # Save feature names
    feature_cols = [col for col in features.columns if col not in ['target', 'next_close']]
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_cols, f)
    
    # Handle class imbalance
    X = features[feature_cols]
    y = features['target']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    
    return X_scaled, y.values, scaler, feature_cols, features['next_close']

def optimize_model(X_train, y_train):
    """Optimized classifier tuning"""
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.7, 0.8]
    }
    
    model = xgb.XGBClassifier(objective='binary:logistic', 
                              eval_metric='logloss',
                              use_label_encoder=False)
    
    tscv = TimeSeriesSplit(n_splits=VALIDATION_WINDOWS)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_

def train_model(use_csv=False, csv_file_path="historical_data.csv"):
    """
    Train the model using data loaded from yfinance (default) or from a CSV file.
    
    Parameters:
        use_csv (bool): If True, load data from the CSV file specified by csv_file_path.
                        Otherwise, data is loaded from yfinance.
        csv_file_path (str): Path to the CSV file with historical data.
    """
    model_path = os.path.join(model_dir, "xgboost_1m_one.model")
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    print("Training new model...")
    if use_csv:
        df = load_data_csv(csv_file_path)
    else:
        df = load_data()
    X, y, scaler, feature_cols, next_closes = preprocess_data(df)

    # Temporal split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Balance classes
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = optimize_model(X_train, y_train)
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=50,
              verbose=True)

    model.save_model(model_path)
    validate_model(model, X_test, y_test)
    return model

def validate_model(model, X_test, y_test):
    """Enhanced validation with feature analysis"""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Validation ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds))
    
    importance = model.get_booster().get_score(importance_type='weight')
    print("\nTop 10 Features:")
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feat}: {score}")

    # Feature importance visualization
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(importance)
    feat_importances.nlargest(10).sort_values().plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

class ForecastStrategy(bt.Strategy):
    """Optimized trading strategy with price forecasting"""
    params = (
        ('risk_pct', 0.5),
        ('stop_loss', 0.3),
        ('take_profit', 0.9),
        ('confidence_threshold', 0.7),
        ('data_df', None)
    )
    
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "xgboost_1m_one.model"))
        self.scaler = joblib.load(SCALER_FILE)
        with open(FEATURE_NAMES_FILE, 'r') as f:
            self.feature_names = json.load(f)
        
        self.calculator = EnhancedFeatureCalculator()
        self.order = None
        self.trade_log = []
        self.forecast_log = []
        self.data_df = self.params.data_df.copy()

    def next(self):
        if len(self.data) < PAST_BARS*2 + 10:
            return
        
        try:
            current_idx = len(self.data) - 1
            df_window = self.data_df.iloc[max(0, current_idx-PAST_BARS):current_idx+1]
            
            features = self.calculator.calculate_core_features(df_window)
            if features.empty:
                return

            if len(features) >= 1:
                current_features = features[self.feature_names].iloc[-1].values.reshape(1, -1)
            else:
                return

            scaled = self.scaler.transform(current_features)
            prob = self.model.predict_proba(scaled)[0][1]
            current_close = df_window.iloc[-1]['close']
            
            # Store forecast details using the data timestamp
            if current_idx + 1 < len(self.data_df):
                actual_next_close = self.data_df.iloc[current_idx + 1]['close']
                forecast_entry = {
                    'timestamp': self.data.datetime.datetime(),
                    'current_close': current_close,
                    'predicted_prob': prob,
                    'predicted_direction': 1 if prob >= 0.5 else 0,
                    'actual_next_close': actual_next_close,
                    'actual_direction': 1 if actual_next_close > current_close else 0
                }
                self.forecast_log.append(forecast_entry)

            # Trading logic
            if prob > self.params.confidence_threshold:
                self.execute_trades(1, prob)
            elif prob < (1 - self.params.confidence_threshold):
                self.execute_trades(-1, prob)

        except Exception as e:
            print(f"Prediction error: {str(e)}")

    def execute_trades(self, direction, confidence):
        """Enhanced trade execution with risk management"""
        if self.order:
            return

        size = self.broker.getvalue() * self.params.risk_pct * confidence
        price = self.data.close[0]
        
        if direction == 1:
            self.order = self.buy(size=size)
            self.sell(exectype=bt.Order.Stop, price=price*(1-self.params.stop_loss))
            self.sell(exectype=bt.Order.Limit, price=price*(1+self.params.take_profit))
        else:
            self.order = self.sell(size=size)
            self.buy(exectype=bt.Order.Stop, price=price*(1+self.params.stop_loss))
            self.buy(exectype=bt.Order.Limit, price=price*(1-self.params.take_profit))

    def notify_trade(self, trade):
        """Advanced trade tracking"""
        if trade.isclosed:
            self.trade_log.append({
                'entry_price': trade.price,
                'exit_price': trade.pnl,
                'pnl': trade.pnlcomm,
                'closed_at': self.data.datetime.datetime()
            })

    def stop(self):
        """Display comprehensive forecasting results"""
        if len(self.forecast_log) > 0:
            df_forecast = pd.DataFrame(self.forecast_log)
            
            print("\n=== Advanced Forecasting Results ===")
            print(f"Total Predictions: {len(df_forecast)}")
            
            # Direction metrics
            y_true = df_forecast['actual_direction']
            y_pred = df_forecast['predicted_direction']
            
            print(f"\nDirection Accuracy: {accuracy_score(y_true, y_pred):.2%}")
            print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.2%}")
            print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.2%}")
            print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.2%}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            
            # Price prediction metrics
            df_forecast['predicted_close'] = df_forecast['current_close'] * \
                (1 + df_forecast['predicted_prob'] * 0.0005)
            
            mae = mean_absolute_error(df_forecast['actual_next_close'], 
                                      df_forecast['predicted_close'])
            rmse = np.sqrt(mean_squared_error(df_forecast['actual_next_close'], 
                                              df_forecast['predicted_close']))
            r2 = r2_score(df_forecast['actual_next_close'], df_forecast['predicted_close'])                          
            print(f"\nMean Absolute Error: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R-squared: {r2:.4f}")
            
            # Sample predictions
            print("\nSample Forecasts:")
            print(df_forecast[['timestamp', 'current_close', 'predicted_prob', 
                             'predicted_close', 'actual_next_close']].tail(5))
            
            # Visualization 1: Price and Predictions
            plt.figure(figsize=(15, 10))
            
            # Price Plot
            plt.subplot(3, 1, 1)
            plt.plot(df_forecast['timestamp'], df_forecast['current_close'], label='Current Close')
            plt.plot(df_forecast['timestamp'], df_forecast['predicted_close'], 
                     label='Predicted Close', alpha=0.7)
            plt.plot(df_forecast['timestamp'], df_forecast['actual_next_close'], 
                     label='Actual Next Close', linestyle='--')
            plt.title('Price Forecast Performance')
            plt.legend()

            # RSI Plot
            plt.subplot(3, 1, 2)
            calculator = EnhancedFeatureCalculator()
            full_features = calculator.calculate_core_features(
                self.data_df.set_index('timestamp').ffill()
            )
            full_features.reset_index(inplace=True)
            merged = pd.merge(df_forecast, full_features, on='timestamp', how='left')
            plt.plot(merged['timestamp'], merged['1m_RSI'], label='RSI')
            plt.axhline(70, linestyle='--', color='r', alpha=0.5)
            plt.axhline(30, linestyle='--', color='g', alpha=0.5)
            plt.title('RSI Indicator')
            plt.legend()

            # Keltner Channel Plot
            plt.subplot(3, 1, 3)
            plt.plot(merged['timestamp'], merged['1m_Keltner_Width'], label='Keltner Width')
            plt.title('Keltner Channel Width')
            plt.legend()

            plt.tight_layout()
            plt.show()

            # Visualization 2: Prediction Analysis
            plt.figure(figsize=(15, 5))
            
            # Residuals Plot
            plt.subplot(1, 2, 1)
            residuals = df_forecast['actual_next_close'] - df_forecast['predicted_close']
            plt.scatter(df_forecast['predicted_close'], residuals, alpha=0.5)
            plt.axhline(0, color='r', linestyle='--')
            plt.xlabel('Predicted Close')
            plt.ylabel('Residuals')
            plt.title('Prediction Residuals')

            # Scatter Plot
            plt.subplot(1, 2, 2)
            plt.scatter(df_forecast['actual_next_close'], df_forecast['predicted_close'], alpha=0.5)
            plt.plot([df_forecast['actual_next_close'].min(), df_forecast['actual_next_close'].max()],
                     [df_forecast['actual_next_close'].min(), df_forecast['actual_next_close'].max()], 
                     'r--')
            plt.xlabel('Actual Close')
            plt.ylabel('Predicted Close')
            plt.title('Actual vs Predicted Close')

            plt.tight_layout()
            plt.show()

def run_backtest():
    """Optimized backtesting engine with enhanced reporting"""
    cerebro = bt.Cerebro(optreturn=False)
    
    # Add data
    df = load_data().reset_index()
    data = bt.feeds.PandasData(dataname=df, datetime='timestamp')
    cerebro.adddata(data)
    
    # Configure strategy
    cerebro.addstrategy(ForecastStrategy, data_df=df)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.0005)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print("Starting Enhanced Backtest...")
    results = cerebro.run()
    strat = results[0]
    
    # Performance report
    print("\n=== Trading Performance ===")
    print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        print("\n=== Analyzer Results ===")
        print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
        max_dd = drawdown.get('max', {}).get('drawdown', 'N/A')
        print(f"Max Drawdown: {max_dd}")
        if 'total' in trades and 'closed' in trades['total']:
            print(f"Total Closed Trades: {trades['total']['closed']}")
        else:
            print("No trade data available.")
    except Exception as e:
        print(f"Analyzer error: {str(e)}")

    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    # Optionally, train the model if not already trained.
    # Set use_csv=True and provide a valid csv file path to load data from CSV.
    train_model(use_csv=False, csv_file_path="./data/NIFTY_BANK_1m.csv")
    
    # Schedule the real-time forecasting job to run every minute.
    schedule.every(1).minutes.do(real_time_forecast)
    
    print("Starting scheduled real-time forecasting. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Real-time forecasting terminated.")
