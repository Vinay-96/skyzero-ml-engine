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

# Additional imports for the ensemble model
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

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

# -------------------------------
# 1. Enhanced Feature Engineering
# -------------------------------
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
        
            # Advanced volatility: Keltner Channel
            keltner = ta.volatility.KeltnerChannel(df.high, df.low, df.close)
            features[f'{self.prefix}Keltner_Width'] = ((keltner.keltner_channel_hband() - 
                                                        keltner.keltner_channel_lband()) / df.close)
        
            # Lagged features
            for lag in [1, 3, 5, 8]:
                features[f'{self.prefix}lag_{lag}'] = df['close'].shift(lag)
            
            # === New Enhanced Features ===
            # Advanced Momentum
            features[f'{self.prefix}Vortex'] = ta.trend.VortexIndicator(df.high, df.low, df.close).vortex_indicator_diff()
            # features[f'{self.prefix}CMO'] = ta.momentum.ChaikinMoneyFlowIndicator(df.high, df.low, df.close, df.volume).chaikin_money_flow()
            
            # Machine Learning-Generated Features
            features[f'{self.prefix}Residuals'] = df.close - df.close.rolling(20).mean()
            # Compute a rolling FFT on the close price percentage change over a 20-bar window,
            # then add the first 5 FFT components as separate features.
            for i in range(5):
                features[f'{self.prefix}Wavelet_{i+1}'] = (
                    df.close.pct_change()
                    .fillna(0)
                    .rolling(window=20)
                    .apply(lambda x: np.abs(np.fft.fft(x))[i], raw=False)
                )
            
            # Market Microstructure Features
            features[f'{self.prefix}OrderImbalance'] = (df.volume * (df.close - df.open)).rolling(5).sum()
            features[f'{self.prefix}VolCluster'] = df['close'].pct_change().rolling(10).skew()
            
            # Forecast target (will be overwritten in preprocess_data)
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            features['next_close'] = df['close'].shift(-1)  # Store actual next close
            
            return features.dropna()
        except Exception as e:
            print(f"Feature calculation error: {str(e)}")
            return pd.DataFrame()

# -------------------------------
# Trend Strength Target Function
# -------------------------------
def calculate_target(series):
    returns = series.pct_change(3).shift(-3)
    return (returns > returns.quantile(0.6)).astype(int)  # Top 40% moves

# -------------------------------
# Data Loading Functions
# -------------------------------
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

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(df):
    """Feature engineering pipeline"""
    calculator = EnhancedFeatureCalculator()
    features = calculator.calculate_core_features(df)
    
    # Overwrite the 'target' column with the trend strength target
    features['target'] = calculate_target(df.close)
    
    # Save feature names (exclude target and next_close)
    feature_cols = [col for col in features.columns if col not in ['target', 'next_close']]
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_cols, f)
    
    # Prepare features and target
    X = features[feature_cols]
    y = features['target']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    
    return X_scaled, y.values, scaler, feature_cols, features['next_close']

# -------------------------------
# 2. Improved Model Architecture - Train Model
# -------------------------------
def train_model(use_csv=False, csv_file_path="historical_data.csv"):
    """
    Train the model using data loaded from yfinance (default) or from a CSV file.
    
    Parameters:
        use_csv (bool): If True, load data from the CSV file specified by csv_file_path.
                        Otherwise, data is loaded from yfinance.
        csv_file_path (str): Path to the CSV file with historical data.
    """
    # Use a new filename for the ensemble model
    model_path = os.path.join(model_dir, "ensemble_model_1m.model")
    
    if os.path.exists(model_path):
        print("Loading existing ensemble model...")
        model = joblib.load(model_path)
        return model

    print("Training new ensemble model...")
    if use_csv:
        df = load_data_csv(csv_file_path)
    else:
        df = load_data()
        
    # Preprocess the data (feature engineering, scaling, etc.)
    X, y, scaler, feature_cols, next_closes = preprocess_data(df)

    # Temporal split: use 80% of the data for training and 20% for testing
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Balance classes with SMOTE
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Build the ensemble stacking classifier
    estimators = [
        ('xgb', xgb.XGBClassifier(
            max_depth=5, 
            learning_rate=0.1, 
            n_estimators=300,
            use_label_encoder=False, 
            eval_metric='logloss'
        )),
        ('lgbm', LGBMClassifier(
            num_leaves=31, 
            learning_rate=0.05, 
            n_estimators=200
        )),
        ('logreg', LogisticRegression(
            max_iter=1000, 
            class_weight='balanced'
        ))
    ]

    final_estimator = xgb.XGBClassifier(
        max_depth=3, 
        learning_rate=0.05, 
        n_estimators=150,
        use_label_encoder=False, 
        eval_metric='logloss'
    )

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method='predict_proba',
        passthrough=True
    )

    # Fit the ensemble model on the training data
    model.fit(X_train, y_train)

    # Optional: perform temporal cross-validation on the full dataset and print metrics
    cv_results = temporal_cross_val(df, model)
    print("Temporal Cross Validation Results:")
    print(cv_results)

    # Evaluate on the hold-out test set
    preds = model.predict(X_test)
    print("\n=== Ensemble Model Validation ===")
    print(classification_report(y_test, preds))

    # Save the trained model for future use
    joblib.dump(model, model_path)
    print(f"Ensemble model saved to {model_path}")
    return model

# -------------------------------
# 3. Temporal Cross-Validation Function
# -------------------------------
def temporal_cross_val(df, model):
    tscv = TimeSeriesSplit(n_splits=5, test_size=500)
    metrics = []
    
    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Feature engineering for train and test splits
        X_train, y_train, _, _, _ = preprocess_data(train)
        X_test, y_test, _, _, _ = preprocess_data(test)
        
        # Dynamic reweighting: give higher weights to more recent samples
        sample_weights = np.exp(np.linspace(-1, 0, len(y_train)))
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_test)
        
        fold_metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0)
        }
        metrics.append(fold_metrics)
    
    return pd.DataFrame(metrics)

# -------------------------------
# Real-Time Forecast Function
# -------------------------------
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
    
    # Load trained ensemble model and scaler
    model_path = os.path.join(model_dir, "ensemble_model_1m.model")
    model = joblib.load(model_path)
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

    # Calculate dynamic volatility
    if (last_high - last_low) == 0:
        base_volatility = last_close * 0.0005  # fallback volatility
    else:
        base_volatility = (last_high - last_low) * 0.5

    direction_multiplier = 1 if predicted_direction else -1
    price_change = base_volatility * prob * direction_multiplier
    next_close = last_close + price_change

    print(f"{timestamp} | Predicted Direction: {'UP' if predicted_direction else 'DOWN'}, Confidence: {prob:.2%}")
    print(f"Last Close Price: {last_close:.2f}, Predicted Next Close: {next_close:.2f}")

    # Save prediction to CSV file
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

# -------------------------------
# Live Plotting Function
# -------------------------------
def update_live_plot(output_file="/plots/live_plot.png"):
    """Update a live Matplotlib plot of actual market close vs. forecast close prices."""
    if not live_predictions:
        return

    df_plot = pd.DataFrame(live_predictions)
    
    plt.figure("Live Forecast vs Actual", figsize=(10, 6))
    plt.plot(df_plot['timestamp'], df_plot['actual_close'], label="Actual Close", marker="o")
    plt.plot(df_plot['timestamp'], df_plot['forecast_close'], label="Forecast Close", marker="x")
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title("Live Actual vs. Forecast Close Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()

# -------------------------------
# Backtrader Strategy
# -------------------------------
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
        self.model = joblib.load(os.path.join(model_dir, "ensemble_model_1m.model"))
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
            
            # Log forecast details
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
            
            y_true = df_forecast['actual_direction']
            y_pred = df_forecast['predicted_direction']
            
            print(f"\nDirection Accuracy: {accuracy_score(y_true, y_pred):.2%}")
            print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.2%}")
            print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.2%}")
            print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.2%}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            
            df_forecast['predicted_close'] = df_forecast['current_close'] * (1 + df_forecast['predicted_prob'] * 0.0005)
            
            mae = mean_absolute_error(df_forecast['actual_next_close'], df_forecast['predicted_close'])
            rmse = np.sqrt(mean_squared_error(df_forecast['actual_next_close'], df_forecast['predicted_close']))
            r2 = r2_score(df_forecast['actual_next_close'], df_forecast['predicted_close'])                          
            print(f"\nMean Absolute Error: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R-squared: {r2:.4f}")
            
            print("\nSample Forecasts:")
            print(df_forecast[['timestamp', 'current_close', 'predicted_prob', 
                             'predicted_close', 'actual_next_close']].tail(5))
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(df_forecast['timestamp'], df_forecast['current_close'], label='Current Close')
            plt.plot(df_forecast['timestamp'], df_forecast['predicted_close'], 
                     label='Predicted Close', alpha=0.7)
            plt.plot(df_forecast['timestamp'], df_forecast['actual_next_close'], 
                     label='Actual Next Close', linestyle='--')
            plt.title('Price Forecast Performance')
            plt.legend()

            plt.subplot(3, 1, 2)
            calculator = EnhancedFeatureCalculator()
            full_features = calculator.calculate_core_features(self.data_df.set_index('timestamp').ffill())
            full_features.reset_index(inplace=True)
            merged = pd.merge(df_forecast, full_features, on='timestamp', how='left')
            plt.plot(merged['timestamp'], merged['1m_RSI'], label='RSI')
            plt.axhline(70, linestyle='--', color='r', alpha=0.5)
            plt.axhline(30, linestyle='--', color='g', alpha=0.5)
            plt.title('RSI Indicator')
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(merged['timestamp'], merged['1m_Keltner_Width'], label='Keltner Width')
            plt.title('Keltner Channel Width')
            plt.legend()

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            residuals = df_forecast['actual_next_close'] - df_forecast['predicted_close']
            plt.scatter(df_forecast['predicted_close'], residuals, alpha=0.5)
            plt.axhline(0, color='r', linestyle='--')
            plt.xlabel('Predicted Close')
            plt.ylabel('Residuals')
            plt.title('Prediction Residuals')

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

# -------------------------------
# Backtest Runner
# -------------------------------
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

# -------------------------------
# Main Execution Block
# -------------------------------
if __name__ == "__main__":
    # Train or load the ensemble model.
    model = train_model(use_csv=False, csv_file_path="./data/NIFTY_BANK_1m.csv")
    
    # Optionally, run temporal cross-validation separately.
    df = load_data()  # Or load_data_csv(...) if preferred.

    # cv_results = temporal_cross_val(df, model)
    # print("Temporal Cross Validation Results:")
    # print(cv_results)
    # run_backtest()
    
    # Schedule the real-time forecasting job to run every minute.
    schedule.every(1).minutes.do(real_time_forecast)
    
    print("Starting scheduled real-time forecasting. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Real-time forecasting terminated.")
