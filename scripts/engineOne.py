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
                             classification_report)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy.signal import hilbert
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

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
            features[f'{self.prefix}log_return'] = np.log(df['close']/df['close'].shift(1))
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
    period = "60d"  # 60 days of historical data
    
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
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
    price_change = (last_high - last_low) * prob * (1 if predicted_direction else -1)
    next_close = last_close + price_change

    print(f"{timestamp} | Predicted Direction: {'UP' if predicted_direction else 'DOWN'}, Confidence: {prob:.2%}")
    print(f"Last Close Price: {last_close:.2f}, Predicted Next Close: {next_close:.2f}")

    # Save prediction to CSV file including the yfinance timestamp in 12-hour format
    data = {
        "Timestamp": [timestamp],
        "Last_Close": [last_close],
        "Predicted_Next_Close": [next_close],
        "Direction": ["UP" if predicted_direction else "DOWN"],
        "Confidence": [prob]
    }

    df_log = pd.DataFrame(data)

    if not os.path.exists(CSV_FILE):
        df_log.to_csv(CSV_FILE, index=False, mode='w')
    else:
        df_log.to_csv(CSV_FILE, index=False, mode='a', header=False)

    return predicted_direction, prob, next_close

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

def train_model():
    """Enhanced training pipeline with model reuse"""
    model_path = os.path.join(model_dir, "xgboost_1m_one.model")
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    print("Training new model...")
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
            print(f"\nMean Absolute Error: {mae:.4f}")
            
            # Sample predictions
            print("\nSample Forecasts:")
            print(df_forecast[['timestamp', 'current_close', 'predicted_prob', 
                             'predicted_close', 'actual_next_close']].tail(5))
            
            # Visualization
            plt.figure(figsize=(12, 6))
            plt.plot(df_forecast['timestamp'], df_forecast['current_close'], label='Current Close')
            plt.plot(df_forecast['timestamp'], df_forecast['predicted_close'], 
                     label='Predicted Close', alpha=0.7)
            plt.plot(df_forecast['timestamp'], df_forecast['actual_next_close'], 
                     label='Actual Next Close', linestyle='--')
            plt.title('Price Forecast Performance')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
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
    # Optionally, train the model if not already trained
    train_model()
    
    # Start real-time forecasting every minute (if desired)
    schedule.every(1).minutes.do(real_time_forecast)
    
    # You can run the backtest separately if needed:
    # run_backtest()
    
    print("Starting scheduled real-time forecasting. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Real-time forecasting terminated.")
