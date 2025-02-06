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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, cross_val_predict
from scipy.signal import hilbert
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import shap

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
# 1. Enhanced Feature Engineering with Multi-Timeframe Support
# -------------------------------
class EnhancedFeatureCalculator:
    """Advanced feature engineering for price forecasting with multi-timeframe support"""
    def __init__(self, timeframe='1m'):
        # For multi-timeframe, these are used to create aggregated features
        self.base_resample_rules = {'5T': '5m', '15T': '15m'}
        self.prefix = '1m_' if timeframe == '1m' else f'{timeframe}_'
        self.timeframe = timeframe

    def calculate_core_features(self, df):
        """Robust feature engineering with validation on given dataframe"""
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
        
            # Cycle detection via Hilbert transform
            features[f'{self.prefix}Hilbert'] = np.unwrap(np.angle(hilbert(df.close)))
        
            # Advanced volatility: Keltner Channel
            keltner = ta.volatility.KeltnerChannel(df.high, df.low, df.close)
            features[f'{self.prefix}Keltner_Width'] = ((keltner.keltner_channel_hband() - 
                                                        keltner.keltner_channel_lband()) / df.close)
        
            # Lagged features
            for lag in [1, 3, 5, 8]:
                features[f'{self.prefix}lag_{lag}'] = df['close'].shift(lag)
            
            # New Enhanced Features:
            # Advanced Momentum: Vortex indicator
            features[f'{self.prefix}Vortex'] = ta.trend.VortexIndicator(df.high, df.low, df.close).vortex_indicator_diff()
            # Machine Learning-Generated Features: Residuals and FFT-based wavelets
            features[f'{self.prefix}Residuals'] = df.close - df.close.rolling(20).mean()
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
            
            # Forecast target (to be overwritten in preprocessing)
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            features['next_close'] = df['close'].shift(-1)  # Store actual next close
            
            # Remove any rows with NaN values that might remain
            return features.dropna()
        except Exception as e:
            print(f"Feature calculation error: {str(e)}")
            return pd.DataFrame()
    
    def calculate_cross_timeframe_features(self, df):
        """Add features aggregated from higher timeframes"""
        # Start with base timeframe features (using core features)
        features = self.calculate_core_features(df)
        
        # For each aggregation rule, calculate additional features and merge them.
        for rule, prefix in self.base_resample_rules.items():
            resampled_df = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).ffill()
            
            # Calculate features on the resampled data
            resampled_features = self.calculate_core_features(resampled_df)
            
            # Reindex to the original dataframe index using forward-fill and interpolation
            resampled_features = resampled_features.reindex(df.index).ffill().interpolate()
            
            # Merge features with the appropriate prefix
            features = features.join(
                resampled_features.add_prefix(f'{prefix}_')
            )
        # Ensure any remaining NaN or infinite values are removed
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        return features

# -------------------------------
# Trend Strength Target Function
# -------------------------------
def calculate_target(series):
    returns = series.pct_change(3).shift(-3)
    return (returns > returns.quantile(0.6)).astype(int)  # Top 40% moves

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_data(timeframe='1m'):
    """
    Load historical data from Yahoo Finance for training.
    For timeframe other than 1m, the interval parameter is adjusted.
    """
    symbol = "^NSEBANK"
    # Map timeframe to yfinance interval and period; for training we use 8d for 1m and longer periods for others.
    tf_mapping = {
        '1m': {"interval": "1m", "period": "8d"},
        '5m': {"interval": "5m", "period": "30d"},
        '15m': {"interval": "15m", "period": "30d"},
        '30m': {"interval": "30m", "period": "60d"},
        '60m': {"interval": "60m", "period": "60d"}
    }
    params = tf_mapping.get(timeframe, tf_mapping['1m'])
    df = yf.download(tickers=symbol, interval=params["interval"], period=params["period"])
    df.reset_index(inplace=True)
    # For some intervals, the datetime column might be named differently
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    else:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
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
    df.columns = [col.lower() for col in df.columns]
    return df.ffill().dropna()

def load_live_data(timeframe='1m', symbol="^NSEBANK"):
    """Fetch real-time price data from Yahoo Finance based on timeframe."""
    tf_mapping = {
        '1m': {"interval": "1m", "period": "1d"},
        '5m': {"interval": "5m", "period": "1d"},
        '15m': {"interval": "15m", "period": "1d"}
    }
    params = tf_mapping.get(timeframe, tf_mapping['1m'])
    df = yf.download(tickers=symbol, interval=params["interval"], period=params["period"])
    df.reset_index(inplace=True)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    else:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    return df.ffill().dropna()

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(df, timeframe='1m'):
    """Feature engineering pipeline using cross-timeframe features"""
    calculator = EnhancedFeatureCalculator(timeframe=timeframe)
    # Use the cross-timeframe features method
    features = calculator.calculate_cross_timeframe_features(df)
    
    # Overwrite the 'target' column with the trend strength target
    features['target'] = calculate_target(df.close)
    
    # Ensure no infinite or NaN values exist
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    
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
# 5. Training Process Optimization with Timeframe Support
# -------------------------------
def train_model(timeframe='1m', use_csv=False, csv_file_path="historical_data.csv"):
    """
    Train the model using data loaded from yfinance (default) or from a CSV file.
    Now supports timeframe-based training.
    
    Parameters:
        timeframe (str): One of '1m', '5m', '15m', etc.
    """
    model_path = os.path.join(model_dir, f"ensemble_model_{timeframe}.model")
    
    if os.path.exists(model_path):
        print(f"Loading existing ensemble model for timeframe {timeframe}...")
        model = joblib.load(model_path)
        return model

    print(f"Training new ensemble model for timeframe {timeframe}...")
    if use_csv:
        df = load_data_csv(csv_file_path)
    else:
        df = load_data(timeframe=timeframe)
        
    # Preprocess the data (feature engineering, scaling, etc.)
    X, y, scaler, feature_cols, next_closes = preprocess_data(df, timeframe=timeframe)

    # Check if we have enough samples after cleaning
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid samples available after preprocessing.")

    # Temporal split: use 80% of the data for training and 20% for testing
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Balance classes with SMOTE (ensure inputs contain no NaN/infs)
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Model configuration with hyperparameter tuning using GridSearchCV
    base_model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.7,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {timeframe}: {grid_search.best_params_}")

    # Evaluate on the hold-out test set
    preds = best_model.predict(X_test)
    print(f"\n=== Ensemble Model Validation for {timeframe} ===")
    print(classification_report(y_test, preds))

    # Save the trained model for future use
    joblib.dump(best_model, model_path)
    print(f"Ensemble model saved to {model_path}")
    return best_model

# -------------------------------
# 2. Temporal Hierarchy Ensemble Architecture
# -------------------------------
def create_temporal_ensemble():
    """Create a stacking ensemble using models trained on different timeframes"""
    ensemble = StackingClassifier(
        estimators=[
            ('1m_model', train_model(timeframe='1m')),
            ('5m_model', train_model(timeframe='5m')),
            ('15m_model', train_model(timeframe='15m'))
        ],
        final_estimator=LogisticRegression(),
        stack_method='predict_proba',
        passthrough=True
    )
    return ensemble

# -------------------------------
# 3. Adaptive Timeframe Selection
# -------------------------------
def optimize_timeframe():
    """Find optimal timeframe through grid search over multiple timeframes"""
    timeframes = ['1m', '5m', '15m', '30m', '60m']
    best_score = -np.inf
    best_tf = '1m'
    # Load some baseline data for evaluation; here we use 1m data features as template.
    df_base = load_data(timeframe='1m')
    X_full, y_full, _, _, _ = preprocess_data(df_base, timeframe='1m')
    
    for tf in timeframes:
        model = train_model(timeframe=tf)
        scores = cross_val_score(
            model, X_full, y_full, 
            cv=TimeSeriesSplit(3),
            scoring='f1_macro'
        )
        score_mean = np.mean(scores)
        print(f"Timeframe: {tf}, F1 Score: {score_mean:.3f}")
        if score_mean > best_score:
            best_score = score_mean
            best_tf = tf
            
    print(f"Optimal timeframe: {best_tf} (F1: {best_score:.3f})")
    return best_tf

# -------------------------------
# 4. Implementation Roadmap: Data Preparation and Feature Fusion
# -------------------------------
def prepare_multi_tf_data(symbol="^NSEBANK"):
    """Create aligned multi-timeframe dataset"""
    base_df = yf.download(symbol, period="60d", interval="1m")
    base_df.reset_index(inplace=True)
    if "Datetime" in base_df.columns:
        base_df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    else:
        base_df.rename(columns={"Date": "timestamp"}, inplace=True)
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    base_df.set_index("timestamp", inplace=True)
    multi_tf_data = {
        '1m': base_df,
        '5m': base_df.resample('5T').last().ffill(),
        '15m': base_df.resample('15T').last().ffill()
    }
    return multi_tf_data

def create_fused_features(multi_tf_data):
    """Combine features from multiple timeframes"""
    fused_features = pd.DataFrame(index=multi_tf_data['1m'].index)
    
    for tf, df in multi_tf_data.items():
        calculator = EnhancedFeatureCalculator(timeframe=tf)
        features = calculator.calculate_core_features(df)
        # Align features on the fused_features index and add prefix with timeframe label
        fused_features = fused_features.join(
            features.add_prefix(f'{tf}_'),
            how='outer'
        )
    return fused_features.ffill().dropna()

# -------------------------------
# Dynamic Timeframe Weighting: Meta-Ensemble
# -------------------------------
class TimeframeWeightedEnsemble:
    """Ensemble with learned timeframe weights for combining predictions"""
    def __init__(self):
        self.models = {
            '1m': joblib.load(os.path.join(model_dir, "ensemble_model_1m.model")),
            '5m': joblib.load(os.path.join(model_dir, "ensemble_model_5m.model")),
            '15m': joblib.load(os.path.join(model_dir, "ensemble_model_15m.model"))
        }
        self.weight_optimizer = LogisticRegression()

    def fit(self, X, y):
        # Train meta-model on cross-validated predictions from each timeframe
        val_preds = []
        for tf, model in self.models.items():
            # Here, X is assumed to be a dict of features for each timeframe
            preds = cross_val_predict(model, X[tf], y, cv=5)
            val_preds.append(preds)
        self.weight_optimizer.fit(np.column_stack(val_preds), y)
        
    def predict(self, X):
        preds = [model.predict(X[tf]) for tf, model in self.models.items()]
        return self.weight_optimizer.predict(np.column_stack(preds))

# -------------------------------
# 6. Validation Strategy: Cross-Timeframe Backtesting
# -------------------------------
def cross_timeframe_backtest():
    """Validate models across multiple time horizons using walk-forward validation"""
    results = {}
    
    for tf in ['1m', '5m', '15m']:
        model = joblib.load(os.path.join(model_dir, f"ensemble_model_{tf}.model"))
        df = load_data(timeframe=tf)
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            
            # Use core features for training; note: dropping target column as before.
            X_train, y_train, _, _, _ = preprocess_data(train, timeframe=tf)
            X_test, y_test, _, _, _ = preprocess_data(test, timeframe=tf)
            
            # Fit model on training split (using sample weighting for recent samples)
            sample_weights = np.exp(np.linspace(-1, 0, len(y_train)))
            model.fit(X_train, y_train, sample_weight=sample_weights)
            preds = model.predict(X_test)
            scores.append(f1_score(y_test, preds))
        
        results[tf] = np.mean(scores)
    
    print("Cross-Timeframe Validation Results:")
    print(pd.DataFrame.from_dict(results, orient='index', columns=['F1 Score']))
    
# -------------------------------
# 7. Real-Time Adaptation: Multi-Timeframe Forecasting
# -------------------------------
def real_time_forecast():
    """Enhanced prediction with multi-timeframe analysis and fusion"""
    # Define weights for each timeframe
    timeframe_weights = {
        '1m': 0.4,
        '5m': 0.3,
        '15m': 0.3
    }
    
    predictions = []
    
    for tf, weight in timeframe_weights.items():
        model_path = os.path.join(model_dir, f"ensemble_model_{tf}.model")
        if not os.path.exists(model_path):
            print(f"Model for timeframe {tf} not found.")
            continue
        model = joblib.load(model_path)
        df = load_live_data(timeframe=tf)
        # IMPORTANT: use the same feature extraction method as in training:
        calculator = EnhancedFeatureCalculator(timeframe=tf)
        # Use cross-timeframe feature extraction (instead of core only)
        features = calculator.calculate_cross_timeframe_features(df)
    
        if features.empty:
            print(f"Feature extraction failed for timeframe {tf}.")
            continue
    
        # Use the saved feature names and scaler from training
        with open(FEATURE_NAMES_FILE, 'r') as f:
            feature_names = json.load(f)
        scaler = joblib.load(SCALER_FILE)
    
        # Make sure the required features exist in the current dataframe
        if not set(feature_names).issubset(features.columns):
            print(f"Not all features found for timeframe {tf}. Skipping.")
            continue
            
        latest_features = features[feature_names].iloc[-1].values.reshape(1, -1)
        latest_scaled = scaler.transform(latest_features)
    
        prob = model.predict_proba(latest_scaled)[0][1]
        predictions.append((prob, weight))
    
    if not predictions:
        print("No predictions available from any timeframe.")
        return None, None, None
    
    # Calculate weighted forecast probability
    weighted_pred = sum(p * w for p, w in predictions) / sum(timeframe_weights.values())
    
    # Use 1m live data for price extraction
    df_1m = load_live_data(timeframe='1m')
    last_close = df_1m["close"].iloc[-1]
    last_high = df_1m["high"].iloc[-1]
    last_low = df_1m["low"].iloc[-1]
    
    # Calculate dynamic volatility (fallback if high-low is zero)
    if (last_high - last_low) == 0:
        base_volatility = last_close * 0.0005  
    else:
        base_volatility = (last_high - last_low) * 0.5
    
    direction_multiplier = 1 if weighted_pred > 0.5 else -1
    price_change = base_volatility * weighted_pred * direction_multiplier
    next_close = last_close + price_change
    
    # Get current timestamp in a friendly format
    last_timestamp = df_1m.index[-1]
    timestamp = last_timestamp.strftime("%Y-%m-%d %I:%M:%S %p")
    
    print(f"{timestamp} | Consensus Forecast: {weighted_pred:.2%} probability")
    print(f"Last Close Price: {last_close:.2f}, Predicted Next Close: {next_close:.2f}")
    
    # Save prediction to CSV file
    data = {
        "Timestamp": [timestamp],
        "Last_Close": [last_close],
        "Predicted_Next_Close": [next_close],
        "Consensus_Probability": [f"{weighted_pred * 100:.2f}%"]
    }
    
    df_log = pd.DataFrame(data)
    if not os.path.exists(CSV_FILE):
        df_log.to_csv(CSV_FILE, index=False, mode='w')
    else:
        df_log.to_csv(CSV_FILE, index=False, mode='a', header=False)
    
    # Append to global live predictions for plotting
    live_predictions.append({
        "timestamp": last_timestamp,
        "actual_close": last_close,
        "forecast_close": next_close
    })
    
    # Update the live plot (adjust output path as needed)
    update_live_plot(output_file="/plots/live_plot.png")
    
    return weighted_pred, last_close, next_close

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
        
        self.calculator = EnhancedFeatureCalculator(timeframe='1m')
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
            calculator = EnhancedFeatureCalculator(timeframe='1m')
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
    df = load_data(timeframe='1m').reset_index()
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
# Additional Utility: SHAP Feature Importance Analysis
# -------------------------------
def analyze_timeframe_importance(model, X):
    """SHAP-based feature importance analysis"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.title('Timeframe Feature Importance')
    plt.show()

# -------------------------------
# Dynamic Position Sizing Utility
# -------------------------------
def dynamic_position_size(base_size, predictions):
    """Adjust trade size based on consensus confidence"""
    confidence = max(predictions) - min(predictions)
    return base_size * (0.5 + confidence * 0.5)

# -------------------------------
# Main Execution Block
# -------------------------------
if __name__ == "__main__":
    # Example: Optimize timeframe based on F1 score (prints optimal timeframe)
    optimal_tf = optimize_timeframe()
    
    # Train or load the ensemble model for the optimal timeframe (or use multiple below)
    model_1m = train_model(timeframe='1m', use_csv=False)
    model_5m = train_model(timeframe='5m', use_csv=False)
    model_15m = train_model(timeframe='15m', use_csv=False)
    
    # Optionally, create a temporal ensemble from different timeframes
    ensemble_model = create_temporal_ensemble()
    ensemble_model_path = os.path.join(model_dir, "temporal_ensemble.model")
    joblib.dump(ensemble_model, ensemble_model_path)
    print(f"Temporal ensemble model saved to {ensemble_model_path}")
    
    # Optionally, run cross-timeframe backtesting
    # cross_timeframe_backtest()
    
    # run_backtest()  # Uncomment to run backtesting with Backtrader
    
    # Schedule the enhanced real-time forecasting job to run every minute.
    schedule.every(1).minutes.do(real_time_forecast)
    
    print("Starting scheduled real-time forecasting. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Real-time forecasting terminated.")
