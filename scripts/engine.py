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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Enhanced Configuration
model_dir = "./models"
TIMEFRAMES = ['1m']
PAST_BARS = 20
PREDICTION_THRESHOLD = 0.001
VALIDATION_WINDOWS = 5
FEATURE_NAMES_FILE = os.path.join(model_dir, "feature_names.json")
SCALER_FILE = os.path.join(model_dir, "scaler.pkl")

os.makedirs(model_dir, exist_ok=True)

# MongoDB setup
client = pymongo.MongoClient(os.getenv("DATABASE_URL"))
db = client['skyZero']

class FeatureCalculator:
    """Centralized feature calculation for consistency"""
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.prefix = f'{timeframe}_'
        
    def calculate_technical_indicators(self, df):
        """Consistent technical indicator calculation"""
        indicators = pd.DataFrame(index=df.index)
        
        # Price Action
        indicators[f'{self.prefix}HL_Ratio'] = df['high'] / df['low']
        indicators[f'{self.prefix}CO_Ratio'] = df['close'] / df['open']
        indicators[f'{self.prefix}True_Range'] = ta.volatility.AverageTrueRange(df.high, df.low, df.close).average_true_range()

        
        # Momentum
        indicators[f'{self.prefix}RSI'] = ta.momentum.RSIIndicator(df.close).rsi()
        indicators[f'{self.prefix}Stoch_RSI'] = ta.momentum.StochRSIIndicator(df.close).stochrsi()
        
        # Volume
        if 'volume' in df.columns:
            indicators[f'{self.prefix}OBV'] = ta.volume.OnBalanceVolumeIndicator(
                df.close, df.volume).on_balance_volume()
        
        # Trend
        for window in [5, 8, 13]:
            indicators[f'{self.prefix}EMA_{window}'] = ta.trend.EMAIndicator(
                df.close, window=window).ema_indicator()
        
        # Volatility
        bb = ta.volatility.BollingerBands(df.close)
        indicators[f'{self.prefix}BB_Width'] = (
            bb.bollinger_hband() - bb.bollinger_lband()) / df.close
        
        return indicators.dropna()

    def create_advanced_features(self, df):
        """Consistent advanced feature creation"""
        features = pd.DataFrame(index=df.index)
        
        # Price changes
        for i in range(1, PAST_BARS):
            features[f'return_lag_{i}'] = df['close'].pct_change(i)
        
        # Momentum
        for window in [3, 5, 8]:
            features[f'momentum_{window}'] = df['close'].diff(window)
        
        # Statistical features
        features['rolling_mean'] = df['close'].rolling(PAST_BARS).mean()
        features['rolling_std'] = df['close'].rolling(PAST_BARS).std()
        
        return features.dropna()

def load_data(timeframe):
    """Improved data loading with outlier handling"""
    collection = db[f'NIFTY_BANK_{timeframe}']
    data = list(collection.find({}, {'_id': 0, 'oneMCandles': 1})
                .sort("timestamp", -1).limit(5000))
    
    if not data:
        raise ValueError(f"No data found for {timeframe}")
    
    candles = data[0].get('oneMCandles', [])
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Improved outlier detection
    outlier_flags = pd.Series(False, index=df.index)
    for col in ['open', 'high', 'low', 'close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_flags |= (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
    
    return df[~outlier_flags].sort_index()

def preprocess_data(df, timeframe):
    """Full preprocessing pipeline with feature persistence"""
    calculator = FeatureCalculator(timeframe)
    
    # Calculate features
    indicators = calculator.calculate_technical_indicators(df)
    advanced_features = calculator.create_advanced_features(df)
    
    # Merge features
    df = df.join(indicators).join(advanced_features).dropna()
    
    # Create target
    target = df['close'].pct_change().shift(-1).dropna()
    df = df.loc[target.index]
    
    # Select features
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Save feature names
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(features, f)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    joblib.dump(scaler, SCALER_FILE)
    
    return X_scaled, target.values, scaler, features, df.index

def optimize_model(X_train, y_train):
    """Hyperparameter optimization"""
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.03],
        'subsample': [0.8, 0.9],
    }
    
    model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist')
    tscv = TimeSeriesSplit(n_splits=VALIDATION_WINDOWS)
    
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_

def train_model(timeframe):
    """Complete training pipeline"""
    model_path = os.path.join(model_dir, f"xgboost_{timeframe}.model")
    
    # Load data
    df = load_data(timeframe)
    X, y, scaler, features, _ = preprocess_data(df, timeframe)
    
    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train or load model
    if os.path.exists(model_path):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    else:
        model = optimize_model(X_train, y_train)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)
        model.save_model(model_path)
    
    # Validate
    validate_model(model, X_test, y_test)
    return model

def validate_model(model, X_test, y_test):
    """Comprehensive model validation"""
    preds = model.predict(X_test)
    
    print(f"\nValidation Metrics:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.6f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.6f}")
    print(f"R²: {r2_score(y_test, preds):.2f}")
    print(f"Direction Accuracy: {np.mean(np.sign(preds) == np.sign(y_test)):.2%}")

class TradingStrategy(bt.Strategy):
    """Consistent Backtrader strategy"""
    params = (
        ('risk_pct', 1),
        ('stop_loss', 0.5),
        ('take_profit', 1),
    )
    
    def __init__(self):
        self.model = xgb.XGBRegressor()
        self.model.load_model(os.path.join(model_dir, "xgboost_1m.model"))
        self.scaler = joblib.load(SCALER_FILE)
        with open(FEATURE_NAMES_FILE, 'r') as f:
            self.feature_names = json.load(f)
        
        self.calculator = FeatureCalculator('1m')
        self.order = None

    def next(self):
        if len(self.data) < PAST_BARS:
            return
        
        try:
            # Prepare features
            df = pd.DataFrame({
                'open': self.data.open.get(size=PAST_BARS),
                'high': self.data.high.get(size=PAST_BARS),
                'low': self.data.low.get(size=PAST_BARS),
                'close': self.data.close.get(size=PAST_BARS),
                'volume': self.data.volume.get(size=PAST_BARS)
            })
            
            indicators = self.calculator.calculate_technical_indicators(df)
            advanced = self.calculator.create_advanced_features(df)
            features = pd.concat([indicators, advanced], axis=1)[self.feature_names].tail(1)
            
            # Predict
            scaled = self.scaler.transform(features)
            pred = self.model.predict(scaled)[0]
            
            # Execute strategy
            self.execute_trades(pred)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")

    def execute_trades(self, prediction):
        """Risk-managed trade execution"""
        if self.order:
            return
            
        if prediction > PREDICTION_THRESHOLD:
            self.buy_signal()
        elif prediction < -PREDICTION_THRESHOLD:
            self.sell_signal()

    def buy_signal(self):
        cash = self.broker.getcash()
        price = self.data.close[0]
        size = (cash * self.params.risk_pct/100) / price
        self.order = self.buy(size=size)
        self.stop_price = price * (1 - self.params.stop_loss/100)
        self.target_price = price * (1 + self.params.take_profit/100)

    def sell_signal(self):
        cash = self.broker.getcash()
        price = self.data.close[0]
        size = (cash * self.params.risk_pct/100) / price
        self.order = self.sell(size=size)
        self.stop_price = price * (1 + self.params.stop_loss/100)
        self.target_price = price * (1 - self.params.take_profit/100)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED @ {order.executed.price:.2f}")
            else:
                print(f"SELL EXECUTED @ {order.executed.price:.2f}")

def run_backtest(timeframe):
    """Complete backtest execution"""
    cerebro = bt.Cerebro()
    
    # Add data
    data = load_data_for_backtrader(timeframe)
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(TradingStrategy)
    
    # Configure broker
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    print("Starting Backtest...")
    results = cerebro.run()
    
    # Show results
    strat = results[0]
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    print(f"Sharpe Ratio: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

def load_data_for_backtrader(timeframe):
    """Prepare Backtrader-compatible data feed"""
    df = load_data(timeframe)
    df['datetime'] = df.index
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    return bt.feeds.PandasData(dataname=df, datetime='datetime')

def schedule_jobs():
    """Job scheduling"""
    schedule.every(1).minutes.do(lambda: train_model('1m'))
    schedule.every(1).minutes.do(lambda: run_backtest('1m'))
    schedule.every().day.at("00:00").do(cleanup_old_results)

def cleanup_old_results():
    """Database maintenance"""
    for timeframe in TIMEFRAMES:
        db[f'forecasts_{timeframe}'].delete_many({
            'timestamp': {'$lt': datetime.now() - timedelta(days=7)}
        })

def main():
    """Main execution"""
    # Validate feature consistency
    for timeframe in TIMEFRAMES:
        df = load_data(timeframe)
        X, _, _, features, _ = preprocess_data(df, timeframe)
        
        if os.path.exists(FEATURE_NAMES_FILE):
            with open(FEATURE_NAMES_FILE) as f:
                saved_features = json.load(f)
                if features != saved_features:
                    raise ValueError("Feature mismatch detected! Retrain models.")
    
    # Initial training and backtest
    for timeframe in TIMEFRAMES:
        train_model(timeframe)
        run_backtest(timeframe)
    
    # Start scheduled jobs
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()