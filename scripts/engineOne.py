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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy.signal import hilbert
from imblearn.over_sampling import SMOTE
import warnings

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

# MongoDB setup
client = pymongo.MongoClient(os.getenv("DATABASE_URL"))
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
            """Engineered features for better predictability"""
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
            features[f'{self.prefix}Keltner_Width'] = (keltner.keltner_channel_hband() - 
            keltner.keltner_channel_lband()) / df.close
        
            # Lagged features
            for lag in [1, 3, 5, 8]:
                features[f'{self.prefix}lag_{lag}'] = df['close'].shift(lag)
            
            # Forecast target: 1 if next candle is up, 0 otherwise
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            return features.dropna()
        except Exception as e:
            print(f"Feature calculation error: {str(e)}")
            return pd.DataFrame()

def load_data():
    """Load and clean 1-minute timeframe data"""
    collection = db['NIFTY_BANK_1m']
    data = list(collection.find({}, {'_id': 0, 'oneMCandles': 1})
    .sort("timestamp", -1).limit(10000))
    
    if not data:
        raise ValueError("No data found for 1m timeframe")
    
    candles = data[0].get('oneMCandles', [])
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop(columns=['volume'])  # Explicitly remove volume column
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df.sort_index().ffill().dropna()

def preprocess_data(df):
    """Feature engineering pipeline"""
    calculator = EnhancedFeatureCalculator()
    features = calculator.calculate_core_features(df)
    
    # Save feature names
    feature_cols = [col for col in features.columns if col != 'target']
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_cols, f)
    
    # Handle class imbalance
    X = features[feature_cols]
    y = features['target']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    
    return X_scaled, y.values, scaler, feature_cols, X.index

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
    """Enhanced training pipeline"""
    model_path = os.path.join(model_dir, "xgboost_1m_one.model")
    
    # Force fresh training
    if os.path.exists(model_path):
        os.remove(model_path)
    
    df = load_data()
    X, y, scaler, features, _ = preprocess_data(df)
    
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
    
    print("\n=== Advanced Validation ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds))
    
    # Feature importance
    importance = model.get_booster().get_score(importance_type='weight')
    print("\nTop 10 Features:")
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feat}: {score}")

class ForecastStrategy(bt.Strategy):
    """Optimized trading strategy with risk management"""
    params = (
        ('risk_pct', 0.5),
        ('stop_loss', 0.3),
        ('take_profit', 0.9),
        ('confidence_threshold', 0.7)
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

    def next(self):
        if len(self.data) < PAST_BARS*2 + 10:
            return
        
        try:
            # Prepare forecasting features
            df = pd.DataFrame({
                'open': self.data.open.get(size=PAST_BARS),
                'high': self.data.high.get(size=PAST_BARS),
                'low': self.data.low.get(size=PAST_BARS),
                'close': self.data.close.get(size=PAST_BARS)
            })

            if len(df) < PAST_BARS:
                return
            
            features = self.calculator.calculate_core_features(df)
            if features.empty:
                return
            

            # Use iloc with explicit check
            if len(features) >= 1:
                current_features = features[self.feature_names].iloc[-1].values.reshape(1, -1)
            else:
                return
            # Predict with confidence
            scaled = self.scaler.transform(current_features)
            prob = self.model.predict_proba(scaled)[0][1]
            
            # Execute trades
            if prob > self.params.confidence_threshold:
                self.execute_trades(1, prob)
            elif prob < (1 - self.params.confidence_threshold):
                self.execute_trades(-1, prob)
        except IndexError as ie:
            print(f"Index error prevented: {str(ie)}")

        except Exception as e:
            print(f"Prediction error: {str(e)}")

    def execute_trades(self, direction, confidence):
        """Sophisticated position sizing"""
        if self.order:
            return
            
        price = self.data.close[0]
        size = (self.broker.getcash() * self.params.risk_pct/100) / price
        
        if direction == 1:
            self.order = self.buy(size=size)
            print(f"LONG @ {price:.2f} | Confidence: {confidence:.2%}")
        else:
            self.order = self.sell(size=size)
            print(f"SHORT @ {price:.2f} | Confidence: {confidence:.2%}")

        # Dynamic exits
        self.stop_price = price * (1 - direction*self.params.stop_loss/100)
        self.target_price = price * (1 + direction*self.params.take_profit/100)

    def notify_trade(self, trade):
        """Enhanced trade tracking"""
        if trade.isclosed:
            profit_pct = trade.pnl / trade.size
            self.trade_log.append({
                'entry': trade.price,
                'exit': trade.price + trade.pnl,
                'duration': trade.barlen,
                'profit': profit_pct
            })
            print(f"Trade closed | Profit: {profit_pct:.2%} | Duration: {trade.barlen} bars")

def run_backtest():
    """Optimized backtesting engine"""
    cerebro = bt.Cerebro(optreturn=False)
    
    # Add data
    df = load_data().reset_index()
    data = bt.feeds.PandasData(dataname=df, datetime='timestamp')
    cerebro.adddata(data)
    
    # Configure strategy
    cerebro.addstrategy(ForecastStrategy)
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
    print("\n=== Final Performance ===")
    print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        print(f"Sharpe Ratio: {sharpe:.2f}")
    except (TypeError, AttributeError):
        print("Sharpe Ratio: N/A (no trades)")
    
    try:
        drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        print(f"Max Drawdown: {drawdown:.2f}%")
    except KeyError:
        print("Max Drawdown: N/A")

    print(f"Sharpe Ratio: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    
    # Safe trade analysis
    trade_stats = strat.analyzers.trades.get_analysis()
    total_trades = trade_stats.total.closed if hasattr(trade_stats, 'total') else 0
    print(f"\nTotal Trades: {total_trades}")
    
    if total_trades > 0:
        win_rate = trade_stats.won.total/total_trades
        print(f"Win Rate: {win_rate:.2%}")
    else:
        print("Win Rate: N/A")

def main():
    """Enhanced execution with validation"""
    # Clear models only if exists
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".model") or f.endswith(".pkl") or f.endswith(".json"):
                os.remove(os.path.join(model_dir, f))
    
    try:
        train_model()
        run_backtest()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()