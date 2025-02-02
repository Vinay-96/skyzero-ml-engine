import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import ta
import warnings
warnings.filterwarnings("ignore")

# Configuration
data_dir = "./data"
model_dir = "./models"
TIMEFRAMES = ['1m', '5m', '10m', '15m']
PAST_BARS = 5  # Use last 5 candles as input features

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

def load_data(timeframe):
    """Load data for a specific timeframe"""
    file_path = os.path.join(data_dir, f'NIFTY_BANK_{timeframe}.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    df = df[['open', 'high', 'low', 'close']].sort_index()
    
    if df.empty:
        raise ValueError(f"CSV file {file_path} is empty")

    return df

def calculate_technical_indicators(df, timeframe):
    """Calculate technical indicators for a given timeframe"""
    prefix = f'{timeframe}_'
    indicators = pd.DataFrame(index=df.index)

    # CPR Levels
    indicators[f'{prefix}CPR_Pivot'] = (df.high + df.low + df.close) / 3
    indicators[f'{prefix}CPR_BC'] = (df.high + df.low) / 2
    indicators[f'{prefix}CPR_TC'] = 2 * indicators[f'{prefix}CPR_Pivot'] - indicators[f'{prefix}CPR_BC']

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df.close, window=20)
    indicators[f'{prefix}BB_High'] = bb.bollinger_hband()
    indicators[f'{prefix}BB_Low'] = bb.bollinger_lband()

    # ATR (Volatility Indicator)
    indicators[f'{prefix}ATR'] = ta.volatility.AverageTrueRange(df.high, df.low, df.close).average_true_range()

    # RSI & MACD
    indicators[f'{prefix}RSI'] = ta.momentum.RSIIndicator(df.close).rsi()
    macd = ta.trend.MACD(df.close)
    indicators[f'{prefix}MACD'] = macd.macd()
    indicators[f'{prefix}MACD_Signal'] = macd.macd_signal()

    # Moving Averages
    for window in [5, 20, 50]:
        indicators[f'{prefix}MA_{window}'] = df.close.rolling(window).mean()

    # Difference Between MAs (Momentum Feature)
    indicators[f'{prefix}MA_Diff_5_20'] = indicators[f'{prefix}MA_5'] - indicators[f'{prefix}MA_20']

    # Time-Based Features
    indicators['hour'] = df.index.hour
    indicators['minute'] = df.index.minute

    return indicators.dropna()

def create_lag_features(df, past_bars):
    """Add previous bars' close prices as features"""
    for i in range(1, past_bars + 1):
        df[f'close_lag_{i}'] = df['close'].shift(i)
    return df.dropna()

def preprocess_data(df, timeframe):
    """Prepare data for XGBoost training"""
    # Calculate indicators
    indicators = calculate_technical_indicators(df, timeframe)
    df = df.merge(indicators, left_index=True, right_index=True, how='left')

    # Add lag features
    df = create_lag_features(df, PAST_BARS)

    # Drop NaN values
    df.dropna(inplace=True)

    # Select features (excluding raw prices)
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close']]
    if not features:
        raise ValueError(f"No features available for training {timeframe}")

    # Target is next period's close price
    target = df['close'].shift(-1)
    
    # Align features and target
    df_features = df[features].ffill().dropna()
    target = target.loc[df_features.index]

    # Remove rows with missing targets
    valid_indices = target.notna()
    X = df_features[valid_indices]
    y = target[valid_indices]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler, X.columns.tolist(), df.index[valid_indices]

def train_and_save_model(X_train, y_train, X_test, y_test, timeframe):
    """Train and save an XGBoost model for a specific timeframe"""
    model_path = os.path.join(model_dir, f"xgboost_{timeframe}.model")

    if os.path.exists(model_path):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"\nLoaded existing model for {timeframe}")
    else:
        print(f"\nTraining new model for {timeframe}...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1200,  # Increased estimators
            learning_rate=0.005,  # Lower learning rate
            max_depth=8,  # Increased depth for better learning
            subsample=0.9,
            colsample_bytree=0.9,
            early_stopping_rounds=50
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        model.save_model(model_path)

    return model

def forecast_next_period(model, X_last, scaler, feature_names):
    """Forecast the next period's close price"""
    X_last_scaled = scaler.transform([X_last])
    y_pred = model.predict(X_last_scaled)
    return y_pred[0]

def main():
    predictions = {}

    for timeframe in TIMEFRAMES:
        print(f"\nProcessing {timeframe} timeframe...")

        df = load_data(timeframe)
        X, y, scaler, feature_names, timestamps = preprocess_data(df, timeframe)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train and save model
        model = train_and_save_model(X_train, y_train, X_test, y_test, timeframe)

        # Forecast next period
        X_last = X[-1]
        forecast_price = forecast_next_period(model, X_last, scaler, feature_names)

        predictions[timeframe] = forecast_price
        print(f"\nForecasted next close price for {timeframe}: {forecast_price:.2f}")

    print("\nFinal Forecasts:", predictions)

if __name__ == "__main__":
    main()
