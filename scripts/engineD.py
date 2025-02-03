import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
df = pd.read_csv('./data/NIFTY BANK_Historical_PR.csv', parse_dates=['Date'])
df = df.sort_values('Date').set_index('Date')

# Feature Engineering
df['Momentum'] = df['Close'].pct_change(1).shift(-1)  # Next day's return
df['Momentum'] = np.where(df['Momentum'] > 0, 1, 0)  # 1=Up, 0=Down
df = df.dropna()

# Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Open','High','Low','Close']])

# Create sequences
def create_sequences(data, target, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps-1):
        X.append(data[i:(i+n_steps)])
        y.append(target[i+n_steps])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10  # Adjust based on experimentation
X, y = create_sequences(scaled_data, df[['Close','Momentum']].values, SEQ_LENGTH)

# Split data
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(2))  # Output both Close and Momentum

model.compile(optimizer='Adam', 
              loss=['mean_squared_error', 'binary_crossentropy'],
              metrics={'dense_1': 'mae', 'dense_2': 'accuracy'})

# Train model
history = model.fit(X_train, 
                    [y_train[:,0], y_train[:,1]], 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1)

# Evaluate
test_predict = model.predict(X_test)
predicted_close = test_predict[0].flatten()
predicted_momentum = np.round(test_predict[1].flatten())

# Calculate metrics
close_mae = np.mean(np.abs(predicted_close - y_test[:,0]))
momentum_acc = accuracy_score(y_test[:,1], predicted_momentum)

print(f"Close MAE: {close_mae:.4f}")
print(f"Momentum Accuracy: {momentum_acc:.2%}")

# Future prediction
def predict_next_day(model, last_sequence):
    last_sequence = last_sequence.reshape(1, SEQ_LENGTH, 4)
    prediction = model.predict(last_sequence)
    return prediction[0][0][0], np.round(prediction[1][0][0])

# Example usage
last_seq = scaled_data[-SEQ_LENGTH:]
next_close, next_momentum = predict_next_day(model, last_seq)
print(f"Predicted Close: {next_close}, Momentum: {'Up' if next_momentum else 'Down'}")