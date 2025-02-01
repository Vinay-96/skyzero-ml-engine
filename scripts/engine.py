import numpy as np
import pandas as pd
import math
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Define the path where the model is saved
model_path = "./models/lstm_model.h5"

# 1. Load and Preprocess Data
def load_data():
    data = pd.read_csv("./data/NIFTY_BANK_Historical_PR.csv", parse_dates=["Date"], index_col="Date")
    return data

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler

def prepare_data(scaled_data, time_step=60):
    X, Y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        Y.append(scaled_data[i + time_step, 0])
    
    # Convert X and Y to numpy arrays
    X, Y = np.array(X), np.array(Y)
    
    # Reshape X to be 3D for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y

# 2. Build and Train Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, Y_train):
    model = build_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
    model.fit(X_train, Y_train, epochs=100, batch_size=32, callbacks=[early_stopping])
    model.save('./models/lstm_model.h5')
    return model

# 3. Make Predictions
def make_predictions(model, X, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# 4. Evaluate Model
def evaluate_model(predictions, actual_values):
    rmse = math.sqrt(mean_squared_error(actual_values, predictions))
    print(f"RMSE: {rmse}")

def plot_predictions(predictions, actual_values):
    plt.figure(figsize=(14, 5))
    plt.plot(actual_values, color='blue', label='Actual Trend')
    plt.plot(predictions, color='red', label='Predicted Trend')
    plt.title('Stock Price Prediction with LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# 5. Run All Steps
def main():
    # Load and preprocess data
    data = load_data()
    scaled_data, scaler = scale_data(data)
    X, Y = prepare_data(scaled_data)

    # Check if the model exists
    if os.path.exists(model_path):
        # Load the existing model
        model = load_model(model_path)
        print("Loaded existing model.")
    else:
        # Train a new model and save it
        model = train_model(X, Y)  # Assuming X, Y are your training data
        model.save(model_path)
        print("Trained and saved a new model.")

    # Make predictions
    predictions = make_predictions(model, X, scaler)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Close'], index=data.index[len(data) - len(predictions):])
    predictions_df.to_csv("./data/predictions.csv")
    print("Predictions saved to ./data/predictions.csv")

    # Adjust the length of actual and predicted values to ensure they match
    actual_values = data['Close'].values[60:len(predictions_df) + 60]  # Adjust actual values to match predictions length

    # Evaluate model
    evaluate_model(actual_values, predictions_df['Predicted_Close'].values)
    plot_predictions(predictions_df['Predicted_Close'].values, actual_values)


if __name__ == "__main__":
    main()
