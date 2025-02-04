# Use the official TensorFlow Docker image with Python
FROM tensorflow/tensorflow:latest-py3

# Set the working directory for your code inside the container
WORKDIR /app

# Install other dependencies you might need
RUN pip install --upgrade pip
RUN pip install pandas numpy matplotlib scikit-learn xgboost ta python-dotenv schedule backtrader imblearn pymongo seaborn yfinance

# Copy your local code into the container's working directory
COPY . /app

# Set the default command to run your script
CMD ["python", "engine.py"]
