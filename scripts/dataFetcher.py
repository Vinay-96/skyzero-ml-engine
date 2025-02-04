import yfinance as yf

def fetch_and_save_data(ticker, period="max", filename="data.csv", interval="1d"):
    """
    Fetch historical data for a given ticker from Yahoo Finance and save it as a CSV file.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").
    - period (str): Data period to download (e.g., "1mo", "1y", "max"). Defaults to "max".
    - filename (str): The name of the CSV file to save the data to.
    - interval (str): Data interval (e.g., "1d", "1wk", "1mo"). Defaults to "1d".

    Returns:
    - None
    """
    try:
        # Download the historical data
        data = yf.download(ticker, period=period, interval=interval)
        
        if data.empty:
            print(f"No data found for ticker {ticker}. Please check the ticker symbol or parameters.")
            return

        # Save the data to a CSV file
        data.to_csv(filename)
        print(f"Data for {ticker} saved to {filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    # Fetch historical data for ^NSEBANK and save it as "NSEBANKa.csv"
    fetch_and_save_data("^NSEBANK", period="max", filename="NSEBANK_data.csv")
