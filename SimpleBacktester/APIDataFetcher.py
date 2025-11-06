import yfinance as yf
from datetime import datetime, timedelta

# Fetches historical stock data for a given stock ticker and date range using yfinance, returns None if error
def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str):

    try:
        date_end_formatted = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = str(date_end_formatted + timedelta(1))[0:10]
        stock_data = yf.download(ticker, start = start_date, end = end_date, interval = interval)

        if stock_data.empty:
            raise ValueError(f"No data returned for ticker '{ticker}' in the specified date range.")

        stock_data['Ticker'] = ticker
        return stock_data

    except Exception as e:
        if e:
            return None
