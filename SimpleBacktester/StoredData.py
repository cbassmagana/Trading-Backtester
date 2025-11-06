import yfinance as yf
import pandas as pd
import os
from datetime import date

file_path = "SourceData/sp500_data.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

def get_current_date():
    return date.today().strftime('%Y-%m-%d')


# Updates the SP500 dataframe whenever necessary to accommodate more recent requests
def update_data(end_date: str):

    global df
    last_date = df.index[-1].strftime('%Y-%m-%d')

    if pd.Timestamp(end_date) > pd.Timestamp(last_date):

        new_data = yf.download("SPY", start=last_date, end=end_date, interval="1d", progress=False)
        new_data.columns = [col[0] if isinstance(col, tuple) else col for col in new_data.columns]

        new_data.rename(columns={
            'Close': 'Close',
            'High': 'High',
            'Low': 'Low',
            'Open': 'Open',
            'Volume': 'Volume'
        }, inplace=True)

        new_data = new_data[['Close', 'High', 'Low', 'Open', 'Volume']]

        if not new_data.empty:
            df = pd.concat([df, new_data]).drop_duplicates()
            df.sort_index(inplace=True)
            df.to_csv(file_path)


# Creates csv dataframe from scratch for the first time given the data to fetch and the file name to store it as
def create_sp500_csv_data(ticker: str, start_date: str, end_date: str, file_name: str, interval: str = "1d"):

    if os.path.exists(file_name):
        print(f"{file_name} already exists. Loading data from file.")
        data = pd.read_csv(file_name, index_col=0, parse_dates=True)
    else:
        print(f"{file_name} not found. Fetching data from API...")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        data.to_csv(file_name)
        print(f"Data saved to {file_name}.")

    return data


if __name__ == "__main__":
    update_data(get_current_date())
