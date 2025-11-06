import pandas as pd
from datetime import datetime, timedelta

file_path = "SourceData/sp500_data.csv"
df = pd.read_csv(file_path)

inflation_file_path = "SourceData/inflation_data.csv"
inflation_df = pd.read_csv(inflation_file_path)

# Returns the open price of the SP500 on a given defined date, taking the next open price if needed
def get_sp500_open_price(date):

    row = df[df['Date'] == date]

    if not row.empty:
        return row['Open'].values[0]

    target_date = datetime.strptime(date, "%Y-%m-%d")

    for i in range(10):
        next_date = target_date + timedelta(days=i)
        temp_row = df[df['Date'] == str(next_date)[0:10]]
        if not temp_row.empty:
            return temp_row['Open'].values[0]

    return None


# Returns the close price of the SP500 on a given defined date, taking the prev close price if needed
def get_sp500_close_price(date):

    row = df[df['Date'] == date]

    if not row.empty:
        return row['Close'].values[0]

    target_date = datetime.strptime(date, "%Y-%m-%d")

    for i in range(10):
        next_date = target_date - timedelta(days=i)
        temp_row = df[df['Date'] == str(next_date)[0:10]]
        if not temp_row.empty:
            return temp_row['Close'].values[0]

    return None


# Returns the inflation rate of a given defined year ('YYYY')
def get_inflation(year):

    row = inflation_df[inflation_df['Year'] == year]

    if not row.empty:
        return row['Rate'].values[0]
    else:
        return None
