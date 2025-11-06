import pandas as pd
from datetime import datetime
import locale
from SimpleBacktester import APIDataFetcher as APIFetcher
from SimpleBacktester import StockPriceFetcher as PriceFetcher
from SimpleBacktester import Strategy
from SimpleBacktester.Strategies.Strategy1 import Strategy1
from SimpleBacktester.Strategies.Strategy2 import Strategy2
from SimpleBacktester.Strategies.Strategy3 import Strategy3
from SimpleBacktester.Strategies.Strategy4 import Strategy4
from SimpleBacktester import StoredData

file_path = "SourceData/sp500_data.csv"
df = pd.read_csv(file_path)

inflation_file_path = "SourceData/inflation_data.csv"
inflation_df = pd.read_csv(inflation_file_path)

starting_balance = 1_000_000.00

# Acquires full result report of specific trading simulation
def simulate_result(action_stock: pd.DataFrame, indicator_stock: pd.DataFrame, strat: Strategy,
                    start_date: str, end_date: str):

    final_value = strat.apply_strategy(action_stock, indicator_stock)

    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])
    start_month = int(start_date[5:7])
    end_month = int(end_date[5:7])
    start_day = int(start_date[8:10])
    end_day = int(end_date[8:10])

    period_in_years = (end_year - start_year) + ((end_month - start_month)/12) + ((end_day - start_day)/365)

    portfolio_yield_ratio = final_value/1000000
    portfolio_yearly_percent = ((portfolio_yield_ratio ** (1 / period_in_years)) - 1) * 100

    sp500_yield_ratio = PriceFetcher.get_sp500_close_price(end_date) / PriceFetcher.get_sp500_open_price(start_date)
    sp500_yearly_percent = ((sp500_yield_ratio**(1/period_in_years)) - 1) * 100

    inflation_ratio = 1.00
    if start_year != end_year:

        for year in range(start_year, end_year + 1):
            if year == start_year:
                first_year_portion = (12-start_month)/12 + (30-start_day)/365
                inflation_ratio *= (1 + (PriceFetcher.get_inflation(year) * first_year_portion /100))
            elif year == end_year:
                last_year_portion = (start_month - 1)/12 + (start_day - 1)/365
                inflation_ratio *= (1 + (PriceFetcher.get_inflation(year) * last_year_portion /100))
            else:
                inflation_ratio *= (1 + (PriceFetcher.get_inflation(year) /100))

    else:
        inflation_ratio = 1 + (PriceFetcher.get_inflation(start_year)/100 * period_in_years)

    inflation_yearly_percent = ((inflation_ratio**(1/period_in_years)) - 1) * 100

    total_yield_after_inflation = portfolio_yield_ratio/inflation_ratio
    yearly_percent_after_inflation = (total_yield_after_inflation**(1/period_in_years) - 1) * 100

    sp500_yield_after_inflation = sp500_yield_ratio/inflation_ratio
    sp500_yearly_percent_after_inflation = (sp500_yield_after_inflation ** (1 / period_in_years) - 1) * 100

    raw_stock_growth = action_stock.iloc[-1]['Close'].iloc[0] / action_stock.iloc[0]['Open'].iloc[0]

    print_results(final_value, start_date, end_date, period_in_years, portfolio_yield_ratio, portfolio_yearly_percent,
                  sp500_yield_ratio, sp500_yearly_percent, inflation_ratio, inflation_yearly_percent,
                  total_yield_after_inflation, yearly_percent_after_inflation, sp500_yield_after_inflation,
                  sp500_yearly_percent_after_inflation, raw_stock_growth)


# Prints full result report of specific trading simulation
def print_results(final_value, start_date, end_date, period_in_years, portfolio_growth_ratio, portfolio_yearly_percent,
                  sp500_growth_ratio, sp500_yearly_percent, inflation_ratio, inflation_yearly_percent,
                  total_yield_after_inflation, yearly_percent_after_inflation, sp500_yield_after_inflation,
                  sp500_yearly_percent_after_inflation, raw_stock_growth):

    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    print("")
    print("REPORT:")

    print("")
    print("After starting with $1,000,000.00, the portfolio is now worth:", locale.currency(final_value, grouping=True))
    print("Buying and holding this stock would have produced a value of:",
          locale.currency(starting_balance * raw_stock_growth, grouping=True))
    print("Buying and holding the SP500 would have produced a value of:",
          locale.currency(starting_balance * sp500_growth_ratio, grouping=True))
    print("")

    print("Start date: ", start_date)
    print ("End date: ", end_date)
    print("Period length: ", "{:.5f}".format(period_in_years), "years")
    print("")

    print("Total portfolio growth ratio: ", "{:.5f}".format(portfolio_growth_ratio))
    print("Annual portfolio return: ", "{:.2f}%".format(portfolio_yearly_percent))
    print("")

    print("Total SP500 growth ratio: ", "{:.5f}".format(sp500_growth_ratio))
    print("Annual SP500 return: ", "{:.2f}%".format(sp500_yearly_percent))
    print("")

    print("Total inflation ratio: ", "{:.5f}".format(inflation_ratio))
    print("Annual inflation rate: ", "{:.2f}%".format(inflation_yearly_percent))
    print("")

    print("Total portfolio return after inflation: ", "{:.5f}".format(total_yield_after_inflation))
    print("Annual portfolio return after inflation: ", "{:.2f}%".format(yearly_percent_after_inflation))
    print("")

    print("Total SP500 return after inflation: ", "{:.5f}".format(sp500_yield_after_inflation))
    print("Annual SP500 return after inflation: ", "{:.2f}%".format(sp500_yearly_percent_after_inflation))
    print("")


# User entry point into program
if __name__ == "__main__":

    print("")

    while True:
        starting_date = input("Enter the starting date for your trading window in the form YYYY-MM-DD:  ").strip()
        try:
            start_date_obj = datetime.strptime(starting_date, '%Y-%m-%d')
            if start_date_obj < datetime(1994, 1, 1):
                print("Starting date cannot be before 1994. Please try again.")
            elif start_date_obj > datetime.now():
                print("Starting date cannot be in the future. Please try again.")
            else:
                break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    print("")

    while True:
        ending_date = input("Enter the ending date for your trading window in the form YYYY-MM-DD:  ").strip()
        try:
            end_date_obj = datetime.strptime(ending_date, '%Y-%m-%d')
            if end_date_obj < datetime(1994, 1, 1):
                print("Ending date cannot be before 1994. Please try again.")
            elif end_date_obj > datetime.now():
                print("Ending date cannot be in the future. Please try again.")
            elif end_date_obj <= start_date_obj:
                print("Ending date must be after the starting date. Please try again.")
            else:
                break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    print("")

    while True:
        action_ticker = input("Enter the ticker of the stock you are looking to trade:  ").strip()
        if action_ticker:
            print("")
            action_dataframe = APIFetcher.fetch_stock_data(action_ticker, starting_date, ending_date, "1d")
            if action_dataframe is not None:
                break
            else:
                print("No data was returned for this ticker in the specified date range. Please try again.")
        else:
            print("Ticker cannot be empty. Please try again.")

    print("")

    while True:
        indicator_ticker = input("Enter the ticker of any indicator stock you wish to use, or 'n' for none:  ").strip()
        if indicator_ticker == 'n':
            indicator_dataframe = None
            break
        elif indicator_ticker:
            print("")
            indicator_dataframe = APIFetcher.fetch_stock_data(indicator_ticker, starting_date, ending_date, "1d")
            if indicator_dataframe is not None:
                break
            else:
                print("No data was returned for this ticker in the specified date range. Please try again.")
        else:
            print("Ticker cannot be empty. Please try again.")


    StoredData.update_data(ending_date)

    print("")

    while True:
        strategy_index = input("Enter the index number of your desired trading strategy. \n"
                               "Enter 1 for strategy1: Simple Buy and Hold \n"
                               "Enter 2 for strategy2: Buy Dips and Hold \n"
                               "Enter 3 for strategy3: Buy Dips, Sell Spikes \n"
                               "Enter 4 for strategy4: Buy Indicator Spikes\n"
                               "Enter your selection here:  ").strip()
        if strategy_index in {'1', '2', '3', '4'}:
            break
        else:
            print("Invalid entry. Please enter a number between 1 and 4.")
            print("")

    print("")

    if strategy_index == '1':
        strategy = Strategy1(starting_balance)
    elif strategy_index == '2':
        strategy = Strategy2(starting_balance)
    elif strategy_index == '3':
        strategy = Strategy3(starting_balance)
    else:
        strategy = Strategy4(starting_balance)

    try:
        simulate_result(action_dataframe, indicator_dataframe, strategy, starting_date, ending_date)
    except Exception as e:
        print(f"An unexpected error occurred during the simulation. Error message: {e}.")