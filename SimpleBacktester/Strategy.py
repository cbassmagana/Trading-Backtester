from abc import ABC, abstractmethod
import pandas as pd
from SimpleBacktester import StockPriceFetcher as fetcher


class Strategy(ABC):

    # Initializes an instance of a strategy object with a given starting amount available to invest
    def __init__(self, initial_investment: float):
        self.stock_shares = 0
        self.sp500_shares = 0
        self.available_cash = initial_investment

    # Buys certain number of stock shares
    def buy_stock_shares(self, num_stock_shares_buying, stock_price):
        dollar_amount = num_stock_shares_buying * stock_price
        self.stock_shares = self.stock_shares + (min(dollar_amount, self.available_cash) / stock_price)
        self.available_cash = max(0, self.available_cash - dollar_amount)

    # Buys certain dollar amount of stock holdings
    def buy_stock_dollars(self, dollar_amount, stock_price):
        self.stock_shares = self.stock_shares + (min(dollar_amount, self.available_cash) / stock_price)
        self.available_cash = max(0, self.available_cash - dollar_amount)

    # Sells certain number of stock shares
    def sell_stock_shares(self, num_stock_shares_selling, stock_price):
        self.available_cash = self.available_cash + (min(self.stock_shares, num_stock_shares_selling) * stock_price)
        self.stock_shares = max(0, self.stock_shares - num_stock_shares_selling)

    # Sells certain dollar amount of stock holdings
    def sell_stock_dollars(self, dollar_amount, stock_price):
        num_stock_shares_selling = dollar_amount / stock_price
        self.available_cash = self.available_cash + (min(self.stock_shares, num_stock_shares_selling) * stock_price)
        self.stock_shares = max(0, self.stock_shares - num_stock_shares_selling)

    # Buys certain number of sp500 shares
    def buy_sp500_shares(self, num_sp500_shares_buying, date, is_open_price):

        if is_open_price:
            sp500_price = fetcher.get_sp500_open_price(date)
        else:
            sp500_price = fetcher.get_sp500_close_price(date)

        dollar_amount = num_sp500_shares_buying * sp500_price
        self.sp500_shares = self.sp500_shares + (min(dollar_amount, self.available_cash) / sp500_price)
        self.available_cash = max(0, self.available_cash - dollar_amount)

    # Buys certain dollar amount of sp500 holdings
    def buy_sp500_dollars(self, dollar_amount, date, is_open_price):

        if is_open_price:
            sp500_price = fetcher.get_sp500_open_price(date)
        else:
            sp500_price = fetcher.get_sp500_close_price(date)

        self.sp500_shares = self.sp500_shares + (min(dollar_amount, self.available_cash) / sp500_price)
        self.available_cash = max(0, self.available_cash - dollar_amount)

    # Sells certain number of sp500 shares
    def sell_sp500_shares(self, num_sp500_shares_selling, date, is_open_price):

        if is_open_price:
            sp500_price = fetcher.get_sp500_open_price(date)
        else:
            sp500_price = fetcher.get_sp500_close_price(date)

        self.available_cash = self.available_cash + (min(self.sp500_shares, num_sp500_shares_selling) * sp500_price)
        self.sp500_shares = max(0, self.sp500_shares - num_sp500_shares_selling)

    # Sells certain dollar amount of sp500 holdings
    def sell_sp500_dollars(self, dollar_amount, date, is_open_price):

        if is_open_price:
            sp500_price = fetcher.get_sp500_open_price(date)
        else:
            sp500_price = fetcher.get_sp500_close_price(date)

        num_sp500_shares_selling = dollar_amount / sp500_price
        self.available_cash = self.available_cash + (min(self.sp500_shares, num_sp500_shares_selling) * sp500_price)
        self.sp500_shares = max(0, self.sp500_shares - num_sp500_shares_selling)

    @abstractmethod
    # Abstract method to be implemented: returns total portfolio value after applying this strategy over full period
    def apply_strategy(self, action_stock: pd.DataFrame, indicator_stock: pd.DataFrame) -> float:
        pass
