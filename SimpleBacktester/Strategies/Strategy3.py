from SimpleBacktester.Strategy import Strategy
import pandas as pd
from collections import deque
from SimpleBacktester import Formatter as fm, StockPriceFetcher as fetcher


# Overrides the strategy parent class with a concrete strategy implementation
class Strategy3(Strategy):

    # Returns the total portfolio value after applying this strategy to the action stock over full period
    #
    # Strategy consists of investing 10% of the remaining available capital whenever 3-5 % below 30 day high
    #                      investing 30% of the remaining available capital whenever 5-10 % below 30 day high
    #                      investing 70% of the remaining available capital whenever 10+ % below 30 day high
    #
    #                      sell 10% of shares invested whenever 3-5 % above 30 day low
    #                      sell 30% of shares invested whenever 5-10 % above 30 day low
    #                      sell 70% of shares invested whenever 10+ % above 30 day low
    #
    #                      All available cash grows in SP500 index
    #
    #  ** Actions only considered at closing points so that trading is not simulated on future knowledge of highs/lows
    def apply_strategy(self, action_stock: pd.DataFrame, indicator_stock: pd.DataFrame) -> float:

        last_30_highs = deque(maxlen=30)
        last_30_lows = deque(maxlen=30)

        super().buy_sp500_dollars(self.available_cash, fm.format_date(action_stock.index[0]), True)

        for index, row in action_stock.iterrows():

            last_30_highs.append(row['High'].iloc[0])
            thirty_day_high = max(last_30_highs)

            last_30_lows.append(row['Low'].iloc[0])
            thirty_day_low = min(last_30_lows)

            if row['Close'].iloc[0] < (0.9 * thirty_day_high):
                super().sell_sp500_shares(self.sp500_shares * 0.7, fm.format_date(index), False)
                super().buy_stock_dollars(self.available_cash, row['Close'].iloc[0])
            elif row['Close'].iloc[0] < (0.95 * thirty_day_high):
                super().sell_sp500_shares(self.sp500_shares * 0.3, fm.format_date(index), False)
                super().buy_stock_dollars(self.available_cash, row['Close'].iloc[0])
            elif row['Close'].iloc[0] < (0.97 * thirty_day_high):
                super().sell_sp500_shares(self.sp500_shares * 0.1, fm.format_date(index), False)
                super().buy_stock_dollars(self.available_cash, row['Close'].iloc[0])

            if row['Close'].iloc[0] > (1.1 * thirty_day_low):
                super().sell_stock_shares(self.stock_shares * 0.7, row['Close'].iloc[0])
                super().buy_sp500_dollars(self.available_cash, fm.format_date(index), False)
            elif row['Close'].iloc[0] > (1.05 * thirty_day_low):
                super().sell_stock_shares(self.stock_shares * 0.3, row['Close'].iloc[0])
                super().buy_sp500_dollars(self.available_cash, fm.format_date(index), False)
            elif row['Close'].iloc[0] > (1.03 * thirty_day_low):
                super().sell_stock_shares(self.stock_shares * 0.1, row['Close'].iloc[0])
                super().buy_sp500_dollars(self.available_cash, fm.format_date(index), False)

        return (self.available_cash +
                (self.stock_shares * action_stock.iloc[-1]['Close'].iloc[0]) +
                (self.sp500_shares * fetcher.get_sp500_close_price(fm.format_date(action_stock.index[-1]))))