from SimpleBacktester.Strategy import Strategy
import pandas as pd
from collections import deque
from SimpleBacktester import Formatter as fm, StockPriceFetcher as fetcher


# Overrides the strategy parent class with a concrete strategy implementation
class Strategy4(Strategy):

    # Returns the total portfolio value after applying this strategy to the action stock over full period
    #
    #   *** THIS STRATEGY MAKES USE OF A SECOND INDICATOR STOCK TO TAKE ACTIONS ON THE FIRST STOCK ***
    # Strategy consists of investing 20% of available capital whenever indicator 2-4 % up in 5 days
    #                      investing 40% of available capital whenever indicator 4-6 % up in 5 days
    #                      investing 60% of available capital whenever indicator 6+ % up in 5 days
    #
    #                      selling 20% of available shares whenever indicator 2-4 % down in 5 days
    #                      selling 40% of available shares whenever indicator 4-6 % down in 5 days
    #                      selling 60% of available shares whenever indicator 6+ % down in 5 days
    #
    #                      All available cash sits in SP500 index
    #
    #  ** Actions only considered at closing points so that trading is not simulated on future knowledge of highs/lows
    def apply_strategy(self, action_stock: pd.DataFrame, indicator_stock: pd.DataFrame) -> float:

        if indicator_stock is None:
            print("")
            print("Indicator stock must be defined for this strategy.")
            print("")
            return self.available_cash

        last_5_closes = deque(maxlen=5)

        super().buy_sp500_dollars(self.available_cash, fm.format_date(action_stock.index[0]), True)

        for index, indic_row in indicator_stock.iterrows():

            action_row = action_stock.loc[index]

            last_5_closes.append(indic_row['Close'].iloc[0])
            five_days_ago_close = last_5_closes[0]

            if indic_row['Close'].iloc[0] < (0.94 * five_days_ago_close):
                super().sell_stock_shares(self.stock_shares * 0.6, action_row['Close'].iloc[0])
                super().buy_sp500_dollars(self.available_cash, fm.format_date(index), False)
            elif indic_row['Close'].iloc[0] < (0.96 * five_days_ago_close):
                super().sell_stock_shares(self.stock_shares * 0.4, action_row['Close'].iloc[0])
                super().buy_sp500_dollars(self.available_cash, fm.format_date(index), False)
            elif indic_row['Close'].iloc[0] < (0.98 * five_days_ago_close):
                super().sell_stock_shares(self.stock_shares * 0.2, action_row['Close'].iloc[0])
                super().buy_sp500_dollars(self.available_cash, fm.format_date(index), False)

            if indic_row['Close'].iloc[0] > (1.06 * five_days_ago_close):
                super().sell_sp500_shares(self.sp500_shares * 0.6, fm.format_date(index), False)
                super().buy_stock_dollars(self.available_cash, action_row['Close'].iloc[0])
            elif indic_row['Close'].iloc[0] > (1.04 * five_days_ago_close):
                super().sell_sp500_shares(self.sp500_shares * 0.4, fm.format_date(index), False)
                super().buy_stock_dollars(self.available_cash, action_row['Close'].iloc[0])
            elif indic_row['Close'].iloc[0] > (1.02 * five_days_ago_close):
                super().sell_sp500_shares(self.sp500_shares * 0.2, fm.format_date(index), False)
                super().buy_stock_dollars(self.available_cash, action_row['Close'].iloc[0])

        return (self.available_cash +
                (self.stock_shares * action_stock.iloc[-1]['Close'].iloc[0]) +
                (self.sp500_shares * fetcher.get_sp500_close_price(fm.format_date(action_stock.index[-1]))))