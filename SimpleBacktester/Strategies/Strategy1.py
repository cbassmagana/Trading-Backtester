from SimpleBacktester.Strategy import Strategy
import pandas as pd

# Overrides the strategy parent class with a concrete strategy implementation
class Strategy1(Strategy):

    # Returns the total portfolio value after applying this strategy to action stock over full period
    #
    # Simple strategy of buying immediately and holding
    def apply_strategy(self, action_stock: pd.DataFrame, indicator_stock: pd.DataFrame) -> float:

        last_close_price = action_stock.iloc[-1]['Close'].iloc[0]
        first_open_price = action_stock.iloc[0]['Open'].iloc[0]

        return self.available_cash * (last_close_price / first_open_price)
