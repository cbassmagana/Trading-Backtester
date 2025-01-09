For this application, I have designed and created a backtesting stock market simulation that allows users to create their own trading strategies and evaluate how they would have performed on historical data for different time windows and stocks. My motivation for creating this was partially to see how commonly preached strategies actually stack up against each other, and also partially to have a sandbox where I can search for potential edges in the market as my quantitative skills and knowledge of trading continue to improve.

Several simple strategies are included in this repository as examples of the structure and tools available to someone creating their own trading strategies. Of course, there is plenty of room for creativity to bring all kinds of techniques into the picture within the lines of the available parameters.

The system is entirely written in Python and makes use of the Yahoo Finance public API. After the user selects a time frame and stock ticker to trade (as well as one optional stock to use as an indicator within their strategy), the program fetches that stock's trading history from the API and asks the user to select a trading strategy. The system then runs a simulation to obtain the results of this strategy and produces an in-depth report of the outcomes, including performance in relation to inflation, the S&P 500 index, and the stock's price itself. Since S&P 500 and inflation statistics are retrieved so frequently, the system contains its own data files for this information to reduce the number of API calls made by the system.

There are several limitations of the system that I am still keen on tackling. The first is that the system only retrieves data from the API with the time granularity of "1 day." If the granularity is any shorter than this, the free Yahoo Finance API limits the request to the last 730 days, which is insufficient to test many trading strategies. Another limitation is that the system only supports trading strategies on 1 or 2 stocks at a time. As a result, you cannot test strategies like "buy the worst performer in the S&P 500 each year." Although I would like to add this feature in the future, my current idea for how to accomplish it seems very computationally inefficient, as it requires many sets of stock data to be retrieved via the API. Another similar limitation is that the dataframe retrieved only contains the columns: Date, High, Low, Open, Close, and Volume. Thus, if you want to trade on an indicator that cannot be derived from these, such as the 'Price/Earnings Ratio,' you are out of luck. Lastly, I plan to add a graphical feedback portion to this project that allows for enhanced visualization of when the strategy buys and sells, as well as grows and diminishes.
