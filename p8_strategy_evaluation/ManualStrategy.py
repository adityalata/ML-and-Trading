import datetime as dt

import pandas as pd

from indicators import simple_moving_average, moving_avg_convergence_divergence, bollinger_band_percentage
from marketsimcode import compute_portvals, print_portfolio_comparison_stats
from util import get_data

LOOKBACK_MANUAL = 14


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


#
class ManualStrategy(object):

    #
    def __init__(self):
        self.long_trades_dates = []
        self.short_trades_dates = []

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    #
    def test_policy(self, symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(['SPY'], axis=1)
        manual_trades = prices.copy()
        manual_trades[symbol] = symbol
        manual_trades['Order'] = 'BUY'
        manual_trades['Shares'] = 0
        manual_trades.rename(columns={symbol: 'Symbol'}, inplace=True)
        total_holdings = 0
        long_manual_date_list = []
        short_manual_date_list = []

        _, simple_moving_average_df = simple_moving_average(prices=prices, lookback=LOOKBACK_MANUAL, symbol=symbol,
                                                            generate_plot=False)
        bollinger_band_df, bollinger_band_percentage_df = bollinger_band_percentage(prices=prices,
                                                                                    lookback=LOOKBACK_MANUAL,
                                                                                    symbol=symbol,
                                                                                    generate_plot=False)
        moving_avg_convergence_divergence_df = moving_avg_convergence_divergence(prices=prices, symbol=symbol,
                                                                                 generate_plot=False)

        for index in range(LOOKBACK_MANUAL - 1, simple_moving_average_df.shape[0]):
            current_date = manual_trades.index[index]
            simple_moving_average_ratio = simple_moving_average_df.at[current_date, 'Price/SMA']
            bollinger_band_percent = bollinger_band_percentage_df.at[current_date, 'Bollinger Band %']
            moving_avg_convergence_divergence_val = moving_avg_convergence_divergence_df.at[current_date, 'MACD']
            moving_avg_convergence_divergence_signal = moving_avg_convergence_divergence_df.at[
                current_date, 'Signal Line']

            if pd.isnull(
                    moving_avg_convergence_divergence_val):  # moving_avg_convergence_divergence lookback not available

                if simple_moving_average_ratio > 1.05 and bollinger_band_percent > 1:  # over priced position
                    if total_holdings == 0:  # short 1000
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        total_holdings -= 1000
                        short_manual_date_list.append(current_date)
                    elif total_holdings == 1000:  # sell and short
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        total_holdings -= 1000

                elif simple_moving_average_ratio < 0.95 and bollinger_band_percent < 0:  # under priced position
                    if total_holdings == 0:  # buy 1000
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        total_holdings += 1000
                        long_manual_date_list.append(current_date)
                    elif total_holdings == -1000:  # buy 2000
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        total_holdings += 1000

            else:  # including moving_avg_convergence_divergence

                if simple_moving_average_ratio > 1.05 and bollinger_band_percent > 1 and moving_avg_convergence_divergence_val > moving_avg_convergence_divergence_signal:  # over priced position
                    if total_holdings == 0:  # short 1000
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        total_holdings -= 1000
                        short_manual_date_list.append(current_date)
                    elif total_holdings == 1000:  # sell and short
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        total_holdings -= 1000

                elif simple_moving_average_ratio < 0.95 and bollinger_band_percent < 0 and moving_avg_convergence_divergence_val < moving_avg_convergence_divergence_signal:  # under priced position
                    if total_holdings == 0:  # buy 1000
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        total_holdings += 1000
                        long_manual_date_list.append(current_date)
                    elif total_holdings == -1000:  # buy 2000
                        manual_trades.loc[current_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        total_holdings += 1000

        self.short_trades_dates = short_manual_date_list
        self.long_trades_dates = long_manual_date_list

        return manual_trades

    #
    def benchmark_policy(self, symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """
            Benchmark trading strategy of investing 1000 shares in a stock and holding that position for the duration of the time interval
        """
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(['SPY'], axis=1)
        actual_start_data = prices.index[0]
        benchmark_trades = prices.copy()
        benchmark_trades[symbol] = symbol
        benchmark_trades['Order'] = 'BUY'
        benchmark_trades['Shares'] = 0
        benchmark_trades.rename(columns={symbol: 'Symbol'}, inplace=True)
        benchmark_trades.at[actual_start_data, 'Shares'] = 1000
        return benchmark_trades

    #
    def compare_manual_strategy_with_benchmark(self, sv=100000):
        # In Sample Comparison - Manual Strategy vs Benchmark
        sd = dt.datetime(2008, 1, 1)
        ed = dt.datetime(2009, 12, 31)

        self.compare_manual_strategy_with_benchmark_for_range(symbol='JPM', sd=sd, ed=ed, start_val=sv,
                                                              title='In Sample Comparison - Manual Strategy vs Benchmark',
                                                              graphTitle="Benchmark & Manual Strategy Portfolio Value (In-Sample) - alata6",
                                                              figureName="ManVsBmInSample.png")
        # Out of Sample Comparison - Manual Strategy vs Benchmark
        sd = dt.datetime(2010, 1, 1)
        ed = dt.datetime(2011, 12, 31)
        self.compare_manual_strategy_with_benchmark_for_range(symbol='JPM', sd=sd, ed=ed, start_val=sv,
                                                              title='Out of Sample Comparison - Manual Strategy vs Benchmark',
                                                              graphTitle="Benchmark & Manual Strategy Portfolio Value (Out-of-Sample) - alata6",
                                                              figureName="ManVsBmOutOfSample.png")

    def compare_manual_strategy_with_benchmark_for_range(self, symbol, sd, ed, start_val, title, graphTitle,
                                                         figureName):
        benchmark_trades = self.benchmark_policy(symbol, sd=sd, ed=ed)
        benchmark_portfolio_value = compute_portvals(benchmark_trades, start_val=start_val)
        benchmark_portfolio_value = benchmark_portfolio_value.to_frame(name='Benchmark PortVal')
        benchmark_portfolio_value /= benchmark_portfolio_value.iloc[0]
        manual_trades = self.test_policy(symbol, sd=sd, ed=ed)
        manual_portfolio_value = compute_portvals(manual_trades, start_val=start_val)
        manual_portfolio_value = manual_portfolio_value.to_frame(name='Manual Strategy PortVal')
        manual_portfolio_value /= manual_portfolio_value.iloc[0]
        print_portfolio_comparison_stats(portfolio1=benchmark_portfolio_value, portfolio1Name="Benchmark PortVal",
                                         portfolio2=manual_portfolio_value,
                                         portfolio2Name='Manual Strategy PortVal'
                                         , graphTitle=graphTitle, figureName=figureName, title=title, sd=sd, ed=ed,
                                         long_trades_dates=self.long_trades_dates,
                                         short_trades_dates=self.short_trades_dates)
