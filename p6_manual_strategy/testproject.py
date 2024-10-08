"""
    Evaluates the Manual Strategy and Theoretically Optimal Strategy against our benchmark strategy
"""

import datetime as dt
import math

import matplotlib.pyplot as plt
import pandas as pd

from TheoreticallyOptimalStrategy import TheoreticallyOptimalStrategy
from indicators import Indicators
from marketsimcode import compute_portvals
from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


class ManualStrategy(object):
    """
        A manual trading strategy using selected technical indicators
    """

    #
    def __init__(self):
        self.long = []
        self.short = []
        self.ind = Indicators()


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    #
    def test_policy(self, symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """
            Provides a trading strategy based on the values of certain technical indicators
        """

        lookback = 14

        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(['SPY'], axis=1)

        _, sma_df = self.ind.simple_moving_average(prices=prices, lookback=lookback, make_plot=False)
        bb_df, bbp_df = self.ind.bollinger_band_percentage(prices=prices, lookback=lookback, make_plot=False)
        macd_df = self.ind.macd(prices, False)

        manual_orders = prices.copy()
        manual_orders[symbol] = symbol
        manual_orders['Order'] = 'BUY'
        manual_orders['Shares'] = 0

        manual_orders.rename(columns={symbol: 'Symbol'}, inplace=True)

        net_holdings = 0
        long_dates = []  # dates where we long
        short_dates = []  # dates where we short

        for index in range(lookback - 1, sma_df.shape[0]):
            curr_date = manual_orders.index[index]

            sma_ratio = sma_df.at[curr_date, 'Price/SMA']
            bb_percent = bbp_df.at[curr_date, 'Bollinger Band %']
            macd_val = macd_df.at[curr_date, 'MACD']
            macd_signal = macd_df.at[curr_date, 'Signal Line']

            if pd.isnull(macd_val):  # still in lookback period for macd -> NaN values

                if sma_ratio > 1.05 and bb_percent > 1:  # overbought -> sell/short

                    if net_holdings == 0:  # short
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        net_holdings -= 1000
                        short_dates.append(curr_date)

                    elif net_holdings == 1000:  # leave position
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        net_holdings -= 1000

                elif sma_ratio < 0.95 and bb_percent < 0:  # oversold -> buy

                    if net_holdings == 0:  # long
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        net_holdings += 1000
                        long_dates.append(curr_date)

                    elif net_holdings == -1000:  # leave position
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        net_holdings += 1000

            else:

                if sma_ratio > 1.05 and bb_percent > 1 and macd_val > macd_signal:  # overbought -> sell/short

                    if net_holdings == 0:  # short
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        net_holdings -= 1000
                        short_dates.append(curr_date)

                    elif net_holdings == 1000:  # leave position
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000})
                        net_holdings -= 1000

                elif sma_ratio < 0.95 and bb_percent < 0 and macd_val < macd_signal:  # oversold -> buy

                    if net_holdings == 0:  # long
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        net_holdings += 1000
                        long_dates.append(curr_date)

                    elif net_holdings == -1000:  # leave position
                        manual_orders.loc[curr_date] = pd.Series({'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000})
                        net_holdings += 1000

        self.short = short_dates
        self.long = long_dates

        return manual_orders

    #
    def base_policy(self, symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """
            Implements a basic trading strategy of investing 1000 shares in a stock and holding
            that position for the duration of the time interval
        """

        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(['SPY'], axis=1)
        actual_start = prices.index[0]

        benchmark_orders = prices.copy()
        benchmark_orders[symbol] = symbol
        benchmark_orders['Order'] = 'BUY'
        benchmark_orders['Shares'] = 0

        benchmark_orders.rename(columns={symbol: 'Symbol'}, inplace=True)
        benchmark_orders.at[actual_start, 'Shares'] = 1000

        return benchmark_orders

    #
    def evaluate(self):
        """
            Compares the Manual Strategy approach to the benchmark approach
        """

        # In-Sample
        sd = dt.datetime(2008, 1, 1)
        ed = dt.datetime(2009, 12, 31)

        benchmark_orders = self.base_policy(symbol='JPM', sd=sd, ed=ed)
        benchmark_portval = compute_portvals(benchmark_orders, start_val=100000)
        benchmark_portval = benchmark_portval.to_frame(name='Benchmark PortVal')
        benchmark_portval /= benchmark_portval.iloc[0]

        manual_orders = self.test_policy(symbol='JPM', sd=sd, ed=ed)
        manual_portval = compute_portvals(manual_orders, start_val=100000)
        manual_portval = manual_portval.to_frame(name='Manual Strategy PortVal')
        manual_portval /= manual_portval.iloc[0]

        portval_df = pd.concat([benchmark_portval, manual_portval], axis=1)

        portval_graph = portval_df.plot(title="Benchmark & Manual Strategy Portfolio Value (In-Sample) - alata6", fontsize=12,
                                        grid=True, color=['blue', 'black'])
        portval_graph.set_xlabel("Date")
        portval_graph.set_ylabel("Normalized Portfolio Value ($)")

        plt.vlines(self.long, 1.0, 1.2, color='g')
        plt.vlines(self.short, 1.0, 1.2, color='r')
        plt.savefig("Figure_5.png")

        benchmark_cr = (benchmark_portval.iloc[-1].at['Benchmark PortVal'] / benchmark_portval.iloc[0].at[
            'Benchmark PortVal']) - 1
        benchmark_adr = benchmark_portval.pct_change(1).mean()['Benchmark PortVal']
        benchmark_sddr = benchmark_portval.pct_change(1).std()['Benchmark PortVal']
        benchmark_sr = math.sqrt(252.0) * (benchmark_adr / benchmark_sddr)

        manual_cr = (manual_portval.iloc[-1].at['Manual Strategy PortVal'] / manual_portval.iloc[0].at[
            'Manual Strategy PortVal']) - 1
        manual_adr = manual_portval.pct_change(1).mean()['Manual Strategy PortVal']
        manual_sddr = manual_portval.pct_change(1).std()['Manual Strategy PortVal']
        manual_sr = math.sqrt(252.0) * (manual_adr / manual_sddr)

        print("In Sample")

        print("Date Range: {} to {}".format(sd, ed))

        print("Cumulative Return of Benchmark: {}".format(benchmark_cr))
        print("Cumulative Return of Manual: {}".format(manual_cr))

        print("Standard Deviation of Benchmark: {}".format(benchmark_sddr))
        print("Standard Deviation of Manual: {}".format(manual_sddr))

        print("Average Daily Return of Benchmark: {}".format(benchmark_adr))
        print("Average Daily Return of Manual: {}".format(manual_adr))

        print("Sharpe Ratio of Benchmark: {}".format(benchmark_sr))
        print("Sharpe Ratio of Manual: {}".format(manual_sr))

        ##########

        # Out-of-Sample
        benchmark_orders = self.base_policy(symbol='JPM')  # roll forward to default date range
        benchmark_portval = compute_portvals(benchmark_orders, start_val=100000)
        benchmark_portval = benchmark_portval.to_frame(name='Benchmark PortVal')
        benchmark_portval /= benchmark_portval.iloc[0]

        manual_orders = self.test_policy(symbol='JPM')
        manual_portval = compute_portvals(manual_orders, start_val=100000)
        manual_portval = manual_portval.to_frame(name='Manual Strategy PortVal')
        manual_portval /= manual_portval.iloc[0]

        portval_df = pd.concat([benchmark_portval, manual_portval], axis=1)

        portval_graph = portval_df.plot(title="Benchmark & Manual Strategy Portfolio Value (Out-of-Sample) - alata6",
                                        fontsize=12, grid=True, color=['blue', 'black'])
        portval_graph.set_xlabel("Date")
        portval_graph.set_ylabel("Normalized Portfolio Value ($)")

        plt.vlines(self.long, 0.9, 1.1, color='g')
        plt.vlines(self.short, 0.9, 1.1, color='r')
        plt.savefig("Figure_6.png")

        benchmark_cr = (benchmark_portval.iloc[-1].at['Benchmark PortVal'] / benchmark_portval.iloc[0].at[
            'Benchmark PortVal']) - 1
        benchmark_adr = benchmark_portval.pct_change(1).mean()['Benchmark PortVal']
        benchmark_sddr = benchmark_portval.pct_change(1).std()['Benchmark PortVal']
        benchmark_sr = math.sqrt(252.0) * (benchmark_adr / benchmark_sddr)

        manual_cr = (manual_portval.iloc[-1].at['Manual Strategy PortVal'] / manual_portval.iloc[0].at[
            'Manual Strategy PortVal']) - 1
        manual_adr = manual_portval.pct_change(1).mean()['Manual Strategy PortVal']
        manual_sddr = manual_portval.pct_change(1).std()['Manual Strategy PortVal']
        manual_sr = math.sqrt(252.0) * (manual_adr / manual_sddr)

        print()
        print("Out-of-Sample")

        print("Date Range: {} to {}".format(sd, ed))

        print("Cumulative Return of Benchmark: {}".format(benchmark_cr))
        print("Cumulative Return of Manual: {}".format(manual_cr))

        print("Standard Deviation of Benchmark: {}".format(benchmark_sddr))
        print("Standard Deviation of Manual: {}".format(manual_sddr))

        print("Average Daily Return of Benchmark: {}".format(benchmark_adr))
        print("Average Daily Return of Manual: {}".format(manual_adr))

        print("Sharpe Ratio of Benchmark: {}".format(benchmark_sr))
        print("Sharpe Ratio of Manual: {}".format(manual_sr))

if __name__ == '__main__':
    ms = ManualStrategy()
    tos = TheoreticallyOptimalStrategy()
    ind = Indicators()
    ms.evaluate()
    tos.evaluate()
    ind.generate_charts()
