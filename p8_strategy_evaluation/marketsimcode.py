"""
        Market simulator
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


#
def compute_portvals(orders, start_val=1000000, commission=9.95, impact=0.005):
    start_date = orders.index[0]
    end_date = orders.index[-1]
    symbols = list(orders['Symbol'].unique())

    prices_df = get_data(symbols, pd.date_range(start_date, end_date)).drop(['SPY'], axis=1)
    prices_df.fillna(method='ffill', inplace=True)
    prices_df.fillna(method='bfill', inplace=True)
    prices_df['Cash'] = 1.0  # add cash column

    trades_df = prices_df.copy() * 0.0
    holdings_df = prices_df.copy() * 0.0

    holdings_df.at[start_date, 'Cash'] = start_val

    for index, row in orders.iterrows():
        date = index
        stock = row['Symbol']
        num_shares = row['Shares']

        stock_price = prices_df.at[date, stock]
        total_price = stock_price * num_shares

        if row['Order'] == 'BUY' and num_shares > 0:
            trades_df.at[date, stock] += num_shares
            transaction_cost = total_price + commission + (total_price * impact)
            trades_df.at[date, 'Cash'] -= transaction_cost

        elif row['Order'] == 'SELL' and num_shares > 0:  # selling / shorting a stock
            trades_df.at[date, stock] -= num_shares
            transaction_payout = total_price - commission - (total_price * impact)
            trades_df.at[date, 'Cash'] += transaction_payout

    holdings_df.loc[start_date] += trades_df.loc[start_date]  # start_date indicates row of index 0

    for i in range(1, trades_df.shape[0]):
        holdings_df.iloc[i] = trades_df.iloc[i] + holdings_df.iloc[i - 1]

    value_df = prices_df * holdings_df
    portval_df = value_df.sum(axis=1)  # actually a pandas series

    return portval_df


def print_portfolio_comparison_stats(portfolio1, portfolio1Name, portfolio2, portfolio2Name, graphTitle,
                                     figureName, title, sd, ed, long_trades_dates=[], short_trades_dates=[]):
    combined_portfolio_value = pd.concat([portfolio1, portfolio2], axis=1)
    combined_portfolio_values_graph = combined_portfolio_value.plot(title=graphTitle,
                                                                    fontsize=12,
                                                                    grid=True, color=['blue', 'black'])
    combined_portfolio_values_graph.set_xlabel("Timeline")
    combined_portfolio_values_graph.set_ylabel("Normalized Portfolio Values ($)")
    plt.vlines(long_trades_dates, 1.0, 1.2, color='g')
    plt.vlines(short_trades_dates, 1.0, 1.2, color='r')
    plt.savefig(figureName)
    manual_cumulative_return = (portfolio2.iloc[-1].at[portfolio2Name] /
                                portfolio2.iloc[0].at[
                                    portfolio2Name]) - 1
    manual_average_daily_return = portfolio2.pct_change(1).mean()[portfolio2Name]
    manual_standard_deviation = portfolio2.pct_change(1).std()[portfolio2Name]
    manual_sharpe_ratio = math.sqrt(252.0) * (manual_average_daily_return / manual_standard_deviation)
    benchmark_cumulative_return = (portfolio1.iloc[-1].at[portfolio1Name] /
                                   portfolio1.iloc[0].at[
                                       portfolio1Name]) - 1
    benchmark_average_daily_return = portfolio1.pct_change(1).mean()[portfolio1Name]
    benchmark_standard_deviation = portfolio1.pct_change(1).std()[portfolio1Name]
    benchmark_sharpe_ratio = math.sqrt(252.0) * (benchmark_average_daily_return / benchmark_standard_deviation)
    print("======================================================================")
    print(title)
    print("Date Range: {} to {}".format(sd, ed))
    print("Cumulative Return of {}: {}".format(portfolio1Name, benchmark_cumulative_return))
    print("Cumulative Return of {}: {}".format(portfolio2Name, manual_cumulative_return))
    print("Standard Deviation of {}: {}".format(portfolio1Name, benchmark_standard_deviation))
    print("Standard Deviation of {}: {}".format(portfolio2Name, manual_standard_deviation))
    print("Average Daily Return of {}: {}".format(portfolio1Name, benchmark_average_daily_return))
    print("Average Daily Return of {}: {}".format(portfolio2Name, manual_average_daily_return))
    print("Sharpe Ratio of {}: {}".format(portfolio1Name, benchmark_sharpe_ratio))
    print("Sharpe Ratio of {}: {}".format(portfolio2Name, manual_sharpe_ratio))
