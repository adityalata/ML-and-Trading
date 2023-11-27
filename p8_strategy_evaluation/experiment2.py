import datetime as dt
import math

import matplotlib.pyplot as plt
import pandas as pd

from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


#
def evaluate_strategy_portval_for_impact(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                         impact=0.0):
    strategy_learner = StrategyLearner(impact=impact)
    strategy_learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    orders = strategy_learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    orders['Symbol'] = symbol
    orders['Order'] = 'BUY'
    strategy_order_count = 0

    for date, row in orders.iterrows():
        if orders.at[date, 'Shares'] == -1000:
            orders.at[date, 'Order'] = 'SELL'
        if row['Shares'] != 0:
            strategy_order_count += 1

    strategy = compute_portvals(orders, start_val=100000, commission=0.0, impact=impact)
    strategy_portfolio_value = strategy.to_frame(name='Strategy Learner PortVal')
    strategy_portfolio_value /= strategy_portfolio_value.iloc[0]
    return strategy_portfolio_value, strategy_order_count


#
def exp2():
    impact = 0.002
    exp2_sl_portvals = []
    print("=======================================================")
    print("Exp2 Impact comparison - In Sample")
    print("Date Range: {} to {}".format(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))

    for i in range(10):
        strategy_portfolio_values, strategy_orders = evaluate_strategy_portval_for_impact(impact=impact)
        strategy_cumulative_return = (strategy_portfolio_values.iloc[-1].at['Strategy Learner PortVal'] /
                                      strategy_portfolio_values.iloc[0].at[
                                          'Strategy Learner PortVal']) - 1
        strategy_avg_daily_ret = strategy_portfolio_values.pct_change(1).mean()['Strategy Learner PortVal']
        strategy_std_dev = strategy_portfolio_values.pct_change(1).std()['Strategy Learner PortVal']
        strategy_sharpe_ratio = math.sqrt(252.0) * (strategy_avg_daily_ret / float(strategy_std_dev))
        print("=======================================================")
        print("Cumulative Return of Strategy (Impact: " + str(impact) + "): {}".format(strategy_cumulative_return))
        print("Standard Deviation of Strategy (Impact: " + str(impact) + "): {}".format(strategy_std_dev))
        print("Average Daily Return of Strategy (Impact: " + str(impact) + "): {}".format(strategy_avg_daily_ret))
        print("Sharpe Ratio of Strategy (Impact: " + str(impact) + "): {}".format(strategy_sharpe_ratio))
        print("Number of Trades for Strategy (Impact: " + str(impact) + "): {}".format(strategy_orders))
        strategy_portfolio_values.rename(columns={"Strategy Learner PortVal": "Impact: " + str(impact)}, inplace=True)
        exp2_sl_portvals.append(strategy_portfolio_values)
        impact *= 2  # doubling impact

    portfolio_value_net = pd.concat(exp2_sl_portvals, axis=1)
    portfolio_value_graph = portfolio_value_net.plot(title="Exp. 2: Strategy Learner Changing Impact", fontsize=12,
                                                     grid=True)
    portfolio_value_graph.set_xlabel("Date")
    portfolio_value_graph.set_ylabel("Normalized Portfolio Value ($)")
    plt.savefig("exp2 impact vs portval.png")
