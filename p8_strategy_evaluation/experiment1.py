import datetime as dt

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals, print_portfolio_comparison_stats


#
def compare(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), impact=0.0, train=False,
            graphTitle="Exp. 1: Strategy Learner vs. & Manual Strategy (In-Sample) - alata6",
            figureName="exp1 In Sample Manual vs Strategy Learner.png",
            title='exp1 In Sample Manual vs Strategy Learner'):
    manual_strategy = ManualStrategy()
    manual_trades = manual_strategy.test_policy(symbol, sd=sd, ed=ed)
    manual_trades_count = 0

    for date, row in manual_trades.iterrows():
        if row['Shares'] != 0:
            manual_trades_count += 1

    manual = compute_portvals(manual_trades, start_val=100000, commission=0.0, impact=impact)
    manual_portfolio_values = manual.to_frame(name='Manual Strategy PortVal')
    manual_portfolio_values /= manual_portfolio_values.iloc[0]

    ##########################
    strategy_learner = StrategyLearner(impact=0.002)
    if train:
        strategy_learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    strategy_learner_trades = strategy_learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    strategy_learner_trades['Symbol'] = symbol
    strategy_learner_trades['Order'] = 'BUY'
    strategy_trades_count = 0

    for date, row in strategy_learner_trades.iterrows():
        if row['Shares'] == -1000:
            strategy_learner_trades.at[date, 'Order'] = 'SELL'
        if row['Shares'] != 0:
            strategy_trades_count += 1

    strategy = compute_portvals(strategy_learner_trades, start_val=100000, commission=0.0, impact=impact)
    strategy_portfolio_values = strategy.to_frame(name='Strategy Learner PortVal')
    strategy_portfolio_values /= strategy_portfolio_values.iloc[0]
    print_portfolio_comparison_stats(portfolio1=strategy_portfolio_values, portfolio1Name='Strategy Learner PortVal',
                                     portfolio2=manual_portfolio_values,
                                     portfolio2Name='Manual Strategy PortVal'
                                     , graphTitle=graphTitle,
                                     figureName=figureName,
                                     title=title, sd=sd,
                                     ed=ed)


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


#
def exp1():
    compare(train=True)
    # Out of Sample Comparison - Manual Strategy vs Benchmark
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    compare(sd=sd, ed=ed, graphTitle="Exp. 1: Strategy Learner vs. & Manual Strategy (Out-Of-Sample) - alata6",
            figureName="exp1 Out of Sample Manual vs Strategy Learner.png",
            title='exp1 Out of Sample Manual vs Strategy Learner')
