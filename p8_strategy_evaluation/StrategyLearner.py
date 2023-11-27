""""""
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""

import datetime as datetime

import pandas as pd

import util as ut
from QLearner import QLearner
from indicators import simple_moving_average, bollinger_band_percentage, moving_avg_convergence_divergence

LOOKBACK_DAYS = 50


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


class StrategyLearner(object):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    """

    # constructor  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.qLearner = QLearner(num_states=1000, num_actions=3, rar=0.0)

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    # this method should create a QLearner, and train it for trading  		  	   		  		 		  		  		    	 		 		   		 		  
    def add_evidence(
            self,
            symbol="IBM",
            sd=datetime.datetime(2008, 1, 1),
            ed=datetime.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		  		 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        """

        # add your code to do learning here
        symbol_list = [symbol]
        dates = pd.date_range(sd - datetime.timedelta(days=LOOKBACK_DAYS), ed)
        prices = ut.get_data(symbol_list, dates).drop(['SPY'], axis=1)

        if self.verbose:
            print('prices with lookback', prices)

        # fetching indicators
        indicators = self.fetch_indicators_tuple(prices, symbol, sd)
        simple_moving_average_df = indicators[0]
        bollinger_band_percentage_df = indicators[1]
        moving_avg_convergence_divergence_df = indicators[2]

        # evaluating daily returns
        prices = prices[sd:]
        daily_return, portfolio_value = self.compute_dailyret_portval(prices, symbol, sv)

        # summarizing results
        summary_data = prices.copy()
        summary_data["SMA"] = simple_moving_average_df[symbol]
        summary_data["BB %"] = bollinger_band_percentage_df["Bollinger Band %"]
        summary_data["MACD"] = moving_avg_convergence_divergence_df["MACD"]
        summary_data["Portfolio"] = portfolio_value
        summary_data["Daily Return"] = daily_return

        # binning
        simple_moving_average_bins = self.quantile_discretize(summary_data["SMA"])
        bollinger_band_bins = self.quantile_discretize(summary_data["BB %"])
        moving_avg_convergence_divergence_bins = self.quantile_discretize(summary_data["MACD"])
        self.state_train = (simple_moving_average_bins * 100) + (
                bollinger_band_bins * 10) + moving_avg_convergence_divergence_bins

        # building model
        count = 0
        orders = prices.copy()
        orders['Shares'] = 0
        orders.drop([symbol], axis=1, inplace=True)

        while count < 400:
            updated_order = orders.copy()
            total_holdings = 0
            self.qLearner.querysetstate(self.state_train.iloc[0])

            for date, row in summary_data.iterrows():
                returns = summary_data.at[date, "Daily Return"] - (total_holdings * self.impact)
                action = self.qLearner.query(int(self.state_train[date]), returns)

                if action == 0:  # buy
                    if total_holdings == 0:  # long
                        updated_order.at[date, 'Shares'] = 1000
                        total_holdings += 1000
                    elif total_holdings == -1000:  # todo 2000?
                        updated_order.at[date, 'Shares'] = 1000
                        total_holdings += 1000

                elif action == 1:  # sell
                    if total_holdings == 0:  # short
                        updated_order.at[date, 'Shares'] = -1000
                        total_holdings -= 1000
                    elif total_holdings == 1000:  # todo 2000?
                        updated_order.at[date, 'Shares'] = -1000
                        total_holdings -= 1000

            if count > 10 and updated_order.equals(orders):  # checking for convergence
                break

            orders = updated_order.copy()  # updating orders
            count += 1

        self.trades = orders

    def testPolicy(
            self,
            symbol="IBM",
            sd=datetime.datetime(2009, 1, 1),
            ed=datetime.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside the training data
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        symbols_list = [symbol]
        dates = pd.date_range(sd - datetime.timedelta(days=LOOKBACK_DAYS),
                              ed)
        prices = ut.get_data(symbols_list, dates).drop(['SPY'], axis=1)

        if self.verbose:
            print('prices with lookback', prices)

        # fetching indicators
        indicators = self.fetch_indicators_tuple(prices, symbol, sd)
        simple_moving_average_df = indicators[0]
        bollinger_band_percentage_df = indicators[1]
        moving_avg_convergence_divergence_df = indicators[2]

        # evaluating daily returns
        prices = prices[sd:]
        daily_return, portfolio_value = self.compute_dailyret_portval(prices, symbol, sv)

        # summarizing results
        summary_data = prices.copy()
        summary_data["SMA"] = simple_moving_average_df[symbol]
        summary_data["BB %"] = bollinger_band_percentage_df["Bollinger Band %"]
        summary_data["MACD"] = moving_avg_convergence_divergence_df["MACD"]
        summary_data["Portfolio"] = portfolio_value
        summary_data["Daily Return"] = daily_return

        # binning
        simple_moving_average_bins = self.quantile_discretize(summary_data["SMA"])
        bollinger_band_bins = self.quantile_discretize(summary_data["BB %"])
        moving_avg_convergence_divergence_bins = self.quantile_discretize(summary_data["MACD"])
        self.state_train = (simple_moving_average_bins * 100) + (
                bollinger_band_bins * 10) + moving_avg_convergence_divergence_bins

        # test model
        orders = prices.copy()
        orders['Shares'] = 0
        orders.drop([symbol], axis=1, inplace=True)
        total_holdings = 0

        for date, row in summary_data.iterrows():
            action = self.qLearner.querysetstate(int(self.state_train[date]))

            if action == 0:  # buy
                if total_holdings == 0:  # long
                    orders.at[date, 'Shares'] = 1000
                    total_holdings += 1000
                elif total_holdings == -1000:  # todo test 2000
                    orders.at[date, 'Shares'] = 1000
                    total_holdings += 1000

            elif action == 1:  # sell
                if total_holdings == 0:  # short
                    orders.at[date, 'Shares'] = -1000
                    total_holdings -= 1000
                elif total_holdings == 1000:  # todo test 2000
                    orders.at[date, 'Shares'] = -1000
                    total_holdings -= 1000

        return orders

        #

    """
     Quantile-based discretization function. Discretize variable into equal-sized buckets based on rank or based on sample quantiles. 
     For example 1000 values for 10 quantiles would produce a Categorical object indicating quantile membership for each data point.
    """

    def quantile_discretize(self, indicator):
        return pd.qcut(indicator, 10, labels=False, retbins=True)[0]

    def compute_dailyret_portval(self, prices, symbol, sv):
        symbol_price = prices.copy()
        symbol_price[symbol] = prices[symbol] / float(prices[symbol][0])
        portfolio_value = symbol_price.sum(axis=1)
        daily_return = portfolio_value.pct_change(1)
        daily_return.iloc[0] = 0
        portfolio_value *= sv
        return daily_return, portfolio_value

    def fetch_indicators_tuple(self, prices, symbol, sd):
        simple_moving_average_tuple = simple_moving_average(prices=prices, lookback=14, symbol=symbol,
                                                            generate_plot=False)
        bollinger_band_tuple = bollinger_band_percentage(prices=prices, lookback=14, symbol=symbol,
                                                         generate_plot=False)

        simple_moving_average_df = simple_moving_average_tuple[0][sd:]
        bollinger_band_percentage_df = bollinger_band_tuple[1][sd:]
        moving_avg_convergence_divergence_df = moving_avg_convergence_divergence(prices=prices, symbol=symbol,
                                                                                 generate_plot=False)[sd:]

        return simple_moving_average_df, bollinger_band_percentage_df, moving_avg_convergence_divergence_df


if __name__ == "__main__":
    print("One does not simply think up a strategy")
