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
        syms = [symbol]
        dates = pd.date_range(sd - datetime.timedelta(days=50), ed)  # account for lookback for indicators #todo const
        prices = ut.get_data(syms, dates).drop(['SPY'], axis=1)

        if self.verbose:
            print(prices)

        # get indicator dataframes
        indicator_tuple = self.getindicators(prices, symbol, sd)
        sma_df = indicator_tuple[0]
        bbp_df = indicator_tuple[1]
        macd_df = indicator_tuple[2]

        # adjust price dataframes and capture daily returns
        prices = prices[sd:]
        daily_ret, port_val = self.stats(prices, symbol, sv)

        # combine data into overall dataframe
        combined_df = prices.copy()
        combined_df["SMA"] = sma_df[symbol]
        combined_df["BB %"] = bbp_df["Bollinger Band %"]
        combined_df["MACD"] = macd_df["MACD"]
        combined_df["Portfolio"] = port_val
        combined_df["Daily Return"] = daily_ret

        # discretize into bins
        sma_bins = self.discretize(combined_df["SMA"])
        bb_bins = self.discretize(combined_df["BB %"])
        macd_bins = self.discretize(combined_df["MACD"])
        self.state_train = (sma_bins * 100) + (bb_bins * 10) + macd_bins

        # build model
        iterations = 0

        trades = prices.copy()
        trades['Shares'] = 0
        trades.drop([symbol], axis=1, inplace=True)

        while iterations < 400:
            trades_copy = trades.copy()
            net_holdings = 0

            self.qLearner.querysetstate(self.state_train.iloc[0])

            for date, row in combined_df.iterrows():
                r = combined_df.at[date, "Daily Return"] - (net_holdings * self.impact)  # account for impact
                a = self.qLearner.query(int(self.state_train[date]), r)

                if a == 0:  # long
                    if net_holdings == 0:  # long
                        trades_copy.at[date, 'Shares'] = 1000
                        net_holdings += 1000

                    elif net_holdings == -1000:  # leave position
                        trades_copy.at[date, 'Shares'] = 1000
                        net_holdings += 1000

                elif a == 1:  # short
                    if net_holdings == 0:  # short
                        trades_copy.at[date, 'Shares'] = -1000
                        net_holdings -= 1000

                    elif net_holdings == 1000:  # leave position
                        trades_copy.at[date, 'Shares'] = -1000
                        net_holdings -= 1000

            if iterations > 10 and trades_copy.equals(trades):  # converged -> current and previous trades are equal
                break

            trades = trades_copy.copy()  # update trades
            iterations += 1

        self.trades = trades

        # # example usage of the old backward compatible util function
        # syms = [symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # if self.verbose:
        #     print(prices)

    #
    # # example use with new colname
    # volume_all = ut.get_data(
    #     syms, dates, colname="Volume"
    # )  # automatically adds SPY
    # volume = volume_all[syms]  # only portfolio symbols
    # volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
    # if self.verbose:
    #     print(volume)

    # this method should use the existing policy and test it against new data  		  	   		  		 		  		  		    	 		 		   		 		  
    def testPolicy(
            self,
            symbol="IBM",
            sd=datetime.datetime(2009, 1, 1),
            ed=datetime.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  		 		  		  		    	 		 		   		 		  
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

        # here we build a fake set of trades  		  	   		  		 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		  	   		  		 		  		  		    	 		 		   		 		  
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # trades.values[:, :] = 0  # set them all to nothing
        # trades.values[0, :] = 1000  # add a BUY at the start
        # trades.values[40, :] = -1000  # add a SELL
        # trades.values[41, :] = 1000  # add a BUY
        # trades.values[60, :] = -2000  # go short from long
        # trades.values[61, :] = 2000  # go long from short
        # trades.values[-1, :] = -1000  # exit on the last day
        # if self.verbose:
        #     print(type(trades))  # it better be a DataFrame!
        # if self.verbose:
        #     print(trades)
        # if self.verbose:
        #     print(prices_all)
        # return trades
        syms = [symbol]
        dates = pd.date_range(sd - datetime.timedelta(days=50), ed)  # account for MACD lookback period
        prices = ut.get_data(syms, dates).drop(['SPY'], axis=1)

        if self.verbose:
            print(prices)

        # get indicator dataframes
        indicator_tuple = self.getindicators(prices, symbol, sd)
        sma_df = indicator_tuple[0]
        bbp_df = indicator_tuple[1]
        macd_df = indicator_tuple[2]

        # adjust price dataframes and capture daily returns
        prices = prices[sd:]
        daily_ret, port_val = self.stats(prices, symbol, sv)

        # combine data into overall dataframe
        combined_df = prices.copy()
        combined_df["SMA"] = sma_df[symbol]
        combined_df["BB %"] = bbp_df["Bollinger Band %"]
        combined_df["MACD"] = macd_df["MACD"]
        combined_df["Portfolio"] = port_val
        combined_df["Daily Return"] = daily_ret

        # discretize into bins
        sma_bins = self.discretize(combined_df["SMA"])
        bb_bins = self.discretize(combined_df["BB %"])
        macd_bins = self.discretize(combined_df["MACD"])
        self.state_train = (sma_bins * 100) + (bb_bins * 10) + macd_bins

        # test model
        trades = prices.copy()
        trades['Shares'] = 0
        trades.drop([symbol], axis=1, inplace=True)

        net_holdings = 0

        for date, row in combined_df.iterrows():
            a = self.qLearner.querysetstate(int(self.state_train[date]))

            if a == 0:  # long
                if net_holdings == 0:  # long
                    trades.at[date, 'Shares'] = 1000
                    net_holdings += 1000

                elif net_holdings == -1000:  # leave position
                    trades.at[date, 'Shares'] = 1000
                    net_holdings += 1000

            elif a == 1:  # short
                if net_holdings == 0:  # short
                    trades.at[date, 'Shares'] = -1000
                    net_holdings -= 1000

                elif net_holdings == 1000:  # leave position
                    trades.at[date, 'Shares'] = -1000
                    net_holdings -= 1000

        return trades

        #

    def discretize(self, indicator):
        return pd.qcut(indicator, 10, labels=False, retbins=True)[0]

    def stats(self, prices, symbol, sv):
        normalize = prices.copy()
        normalize[symbol] = prices[symbol] / float(prices[symbol][0])

        port_val = normalize.sum(axis=1)
        daily_ret = port_val.pct_change(1)

        daily_ret.iloc[0] = 0
        port_val *= sv

        return daily_ret, port_val

        #

    def getindicators(self, prices, symbol, sd):
        sma_tuple = simple_moving_average(prices=prices, lookback=14, symbol=symbol, generate_plot=False)
        bb_tuple = bollinger_band_percentage(prices=prices, lookback=14, symbol=symbol,
                                             generate_plot=False)

        sma_df = sma_tuple[0][sd:]
        bbp_df = bb_tuple[1][sd:]
        macd_df = moving_avg_convergence_divergence(prices=prices, symbol=symbol, generate_plot=False)[sd:]

        return sma_df, bbp_df, macd_df


if __name__ == "__main__":
    print("One does not simply think up a strategy")
