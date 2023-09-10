""""""
import math

"""MC1-P2: Optimize a portfolio.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Aditya Lata (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: alata6 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903952381 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  

import pandas as pd
import scipy.optimize as spo
from util import get_data


# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def get_optimal_allocations(portfolio_prices, portfolio_size):
    """

    :param portfolio_prices:
    :param portfolio_size:
    :return:
    """
    initial_guess = np.asarray([1.0 / portfolio_size] * portfolio_size)
    # reference : https://saturncloud.io/blog/scipyoptimizeminimize-slsqp-a-guide-to-handling-bounds-and-constraints/#calling-slsqp-with-bounds-and-constraints
    optimization_result = spo.minimize(
        fun=get_sharpe_ratio,
        x0=initial_guess,
        args=portfolio_prices,
        method='SLSQP',
        bounds=[(0, 1)] * portfolio_size,
        constraints={'type': 'eq', 'fun': lambda allocations: 1.0 - np.sum(allocations)}
    )
    return optimization_result.x


def get_sharpe_ratio(allocations_list, portfolio_prices):
    # we want to use a minimize optimizer, for a portfolio return maximization objective, thus * -1
    return generate_portfolio_stats(portfolio_prices=portfolio_prices, allocations_list=allocations_list)[-2] * -1


def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=False)


def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  		 		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 		  		  		    	 		 		   		 		  
    statistics.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		  		 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  		 		  		  		    	 		 		   		 		  
    portfolio_prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    fill_missing_values(portfolio_prices)
    fill_missing_values(prices_SPY)
  		  	   		  		 		  		  		    	 		 		   		 		  
    # find the allocations for the optimal portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		  		 		  		  		    	 		 		   		 		  
    allocs = get_optimal_allocations(portfolio_prices=portfolio_prices,portfolio_size=len(syms))
    cr, adr, sddr, sr, port_val = generate_portfolio_stats(portfolio_prices=portfolio_prices, allocations_list=allocs)
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  		 		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		  		 		  		  		    	 		 		   		 		  
        # add code to plot here
        normed_spy = prices_SPY/prices_SPY.iloc[0]
        df_temp = pd.concat(  		  	   		  		 		  		  		    	 		 		   		 		  
            [port_val, normed_spy], keys=["Portfolio", "SPY"], axis=1
        )  		  	   		  		 		  		  		    	 		 		   		 		  
        save_plot_data(df_temp)
  		  	   		  		 		  		  		    	 		 		   		 		  
    return allocs, cr, adr, sddr, sr  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  

def generate_portfolio_stats(portfolio_prices, allocations_list, initial_investment=1, risk_free_return=0, sample_frequency=252):
    """
    generate_portfolio_stats
    :param initial_investment: default 1, if we just want stats, then since it's a constant, it would not affect
    :param portfolio_prices:
    :param allocations_list:
    :param risk_free_return: default 0 based on Dr. B's comments in video
    :param sample_frequency: default 252 for number of trading days in a year for SPY
    :return: A tuple containing the cumulative return, average daily returns,
        standard deviation of daily returns, Sharpe ratio, daily Portfolio Valuation
    """
    normed_portfolio_prices = portfolio_prices/portfolio_prices.iloc[0]
    alloced_normed_portfolio_prices = normed_portfolio_prices * allocations_list
    portfolio_position_values = alloced_normed_portfolio_prices * initial_investment
    portfolio_valuation = portfolio_position_values.sum(axis=1)

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
    daily_returns = (portfolio_valuation.pct_change()).iloc[1:]  # since daily ret cant be defined for 1st day
    cumulative_return = (portfolio_valuation[-1] / portfolio_valuation[0] - 1)
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    portfolio_return_diff_rfr = avg_daily_return-risk_free_return

    sharpe_ratio = math.sqrt(sample_frequency) * portfolio_return_diff_rfr / std_daily_return
    return cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio, portfolio_valuation


def save_plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", figure_number=1):
    import matplotlib.pyplot as plt

    """Plot stock prices with a custom title and meaningful axis labels."""
    plt.figure(figure_number)
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('images/Figure_{}.png'.format(figure_number))
    plt.close(figure_number)


def test_code():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2009, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2010, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")
    print(f"Allocations Sum:{np.sum(allocations)}")
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		  		 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
