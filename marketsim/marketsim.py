""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import os  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data


def read_orders(orders_file):
    """
    The files containing orders are CSV files with the following columns:
    Date (yyyy-mm-dd)
    Symbol (e.g. AAPL, GOOG)
    Order (BUY or SELL)
    Shares (no. of shares to trade)
    :param orders_file:  Path of the order file or the file object
    :return: dataframe of sorted orders
    :rtype: pandas.DataFrame
    # TODO: NOTE: orders_file may be a string, or it may be a file object. Your code should work correctly with either input
    # Note: The orders may not appear in sequential order in the file
    """
    if isinstance(orders_file, pd.DataFrame):
        return orders_file
    else:
        orders = pd.read_csv(orders_file, index_col=['Date'], dtype='|str, str, str,  i4', parse_dates=['Date'])
        orders.sort_values(by="Date", inplace=True)
        return orders


def get_orderdf_stats(orders_dataframe):
    """
    Identify key stats for given orders and return
    :param orders_dataframe: dataframe of orders
    :return: start_date, end_date, and list of unique symbols
    """
    start_date = orders_dataframe.index[0]
    end_date = orders_dataframe.index[-1]
    symbols = list(orders_dataframe['Symbol'].unique())  # ndarray to list
    return start_date, end_date, symbols


def get_adj_close_prices(symbols, start_date, end_date):
    symbols_adj_close = get_data(symbols=symbols, dates=pd.date_range(start_date, end_date), addSPY=False)
    symbols_adj_close.dropna(inplace=True)  # todo check
    return symbols_adj_close


def initialize_trades_holdings_dfs(symbols_adj_close, start_date, start_val):
    symbols_adj_close['CashBalance'] = 1.0  # add cash column
    trades_df = symbols_adj_close.copy() * 0.0
    holdings_df = symbols_adj_close.copy() * 0.0
    holdings_df.at[start_date, 'CashBalance'] = start_val
    return trades_df, holdings_df


def evaluate_order(symbols_adj_close, trades_df, date, order, commission, impact, debug=False):
    symbol, order_type, shares_count = order

    # Data cleansing
    if pd.isnull(shares_count):
        if debug:
            print("shares_count is nan")
        return
    if shares_count == 0 and order_type == "":
        if debug:
            print("empty order")
        return
    # Allow indicating buying and selling via shares_count. If shares is positive we buy and if it is negative we sell.
    if order_type == "":
        if shares_count > 0:
            if debug:
                print("assuming BUY order since shares_count > 0")
            order_type = "BUY"
        elif shares_count < 0:
            if debug:
                print("assuming SELL order since shares_count < 0")
            order_type = "SELL"
            shares_count = abs(shares_count)
    else:
        if shares_count < 0:
            if debug:
                print("found shares_count : ", shares_count, " < 0 , thus flipping order_type")
            shares_count = abs(shares_count)
            if order_type == "BUY":
                order_type = "SELL"
            elif order_type == "SELL":
                order_type = "BUY"

    stock_price = symbols_adj_close.at[date, symbol]
    total_price = stock_price * shares_count
    transaction_cost = commission + (total_price * impact)
    if order_type == 'BUY' and shares_count > 0:
        trades_df.at[date, symbol] += shares_count
        cash_balance_outflow = total_price + transaction_cost
        trades_df.at[date, 'CashBalance'] -= cash_balance_outflow
    elif order_type == 'SELL' and shares_count > 0:  # selling / shorting a stock
        trades_df.at[date, symbol] -= shares_count
        cash_balance_inflow = total_price - transaction_cost
        trades_df.at[date, 'CashBalance'] += cash_balance_inflow
    else:
        raise Exception("Unexpected order parameters")


def compute_portvals(
    orders_file="./orders/orders.csv",  		  	   		  		 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		  		 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		  		 		  		  		    	 		 		   		 		  
    impact=0.005,
    debug=False,
):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param debug: flag to decide if logs are to be printed, default False
    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  		 		  		
    
    Given :   		    
    In terms of execution prices, you should assume you get the adjusted close price for the day of the trade.	 	
    The Sharpe ratio uses the sample standard deviation.
    Note that negative shares and negative cash are possible. Negative shares mean that the portfolio is in a short position for that stock. Negative cash means that you’ve borrowed money from the broker. 	 		   		 		  
    """
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    orders_dataframe = read_orders(orders_file)
    start_date, end_date, symbols = get_orderdf_stats(orders_dataframe)
    symbols_adj_close = get_adj_close_prices(symbols, start_date, end_date)
    daily_trade_df, cumulative_trade_df = initialize_trades_holdings_dfs(symbols_adj_close, start_date, start_val)
    for date, order in orders_dataframe.iterrows():
        evaluate_order(symbols_adj_close, trades_df, date, order, commission, impact, debug=debug)


    portvals = symbols_adj_close[["IBM"]]  # remove SPY
    rv = pd.DataFrame(index=portvals.index, data=portvals.values)
    return rv
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		  		 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    of = "./orders/orders-02.csv"
    sv = 1000000  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Process orders  		  	   		  		 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv, debug=True)
    if isinstance(portvals, pd.DataFrame):  		  	   		  		 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		  		 		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2008, 6, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		  		 		  		  		    	 		 		   		 		  
        0.2,  		  	   		  		 		  		  		    	 		 		   		 		  
        0.01,  		  	   		  		 		  		  		    	 		 		   		 		  
        0.02,  		  	   		  		 		  		  		    	 		 		   		 		  
        1.5,  		  	   		  		 		  		  		    	 		 		   		 		  
    ]  		  	   		  		 		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		  		 		  		  		    	 		 		   		 		  
        0.2,  		  	   		  		 		  		  		    	 		 		   		 		  
        0.01,  		  	   		  		 		  		  		    	 		 		   		 		  
        0.02,  		  	   		  		 		  		  		    	 		 		   		 		  
        1.5,  		  	   		  		 		  		  		    	 		 		   		 		  
    ]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


if __name__ == "__main__":
    # will not be called to evaluate, this is only for testing using following command
    # PYTHONPATH=../:. python grade_marketsim.py
    print(author())
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
