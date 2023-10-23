import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, symbol_to_path


class Indicators(object):
    def __init__(self):
        pass

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    #
    def generate_charts(self):
        start_date = dt.datetime(2008, 1, 1)
        end_date = dt.datetime(2009, 12, 31)
        shorter_end_date = dt.datetime(2008, 9, 30)
        dates = pd.date_range(start_date, end_date)
        shorter_range = pd.date_range(start_date, shorter_end_date)
        lookback = 14
        symbols = ['JPM']
        prices = get_data(symbols, dates).drop(['SPY'], axis=1)
        all_data = self.get_all_data(symbols, shorter_range, addSPY=False)
        self.simple_moving_average(prices=prices, lookback=lookback, make_plot=True)
        self.bollinger_band_percentage(prices, lookback, True)
        self.macd(prices, True)
        self.stochastic_osc(all_data, make_plot=True)
        self.commodity_channel_index(all_data, True)

    """
    Simple moving averages (SMAs) are an average of prices over the specified timeframe
    """
    def simple_moving_average(self, prices, lookback, make_plot):
        sma_df = prices.rolling(window=lookback, min_periods=lookback).mean()
        sma_normalized = sma_df.copy()

        sma_normalized.iloc[lookback - 1:] /= prices.iloc[0]  # normalized
        sma_normalized.rename(columns={'JPM': 'SMA'}, inplace=True)
        sma_normalized['Prices'] = prices / prices.iloc[0]  # normalized
        sma_normalized['Price/SMA'] = sma_normalized['Prices'] / sma_normalized['SMA']

        if make_plot:
            sma_graph = sma_normalized.plot(title='Simple Moving Average (Normalized) for JPM - alata6', fontsize=12, grid=True)
            sma_graph.set_xlabel('Date')
            sma_graph.set_ylabel('Normalized $ Value')
            plt.savefig('Figure_1.png')

        return sma_df, sma_normalized

    """
    the relationship between price and standard deviation Bollinger Bands.
    """
    def bollinger_band_percentage(self, prices, lookback, make_plot):
        sma_df, _ = self.simple_moving_average(prices=prices, lookback=lookback, make_plot=False)

        rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
        top_band = sma_df + (2 * rolling_std)
        bottom_band = sma_df - (2 * rolling_std)

        # values greater than 1 indicate price is above the upper band
        # values less than 0 indicate price is below the bottom band
        bbp_df = (prices - bottom_band) / (top_band - bottom_band)
        bbp_df.rename(columns={'JPM': 'Bollinger Band %'}, inplace=True)

        bb_df = prices.copy()
        bb_df.rename(columns={'JPM': 'JPM Price'}, inplace=True)
        bb_df['SMA'] = sma_df
        bb_df['Upper Band'] = top_band
        bb_df['Lower Band'] = bottom_band

        if make_plot:
            figure, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            bb_df.plot(ax=ax1, fontsize=12, grid=True)  # bollinger band subplot
            bbp_df.plot(ax=ax2, fontsize=12, grid=True)  # bollinger band percentage subplot
            ax1.set_title('Bollinger Bands Against JPM Price Data - alata6')
            ax1.legend(prop={'size': '10'}, loc=4)
            ax1.set(ylabel='$ Value')
            ax2.set_title('Bollinger Band % for JPM - alata6')
            ax2.legend(prop={'size': '10'}, loc=4)
            ax2.set(ylabel='% Value')
            plt.xlabel('Date')
            plt.savefig('Figure_2.png')

        return bb_df, bbp_df

    """
    (Moving Average Convergence/Divergence Oscillator)
    A momentum oscillator based on the difference between two EMAs.
    """
    def macd(self, prices, make_plot):
        ema_26 = prices.ewm(span=26, min_periods=26, adjust=False).mean()
        ema_12 = prices.ewm(span=12, min_periods=12, adjust=False).mean()

        macd_df = ema_12 - ema_26
        macd_df.rename(columns={'JPM': 'MACD'}, inplace=True)
        ema_9 = macd_df.ewm(span=9, min_periods=1, adjust=False).mean()  # ema of macd graph (signal line)
        macd_df['Signal Line'] = ema_9

        if make_plot:
            macd_graph = macd_df.plot(title='Moving Average Convergence Divergence for JPM - alata6', fontsize=12, grid=True)
            macd_graph.set_xlabel('Date')
            macd_graph.set_ylabel('Normalized $ Value')
            plt.savefig('Figure_3.png')

        return macd_df


    """
    The stochastic oscillator is a momentum indicator used to signal trend reversals in the stock market. 
       It describes the current price relative to the high and low prices over a trailing number of previous trading periods.
       fast_period : previous days we want to use to generate our fast signal
       slow_period : previous days we want to use to generate our slow signal
    """
    def stochastic_osc(self, symbol_df, fast_period=14, slow_period=3, make_plot=False):
        # Adds a "n_high" column with max value of previous 14 periods
        symbol_df['n_high'] = symbol_df['High'].rolling(fast_period).max()
        # Adds an "n_low" column with min value of previous 14 periods
        symbol_df['n_low'] = symbol_df['Low'].rolling(fast_period).min()
        # Uses the min/max values to calculate the %k (as a percentage)
        symbol_df['%K'] = (symbol_df['Close'] - symbol_df['n_low']) * 100 / (symbol_df['n_high'] - symbol_df['n_low'])
        # Uses the %k to calculates a SMA over the past 3 values of %k
        symbol_df['%D'] = symbol_df['%K'].rolling(slow_period).mean()

        stoch_df = symbol_df.copy()
        stoch_df.drop(stoch_df.iloc[:, 0:-2], inplace=True, axis=1)
        stoch_df['%K norm'] = stoch_df['%K'] / stoch_df['%K'].iloc[fast_period-1]  # normalized
        stoch_df['%D norm'] = stoch_df['%D'] / stoch_df['%D'].iloc[fast_period+slow_period-2]  # normalized
        stoch_df.drop(stoch_df.iloc[:, 0:2], inplace=True, axis=1)
        adj_close_prices = symbol_df['Adj Close']
        stoch_df['Prices'] = adj_close_prices / adj_close_prices.iloc[0]  # normalized

        if make_plot:
            stoch_graph = stoch_df.plot(title='Stochastic Oscillator for JPM - alata6', fontsize=12,
                                    grid=True)
            stoch_graph.set_xlabel('Date')
            stoch_graph.set_ylabel('Normalized Values')
            plt.savefig('Figure_7.png')

        return stoch_df

    """
        CCI measures the current price level relative to an average price level over a given period of time. 
        CCI is relatively high when prices are far above their average, but is relatively low when prices are far below their average. 
        In this manner, CCI can be used to identify overbought and oversold levels.
    """
    def commodity_channel_index(self, prices, make_plot, ndays=20):
        prices['TP'] = (prices['High'] + prices['Low'] + prices['Close']) / 3
        prices['sma'] = prices['TP'].rolling(ndays).mean()
        prices['mad'] = prices['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad(), raw=False)
        prices['CCI'] = (prices['TP'] - prices['sma']) / (0.015 * prices['mad'])

        cci_df = prices.copy()
        cci_df.drop(cci_df.iloc[:, 0:-1], inplace=True, axis=1)
        cci_df['CCI'] = cci_df['CCI'] / cci_df['CCI'].iloc[ndays-1]  # normalized
        adj_close_prices = prices['Adj Close']
        cci_df['Prices'] = adj_close_prices / adj_close_prices.iloc[0]  # normalized

        if make_plot:
            cci_graph = cci_df.plot(title='Commodity Channel Index for JPM - alata6', fontsize=12,
                                      grid=True)
            cci_graph.set_xlabel('Date')
            cci_graph.set_ylabel('Normalized Values')
            plt.savefig('Figure_8.png')

        return cci_df

    def get_all_data(self, symbols, dates, addSPY=True):
        """Read stock data for given symbols from CSV files."""
        df = pd.DataFrame(index=dates)
        if addSPY and "SPY" not in symbols:  # add SPY for reference, if absent
            symbols = ["SPY"] + list(
                symbols
            )  # handles the case where symbols is np array of 'object'

        for symbol in symbols:
            df_temp = pd.read_csv(
                symbol_to_path(symbol),
                index_col="Date",
                parse_dates=True,
                na_values=["nan"],
            )
            df = df.join(df_temp)
            if symbol == "SPY":  # drop dates SPY did not trade
                df = df.dropna(subset=["SPY"])
            df.dropna(inplace=True)
        return df
