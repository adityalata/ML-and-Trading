import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, symbol_to_path


class Indicators(object):
    def __init__(self):
        pass

    #
    def generate_charts(self):
        start_date = dt.datetime(2008, 1, 1)
        end_date = dt.datetime(2009, 12, 31)
        dates = pd.date_range(start_date, end_date)
        lookback = 14
        symbols = ['JPM']
        prices = get_data(symbols, dates).drop(['SPY'], axis=1)
        all_data = self.get_all_data(symbols, dates, addSPY=False)
        self.simple_moving_average(prices=prices, lookback=lookback, make_plot=True)
        self.bollinger_band_percentage(prices, lookback, True)
        self.macd(prices, True)
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
            cci_graph.set_ylabel('Normalized $ Value')
            plt.savefig('Figure_7.png')

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
