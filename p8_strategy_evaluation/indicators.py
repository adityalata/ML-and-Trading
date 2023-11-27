import matplotlib.pyplot as plt


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username

def simple_moving_average(prices, lookback, symbol, generate_plot, figure_name='simple_moving_average.png'):
    simple_moving_average_df = prices.rolling(window=lookback, min_periods=lookback).mean()
    simple_moving_average_normalized = simple_moving_average_df.copy()

    simple_moving_average_normalized.iloc[lookback - 1:] /= prices.iloc[0]  # normalize for easier comparison
    simple_moving_average_normalized.rename(columns={symbol: 'SMA_' + symbol}, inplace=True)
    simple_moving_average_normalized['Prices'] = prices / prices.iloc[0]  # normalize for easier comparison
    simple_moving_average_normalized['Price/SMA'] = simple_moving_average_normalized['Prices'] / \
                                                    simple_moving_average_normalized['SMA_' + symbol]

    if generate_plot:
        simple_moving_average_graph = simple_moving_average_normalized.plot(
            title='Simple Moving Average (Normalized) for ' + symbol, fontsize=12, grid=True)
        simple_moving_average_graph.set_xlabel('Date')
        simple_moving_average_graph.set_ylabel('Normalized $ Value')

        plt.savefig(figure_name)

    return simple_moving_average_df, simple_moving_average_normalized

"""
the relationship between price and standard deviation Bollinger Bands.
bollinger_band_percentage values greater than 1 indicate price is above the upper band
bollinger_band_percentage values less than 0 indicate price is below the bottom band
"""

def bollinger_band_percentage(prices, lookback, symbol, generate_plot,
                              figure_name='bollinger_band_percentage.png'):
    simple_moving_average_df, _ = simple_moving_average(prices=prices, lookback=lookback, symbol=symbol,
                                                             generate_plot=False)

    rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
    top_band = simple_moving_average_df + (2 * rolling_std)
    bottom_band = simple_moving_average_df - (2 * rolling_std)

    bollinger_band_percentage_df = (prices - bottom_band) / (top_band - bottom_band)
    bollinger_band_percentage_df.rename(columns={symbol: 'Bollinger Band %'}, inplace=True)

    bollinger_band_df = prices.copy()
    bollinger_band_df.rename(columns={symbol: symbol + ' Price'}, inplace=True)
    bollinger_band_df['SMA'] = simple_moving_average_df
    bollinger_band_df['Upper Band'] = top_band
    bollinger_band_df['Lower Band'] = bottom_band

    if generate_plot:
        figure, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        bollinger_band_df.plot(ax=ax1, fontsize=12, grid=True)  # bollinger band subplot
        bollinger_band_percentage_df.plot(ax=ax2, fontsize=12, grid=True)  # bollinger band percentage subplot
        ax2.set_title('Bollinger Band % for ' + symbol + ' - alata6')
        ax2.legend(prop={'size': '10'}, loc=4)
        ax2.set(ylabel='% Value')
        ax1.set_title('Bollinger Bands Against ' + symbol + ' Price Data - alata6')
        ax1.legend(prop={'size': '10'}, loc=4)
        ax1.set(ylabel='$ Value')
        plt.xlabel('Date')
        plt.savefig(figure_name)

    return bollinger_band_df, bollinger_band_percentage_df

"""
(Moving Average Convergence/Divergence Oscillator)
A momentum oscillator based on the difference between two EMAs.
"""

def moving_avg_convergence_divergence(prices, symbol, generate_plot,
                                      figure_name='moving_avg_convergence_divergence.png'):
    ema_26 = prices.ewm(span=26, min_periods=26, adjust=False).mean()
    ema_12 = prices.ewm(span=12, min_periods=12, adjust=False).mean()

    moving_avg_convergence_divergence_df = ema_12 - ema_26
    moving_avg_convergence_divergence_df.rename(columns={symbol: 'MACD'}, inplace=True)
    ema_9 = moving_avg_convergence_divergence_df.ewm(span=9, min_periods=1,
                                                     adjust=False).mean()  # ema of moving_avg_convergence_divergence graph (signal line)
    moving_avg_convergence_divergence_df['Signal Line'] = ema_9

    if generate_plot:
        moving_avg_convergence_divergence_graph = moving_avg_convergence_divergence_df.plot(
            title='Moving Average Convergence Divergence for ' + symbol, fontsize=12,
            grid=True)
        moving_avg_convergence_divergence_graph.set_xlabel('Date')
        moving_avg_convergence_divergence_graph.set_ylabel('Normalized $ Value')

        plt.savefig(figure_name)

    return moving_avg_convergence_divergence_df

"""
The stochastic oscillator is a momentum indicator used to signal trend reversals in the stock market. 
   It describes the current price relative to the high and low prices over a trailing number of previous trading periods.
   fast_period : previous days we want to use to generate our fast signal
   slow_period : previous days we want to use to generate our slow signal
"""

"""
    CCI measures the current price level relative to an average price level over a given period of time. 
    CCI is relatively high when prices are far above their average, but is relatively low when prices are far below their average. 
    In this manner, CCI can be used to identify overbought and oversold levels.
"""