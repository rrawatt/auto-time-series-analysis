import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.stattools import adfuller

def plot_df_prices(df):
    """ Plot stock price and volume over time """
    fig_prices = go.Figure()
    fig_vol = go.Figure()

    for col in df.columns[:5]:
        fig_prices.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=str(col)))

    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volume'], mode='lines', name='Volume'))

    return (fig_prices, fig_vol)

def volatility(df):
    """ Plot the stock volatility over time """
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
    df['Volatility_90'] = df['Daily_Return'].rolling(window=90).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_30'], mode='lines', name='30-Day Volatility', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_90'], mode='lines', name='90-Day Volatility', line=dict(color='red')))
    fig.update_layout(title='Stock Volatility Over Time', xaxis_title='Date', yaxis_title='Volatility')

    return fig

def seasonal_decomposition(df):
    """ Plot seasonal decomposition of the close price """
    close = df['Close']
    result = seasonal_decompose(close, model='additive', period=90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal'))
    fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode='lines', name='Residual'))

    fig.update_layout(title='Seasonal Decomposition of Stock Price', xaxis_title='Date', yaxis_title='Price')
    return fig

    
def dicky_fuller(df):
    """ Perform Dickey-Fuller test on the Close prices """
    close = df['Close']
    differenced_close = close.diff().dropna()
    dicftest = adfuller(close)
    diff_dicftest = adfuller(differenced_close)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=differenced_close.index,
        y=differenced_close.values,
        mode='lines',
        name='Differenced Close Prices'
    ))

    # Update layout
    fig.update_layout(
        title="Differenced Close Prices",
        xaxis_title="Date",
        yaxis_title="Differenced Close Price",
    )

    # Show the plot
    return (dicftest, diff_dicftest, fig)

def plot_moving_averages(df):
    """ Plot moving averages on stock data """
    close = df['Close']
    moving_avg_30 = close.rolling(window=30).mean()
    moving_avg_90 = close.rolling(window=90).mean()
    moving_avg_120 = close.rolling(window=120).mean()
    moving_avg_365 = close.rolling(window=365).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=moving_avg_30, mode='lines', name='30-Day Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=moving_avg_90, mode='lines', name='90-Day Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=moving_avg_120, mode='lines', name='120-Day Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=moving_avg_365, mode='lines', name='365-Day Moving Average'))

    fig.update_layout(title='Moving Averages of Close Price', xaxis_title='Date', yaxis_title='Price')
    return fig

def dividend_yield(df):
    df['Yield'] = df['Dividends'] / df['Close']

    df_filtered = df[df['Dividends'] != 0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_filtered.index,  # Date is the index
        y=df_filtered['Yield'],
        mode='markers',  # Use markers for scatter plot
        name='Yield',
    ))

    fig.update_layout(
        title='Yield vs Time (Dividends != 0)',
        xaxis_title='Date',
        yaxis_title='Yield',
        template='plotly_dark'  # Optional, choose your template
    )

    return fig
