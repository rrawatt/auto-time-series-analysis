# src/visualizations.py
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.stattools import adfuller

def plot_df_prices(df):
    """Plot stock prices and volume."""
    fig_prices = go.Figure()
    fig_vol = go.Figure()
    for col in df.columns[:5]:
        fig_prices.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=str(col)))
    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volume'], mode='lines', name='Volume'))
    return fig_prices, fig_vol

def volatility(df):
    """Plot stock volatility."""
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
    df['Volatility_90'] = df['Daily_Return'].rolling(window=90).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_30'], mode='lines', name='30-Day Volatility', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_90'], mode='lines', name='90-Day Volatility', line=dict(color='red')))
    fig.update_layout(title='Stock Volatility Over Time', xaxis_title='Date', yaxis_title='Volatility')
    return fig

def seasonal_decomposition_plot(df):
    """Plot seasonal decomposition of the close price."""
    result = seasonal_decompose(df['Close'], model='additive', period=90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal'))
    fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode='lines', name='Residual'))
    fig.update_layout(title='Seasonal Decomposition of Stock Price', xaxis_title='Date', yaxis_title='Price')
    return fig

def dicky_fuller_test(df):
    """Perform and plot Dickey-Fuller test results."""
    close = df['Close']
    differenced_close = close.diff().dropna()
    dicftest = adfuller(close)
    diff_dicftest = adfuller(differenced_close)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=differenced_close.index, y=differenced_close.values, mode='lines', name='Differenced Close Prices'))
    fig.update_layout(title="Differenced Close Prices", xaxis_title="Date", yaxis_title="Differenced Close Price")
    return dicftest, diff_dicftest, fig

def plot_moving_averages(df):
    """Plot moving averages of the close price."""
    close = df['Close']
    ma30 = close.rolling(window=30).mean()
    ma90 = close.rolling(window=90).mean()
    ma120 = close.rolling(window=120).mean()
    ma365 = close.rolling(window=365).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma30, mode='lines', name='30-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=ma90, mode='lines', name='90-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=ma120, mode='lines', name='120-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=ma365, mode='lines', name='365-Day MA'))
    fig.update_layout(title='Moving Averages of Close Price', xaxis_title='Date', yaxis_title='Price')
    return fig

def dividend_yield_plot(df):
    """Plot dividend yield over time."""
    df['Yield'] = df['Dividends'] / df['Close']
    df_filtered = df[df['Dividends'] != 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Yield'], mode='markers', name='Yield'))
    fig.update_layout(title='Dividend Yield Over Time', xaxis_title='Date', yaxis_title='Yield', template='plotly_dark')
    return fig
