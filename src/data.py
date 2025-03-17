# src/data.py
import pandas as pd
import yfinance as yf

def load_data(ticker_symbol, start_date, end_date):
    ticker = yf.Ticker(ticker_symbol.upper())
    df = pd.DataFrame(ticker.history(start=start_date, end=end_date))
    df.index = pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    return df

def data_split(df):

    df_prices = df[['Close','Volume']]

    if 'Stock Splits' in df.columns:
        df_actions = df[['Dividends', 'Stock Splits','Close']]
    else:
        df_actions = df[['Dividends', 'Close']]
    return df_prices, df_actions

def fill_data(df):
    
    df.index = pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    df = df.reset_index().rename(columns={'index': 'Date'})
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df = df.set_index('Date').reindex(full_date_range)
    for col in df.columns:
        df[col] = df[col].interpolate()
    return df
