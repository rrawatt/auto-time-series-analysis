# src/data.py
import pandas as pd

def load_data(ticker, start_date, end_date):
    """Load stock data for the given ticker and date range."""
    df = pd.DataFrame(ticker.history(start=start_date, end=end_date))
    df.index = pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    return df

def data_split(df):
    """Split the dataframe into prices and corporate actions."""
    df_prices = df[['Open', 'Close', 'High', 'Low', 'Volume']]
    df_corporate = df[['Dividends', 'Stock Splits']]
    return df_prices, df_corporate

def fill_data(df):
    """Fill missing data by interpolation."""
    df.index = pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    df = df.reset_index().rename(columns={'index': 'Date'})
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df = df.set_index('Date').reindex(full_date_range)
    for col in df.columns:
        df[col] = df[col].interpolate()
    return df
