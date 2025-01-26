import subprocess
import sys
import importlib.util
import pandas as pd

def install_requirements(mod):
    """ Installs required libraries if they are not already installed """
    modules = mod.split()
    for module in modules:
        if importlib.util.find_spec(module) is None:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"Successfully installed {module}.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while installing {module}: {e}")
                sys.exit(1)

def parse_args():
    """ Parse command-line arguments """
    import argparse
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('start_date', type=str, help='Start date for stock data (format: YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='End date for stock data (format: YYYY-MM-DD)')
    return parser.parse_args()

def load_data(ticker, start_date, end_date):
    """ Load stock data for the given ticker and date range """
    df = pd.DataFrame(ticker.history(start=start_date, end=end_date))
    df.index = pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    return df

def data_split(df):
    """ Split the data into price and corporate actions """
    df_prices = df[['Open', 'Close', 'High', 'Low', 'Volume']]
    df_corporate_actions = df[['Dividends', 'Stock Splits']]
    return (df_prices, df_corporate_actions)

def fill_data(df):
    """ Fill missing data by interpolation """
    df.index = pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    df = df.reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df.set_index('Date', inplace=True)
    df = df.reindex(full_date_range)
    for col in df.columns:
        df[col] = df[col].interpolate()
    return df
