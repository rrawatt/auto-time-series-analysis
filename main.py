# main.py
import yfinance as yf
from utils import install_requirements, parse_args, load_data, data_split, fill_data
from visualizations import plot_df_prices, volatility, seasonal_decomposition, dicky_fuller, plot_moving_averages, dividend_yield
from mlmodels import ml_output
from dlmodels import dl_output


def main():
    # Parse arguments
    args = parse_args()
    ticker_symbol = args.ticker
    start_date = args.start_date
    end_date = args.end_date

    # Install required libraries
    install_requirements("yfinance pandas numpy plotly statsmodels darts scikit-learn keras")

    # Get stock data
    ticker = yf.Ticker(ticker_symbol.upper())
    full_df = load_data(ticker, start_date, end_date)
    df=data_split(full_df)[0]
    df1=df[0]
    df2=df[1]
    fill_data(df)


    plot_prices, plot_volume = plot_df_prices(df1)
    volatility_fig = volatility(df1)
    seasonal_fig = seasonal_decomposition(df1)
    dickey_test = dicky_fuller(df1)
    moving_avg=plot_moving_averages(df1)
    div_yield=dividend_yield(df2)

    ml_output(df1)
    dl_output(df1)


if __name__ == "__main__":
    main()
