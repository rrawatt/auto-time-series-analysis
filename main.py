# main.py
import yfinance as yf
from src.utils import install_requirements, parse_args
from src.data import load_data, data_split, fill_data
from src.visualizations import (
    plot_df_prices, volatility, seasonal_decomposition_plot,
    dicky_fuller_test, plot_moving_averages, dividend_yield_plot
)
from src.dlmodels import dl_output
from src.evaluation import (
    calculate_residuals, plot_residuals, theils_u, plot_visual_comparison,
    time_series_cross_validation, backtesting_forecast
)
import numpy as np
import plotly.graph_objects as go

# A simple naive forecast model for demonstration purposes
def naive_forecast(train, forecast_horizon):
    """Return the last observed value repeated for forecast_horizon steps."""
    last_value = train[-1]
    return [last_value] * forecast_horizon

def main():
    # Install dependencies
    install_requirements("yfinance pandas numpy plotly statsmodels darts scikit-learn keras")

    # Parse command line arguments
    args = parse_args()
    ticker_symbol, start_date, end_date = args.ticker, args.start_date, args.end_date

    # Fetch and prepare data
    ticker = yf.Ticker(ticker_symbol.upper())
    full_df = load_data(ticker, start_date, end_date)
    df_prices, df_actions = data_split(full_df)
    df_filled = fill_data(df_prices)

    # Generate visualizations
    fig_prices, fig_volume = plot_df_prices(df_filled)
    vol_fig = volatility(df_filled)
    season_fig = seasonal_decomposition_plot(df_filled)
    dicky_results = dicky_fuller_test(df_filled)
    ma_fig = plot_moving_averages(df_filled)
    dy_fig = dividend_yield_plot(df_actions)

    # Evaluate deep learning models (using LSTM for example)
    dl_results = dl_output(df_filled)
    lstm_metrics, lstm_fig, lstm_predictions, actual_values = dl_results['LSTM']
    print("LSTM Metrics:", lstm_metrics)

    # --- Additional Evaluation ---

    # Residual Analysis
    residuals = calculate_residuals(np.array(actual_values), np.array(lstm_predictions))
    resid_fig = plot_residuals(residuals)
    resid_fig.show()

    # Theil's U Statistic
    u_stat = theils_u(actual_values, lstm_predictions)
    print("Theil's U statistic:", u_stat)

    # Visual Comparison of Actual vs. Predicted
    comparison_fig = plot_visual_comparison(actual_values, lstm_predictions)
    comparison_fig.show()

    # Cross-Validation using a naive forecast model for demonstration
    series = df_filled['Close'].values.flatten()
    initial_train_size = int(0.7 * len(series))
    forecast_horizon = 1
    step_size = 10
    actual_cv, predictions_cv = time_series_cross_validation(naive_forecast, series, initial_train_size, forecast_horizon, step_size)
    cv_fig = plot_visual_comparison(actual_cv, predictions_cv)
    cv_fig.update_layout(title="Naive Forecast Cross-Validation")
    cv_fig.show()

    # Backtesting using the naive forecast model (assuming forecast_horizon=1)
    rolling_window = 10
    forecast_times, actual_bt, predictions_bt = backtesting_forecast(naive_forecast, series, forecast_horizon, rolling_window)
    bt_fig = plot_visual_comparison(actual_bt, predictions_bt)
    bt_fig.update_layout(title="Naive Forecast Backtesting")
    bt_fig.show()

    # Optionally, show the original figures
    fig_prices.show()
    vol_fig.show()
    season_fig.show()
    ma_fig.show()
    dy_fig.show()
    lstm_fig.show()

if __name__ == "__main__":
    main()
