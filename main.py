import os
import yfinance as yf
from src.utils import install_requirements, parse_args, save_fig
from src.data import load_data, data_split, fill_data
from src.visualizations import (
    plot_df_prices, volatility, seasonal_decomposition_plot,
    dicky_fuller_test, plot_moving_averages, dividend_yield_plot
)
from src.dlmodels import dl_output
from src.evaluation import (
    theils_u, plot_visual_comparison,
    time_series_cross_validation, backtesting_forecast, naive_forecast
)


def main():

    install_requirements("yfinance pandas numpy plotly statsmodels scikit-learn keras tensorflow")
    
    args = parse_args()
    ticker_symbol, start_date, end_date = args.ticker, args.start_date, args.end_date

    full_df = load_data(ticker_symbol, start_date, end_date)
    df_prices, df_actions = data_split(full_df)
    df_filled = fill_data(df_prices)

    fig_prices, fig_volume = plot_df_prices(df_filled)
    vol_fig = volatility(df_filled)
    season_fig = seasonal_decomposition_plot(df_filled)
    dicky_results = dicky_fuller_test(df_filled)
    dicky_fig = dicky_results[2]
    ma_fig = plot_moving_averages(df_filled)
    dy_fig = dividend_yield_plot(df_actions)

    dl_results = dl_output(df_filled)
    lstm_metrics, lstm_fig, lstm_predictions, actual_values = dl_results['LSTM']
    gru_metrics, gru_fig, gru_predictions, _ = dl_results['GRU']

    u_stat = theils_u(actual_values, lstm_predictions)

    series = df_filled['Close'].values.flatten()
    initial_train_size = int(0.7 * len(series))
    forecast_horizon = 1
    step_size = 10
    actual_cv, predictions_cv = time_series_cross_validation(
        naive_forecast, series, initial_train_size, forecast_horizon, step_size
    )
    cv_fig = plot_visual_comparison(actual_cv, predictions_cv)
    cv_fig.update_layout(title="Naive Forecast Cross-Validation")

    rolling_window = 10
    forecast_times, actual_bt, predictions_bt = backtesting_forecast(
        naive_forecast, series, forecast_horizon, rolling_window
    )
    bt_fig = plot_visual_comparison(actual_bt, predictions_bt)
    bt_fig.update_layout(title="Naive Forecast Backtesting")

    os.makedirs("plots", exist_ok=True)

    save_fig(fig_prices, "plots/prices.png")
    save_fig(fig_volume, "plots/volume.png")
    save_fig(vol_fig, "plots/volatility.png")
    save_fig(season_fig, "plots/seasonal_decomposition.png")
    save_fig(dicky_fig, "plots/dickey_fuller.png")
    save_fig(ma_fig, "plots/moving_averages.png")
    save_fig(dy_fig, "plots/dividend_yield.png")
    save_fig(lstm_fig, "plots/lstm_predictions.png")
    save_fig(gru_fig, "plots/gru_predictions.png")
    save_fig(cv_fig, "plots/cross_validation.png")
    save_fig(bt_fig, "plots/backtesting.png")

    report_lines = []

    # 1. Title and Introduction
    report_lines.append(f"# Stock Analysis Report: {ticker_symbol.upper()}")
    report_lines.append("")
    report_lines.append("## 1. Introduction and Overview")
    report_lines.append(f"This report presents an analysis of stock data for the ticker symbol **{ticker_symbol.upper()}** from **{start_date}** to **{end_date}**. "
                        "It covers data loading, visualization, deep learning forecasts, and forecast evaluation metrics.")
    report_lines.append("")

    # 2. Detailed Definitions and Explanations
    report_lines.append("## 2. Definitions and Explanations")
    report_lines.append("### 2.1 Basic Stock Metrics")
    report_lines.append("- **Close Price**: The final trading price of the stock for the day. This value is critical for gauging market sentiment at the end of trading sessions.")
    report_lines.append("- **Volume**: The total number of shares traded during a given period. High volume often signals increased market interest or volatility.")
    report_lines.append("")
    report_lines.append("### 2.2 Time Series and Statistical Analysis")
    report_lines.append("- **Volatility**: A measure of how much the stock price fluctuates over time, calculated over intervals such as 30-day and 90-day periods.")
    report_lines.append("- **Seasonal Decomposition**: A process that splits a time series into trend, seasonal, and residual components to better understand underlying patterns.")
    report_lines.append("- **Dickey-Fuller Test**: A statistical test used to determine if a time series is stationary, which is a necessary condition for many forecasting models.")
    report_lines.append("")
    report_lines.append("### 2.3 Technical Indicators")
    report_lines.append("- **Moving Averages**: Techniques that smooth out short-term fluctuations to reveal longer-term trends. Common periods include 30, 90, 120, and 365 days.")
    report_lines.append("- **Dividend Yield**: The ratio of a company’s annual dividend relative to its share price, used to assess the income potential of an investment.")
    report_lines.append("")
    report_lines.append("### 2.4 Forecasting Models and Evaluation Metrics")
    report_lines.append("- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network that captures long-term dependencies, used here to forecast closing prices.")
    report_lines.append("- **GRU (Gated Recurrent Unit)**: A streamlined version of LSTM with fewer parameters, offering similar forecasting capabilities with improved computational efficiency.")
    report_lines.append("- **Theil's U Statistic**: A metric that compares the performance of the forecast model to a naive forecast. Lower values indicate better performance.")
    report_lines.append("- **Naive Forecast**: A baseline forecasting method that simply carries forward the last observed value as the future prediction.")
    report_lines.append("- **Cross-Validation**: A method to assess model generalizability by partitioning data into training and validation sets.")
    report_lines.append("- **Backtesting**: The process of evaluating a forecasting model using historical data to simulate its performance in real-world scenarios.")
    report_lines.append("")

    # 3. Data Loading and Preparation
    report_lines.append("## 3. Data Loading and Preparation")
    report_lines.append(f"The dataset was obtained from Yahoo Finance for the ticker symbol **{ticker_symbol.upper()}** between **{start_date}** and **{end_date}**. "
                        "Missing values were filled using interpolation to create a continuous time series for analysis.")
    report_lines.append("")

    # 4. Visualizations and Detailed Results
    report_lines.append("## 4. Visualizations and Detailed Results")
    report_lines.append("")
    # 4.1 Stock Prices
    report_lines.append("### 4.1 Stock Prices")
    report_lines.append("**Description**: This plot displays the daily closing prices over the analysis period.")
    report_lines.append("**Interpretation**: Fluctuations in the closing price indicate overall market trends, investor sentiment, and potential support/resistance levels.")
    report_lines.append("![Stock Prices](plots/prices.png)")
    report_lines.append("")
    # 4.2 Trading Volume
    report_lines.append("### 4.2 Trading Volume")
    report_lines.append("**Description**: This chart shows the volume of shares traded each day.")
    report_lines.append("**Interpretation**: High trading volume may signal increased market interest or impending price movements.")
    report_lines.append("![Trading Volume](plots/volume.png)")
    report_lines.append("")
    # 4.3 Volatility
    report_lines.append("### 4.3 Volatility")
    report_lines.append("**Description**: This visualization illustrates the stock's volatility over 30-day and 90-day intervals.")
    report_lines.append("**Interpretation**: Volatility is a key risk metric; higher volatility implies greater uncertainty and risk.")
    report_lines.append("![Volatility](plots/volatility.png)")
    report_lines.append("")
    # 4.4 Seasonal Decomposition
    report_lines.append("### 4.4 Seasonal Decomposition")
    report_lines.append("**Description**: The seasonal decomposition plot breaks the stock price data into trend, seasonal, and residual components.")
    report_lines.append("**Interpretation**: This helps identify underlying patterns, cyclical behavior, and irregular fluctuations.")
    report_lines.append("![Seasonal Decomposition](plots/seasonal_decomposition.png)")
    report_lines.append("")
    # 4.5 Dickey-Fuller Test
    report_lines.append("### 4.5 Dickey-Fuller Test (Differenced Close Prices)")
    report_lines.append("**Description**: This plot shows the differenced close prices used for assessing the stationarity of the time series.")
    report_lines.append("**Interpretation**: A stationary time series is crucial for reliable forecasting; the Dickey-Fuller test helps determine stationarity.")
    report_lines.append("![Dickey-Fuller Test](plots/dickey_fuller.png)")
    report_lines.append("")
    # 4.6 Moving Averages
    report_lines.append("### 4.6 Moving Averages")
    report_lines.append("**Description**: This plot overlays several moving averages (30, 90, 120, and 365 days) on the closing price data.")
    report_lines.append("**Interpretation**: Moving averages smooth out short-term fluctuations, clarifying the overall trend in stock prices.")
    report_lines.append("![Moving Averages](plots/moving_averages.png)")
    report_lines.append("")
    # 4.7 Dividend Yield
    report_lines.append("### 4.7 Dividend Yield")
    report_lines.append("**Description**: This visualization displays the dividend yield over time.")
    report_lines.append("**Interpretation**: Dividend yield is an important metric for income-focused investors, indicating the cash return on investment relative to the stock price.")
    report_lines.append("![Dividend Yield](plots/dividend_yield.png)")
    report_lines.append("")

    # 5. Deep Learning Model Forecasts
    report_lines.append("## 5. Deep Learning Model Forecasts")
    report_lines.append("The deep learning models (LSTM and GRU) were trained on 90% of the data to forecast the stock's closing price.")
    report_lines.append("")
    # 5.1 LSTM Model
    report_lines.append("### 5.1 LSTM Model Forecast")
    report_lines.append("**Metrics:**")
    for key, value in lstm_metrics.items():
        report_lines.append(f"- **{key}**: {value:.4f}")
    report_lines.append("**Visualization:** The plot below compares the LSTM model's predictions to the actual stock prices.")
    report_lines.append("![LSTM Predictions](plots/lstm_predictions.png)")
    report_lines.append("")
    # 5.2 GRU Model
    report_lines.append("### 5.2 GRU Model Forecast")
    report_lines.append("**Metrics:**")
    for key, value in gru_metrics.items():
        report_lines.append(f"- **{key}**: {value:.4f}")
    report_lines.append("**Visualization:** The plot below shows the GRU model's predictions alongside the actual values.")
    report_lines.append("![GRU Predictions](plots/gru_predictions.png)")
    report_lines.append("")

    # 6. Forecast Evaluation
    report_lines.append("## 6. Forecast Evaluation")
    # 6.1 Theil's U Statistic
    report_lines.append("### 6.1 Theil's U Statistic")
    report_lines.append(f"- **Theil's U**: {u_stat:.4f}")
    report_lines.append("**Interpretation:** A lower Theil's U value indicates that the forecasting model performs better than the naive forecast.")
    report_lines.append("")
    # 6.2 Cross-Validation
    report_lines.append("### 6.2 Cross-Validation (Naive Forecast)")
    report_lines.append("**Description:** This plot shows the results of cross-validation using a naive forecasting approach.")
    report_lines.append("**Interpretation:** Cross-validation assesses the consistency of the naive forecast across different data segments.")
    report_lines.append("![Cross-Validation](plots/cross_validation.png)")
    report_lines.append("")
    # 6.3 Backtesting Forecast
    report_lines.append("### 6.3 Backtesting Forecast (Naive Forecast)")
    report_lines.append("**Description:** The backtesting chart evaluates forecast performance using a rolling window on historical data.")
    report_lines.append("**Interpretation:** Backtesting helps verify that the forecast model's performance is robust over time.")
    report_lines.append("![Backtesting Forecast](plots/backtesting.png)")
    report_lines.append("")

    # 7. Conclusion
    report_lines.append("## 7. Conclusion")
    report_lines.append("This report summarizes the key findings from the stock analysis. The detailed visualizations reveal important trends and patterns in the stock's performance, while the deep learning models offer forecasts benchmarked against naive methods. "
                        "The comprehensive definitions and explanations provided throughout this report are intended to assist in interpreting the analysis results and making informed investment decisions.")

    report_content = "\n".join(report_lines)

    with open("report.md", "w") as f:
        f.write(report_content)

    print("Markdown report generated as report.md.")

    
    report_content = "\n".join(report_lines)
    with open("report.md", "w") as f:
        f.write(report_content)
    
    print("Report generated as report.md with accompanying plots in the 'plots' directory.")

if __name__ == "__main__":
    main()
