# src/evaluation.py
import numpy as np
import plotly.graph_objects as go

def calculate_residuals(actual, predicted):
    """Compute the residuals between actual and predicted values."""
    residuals = actual - predicted
    return residuals

def plot_residuals(residuals):
    """Plot a histogram of residuals."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=50))
    fig.update_layout(
        title="Residuals Histogram",
        xaxis_title="Residual",
        yaxis_title="Count"
    )
    return fig

def theils_u(actual, predicted):
    """
    Compute Theil's U statistic.
    This statistic compares the forecasting performance of your model against a naive forecast.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    # Naive forecast: use the previous actual value
    naive_forecast = np.roll(actual, 1)
    naive_forecast[0] = actual[0]  # First value cannot be shifted
    mse_model = np.mean((predicted - actual)**2)
    mse_naive = np.mean((naive_forecast - actual)**2)
    u = np.sqrt(mse_model / mse_naive)
    return u

def plot_visual_comparison(actual, predicted):
    """Plot actual vs. predicted values over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(actual)),
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(predicted)),
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="Visual Comparison: Actual vs. Predicted",
        xaxis_title="Time",
        yaxis_title="Value"
    )
    return fig

def time_series_cross_validation(model_func, series, initial_train_size, forecast_horizon, step_size):
    """
    Perform rolling cross-validation on a time series.
    
    Args:
        model_func: Function that accepts training data and forecast horizon, then returns a forecast.
        series: 1D array-like actual time series.
        initial_train_size: Number of observations used for the first training.
        forecast_horizon: Number of steps to forecast each time.
        step_size: Number of observations to move forward in each iteration.
        
    Returns:
        actuals, predictions: Arrays of actual values and forecasts.
    """
    n = len(series)
    predictions = []
    actuals = []
    for start in range(initial_train_size, n - forecast_horizon + 1, step_size):
        train = series[:start]
        test = series[start:start + forecast_horizon]
        forecast = model_func(train, forecast_horizon)
        predictions.extend(forecast)
        actuals.extend(test)
    return np.array(actuals), np.array(predictions)

def backtesting_forecast(model_func, series, forecast_horizon, rolling_window):
    """
    Perform backtesting by generating forecasts using a rolling window approach.
    
    Args:
        model_func: Function that accepts training data and forecast horizon, then returns a forecast.
        series: 1D array-like time series.
        forecast_horizon: Steps ahead to forecast.
        rolling_window: Number of observations to move forward each iteration.
        
    Returns:
        forecast_times: Indices where forecasts were made.
        actual: Actual values at forecast times.
        predictions: Forecasted values.
    """
    n = len(series)
    predictions = []
    forecast_times = []
    for start in range(0, n - forecast_horizon, rolling_window):
        train = series[:start + forecast_horizon]
        forecast = model_func(train, forecast_horizon)
        predictions.append(forecast[0])  # Assuming forecast_horizon=1 for simplicity
        forecast_times.append(start + forecast_horizon)
    predictions = np.array(predictions)
    actual = series[forecast_times]
    return forecast_times, actual, predictions
