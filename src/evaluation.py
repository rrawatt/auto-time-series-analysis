import numpy as np
import plotly.graph_objects as go

def theils_u(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)
    naive_forecast = np.roll(actual, 1)
    naive_forecast[0] = actual[0]
    mse_model = np.mean((predicted - actual)**2)
    mse_naive = np.mean((naive_forecast - actual)**2)
    u = np.sqrt(mse_model / mse_naive)
    return u

def plot_visual_comparison(actual, predicted):

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

def naive_forecast(train, forecast_horizon):
    """Return the last observed value repeated for forecast_horizon steps."""
    last_value = train[-1]
    return [last_value] * forecast_horizon