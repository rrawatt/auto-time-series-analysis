# models.py
import pandas as pd
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import LinearRegressionModel, RandomForest, XGBModel
from darts.metrics import rmse, mape, r2_score, mae
from sklearn.preprocessing import MinMaxScaler

def output(df, mod):
    """ Generate machine learning model output using a specified model """
    ts_series = TimeSeries.from_dataframe(df[['Close']])
    train, test = ts_series.split_before(0.7    )

    def generate_forecast(training_data):
        model_instance = mod
        model_instance.fit(training_data)
        predicted_values = model_instance.predict(n=len(test))
        return predicted_values[0].values()[0]

    training_history = train
    forecasted_values = []

    for time_point in range(len(test)):
        predicted_value = generate_forecast(training_history)
        forecasted_values.append(predicted_value)
        actual_observation = test[time_point]
        training_history = training_history.append(actual_observation)

    predictions_df = pd.DataFrame({'Date': test.time_index, 'Predicted': forecasted_values})
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df.set_index('Date', inplace=True)
    predictions_df['Predicted'] = predictions_df['Predicted'].astype(float)

    pred = TimeSeries.from_dataframe(predictions_df, value_cols='Predicted')
    rmseval = rmse(test, pred)
    mapeval = mape(test, pred)
    r2 = r2_score(test, pred)
    maeval = mae(test, pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['Predicted'], mode='lines', name='Predicted'))

    return (rmseval, mapeval, r2, maeval, fig)


def ml_output(df):
    lr=output(df, LinearRegressionModel(lags=7))
    rf=output(RandomForest(7))
    xg=output(XGBModel(7))
    return (lr, rf, xg)
