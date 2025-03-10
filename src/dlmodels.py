# src/models/dlmodels.py
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def dl_output(df):
    """Build and evaluate LSTM and GRU models on the Close price data."""
    def metric_funcs(actual, pred):
        def MAPE(actual, pred):
            return mean_absolute_percentage_error(actual, pred)
        def R2(actual, pred):
            return r2_score(actual, pred)
        def MAE(actual, pred):
            return mean_absolute_error(actual, pred)
        def RMSE(actual, pred):
            return np.sqrt(mean_squared_error(actual, pred))
        return MAPE, R2, MAE, RMSE

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.90))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len, :]

    x_train, y_train = [], []
    for i in range(90, len(train_data)):
        x_train.append(train_data[i-90:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    def rnn_model(model_type):
        model = Sequential()
        if model_type == 'GRU':
            from keras.layers import GRU
            model.add(GRU(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(GRU(64, return_sequences=False))
        elif model_type == 'LSTM':
            from keras.layers import LSTM
            model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

        test_data = scaled_data[training_data_len - 90:]
        x_test = []
        y_test = dataset[training_data_len:]
        for i in range(90, len(test_data)):
            x_test.append(test_data[i-90:i, 0])
        x_test = np.array(x_test).reshape(len(x_test), 90, 1)
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate metrics
        MAPE, R2, MAE, RMSE = metric_funcs(y_test, predictions)
        metrics = {
            'RMSE': RMSE(y_test, predictions),
            'MAPE': MAPE(y_test, predictions),
            'MAE': MAE(y_test, predictions),
            'R2': R2(y_test, predictions)
        }

        # Visualization
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid = valid.copy()
        valid['Predictions'] = predictions
        trace_train = go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train', line=dict(color='blue'))
        trace_valid = go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual', line=dict(color='blue'))
        trace_pred = go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted', line=dict(color='red', dash='dash'))
        fig = go.Figure(data=[trace_train, trace_valid, trace_pred])
        fig.update_layout(title=f'{model_type} Model Predictions', xaxis_title='Date', yaxis_title='Close Price')
        return metrics, fig

    return {
        'LSTM': rnn_model('LSTM'),
        'GRU': rnn_model('GRU')
    }
