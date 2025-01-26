from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import numpy as np

def dl_output(df):
    def MAPE(actual,prediction):
        mape = mean_absolute_percentage_error(actual,prediction)
        return mape

    def R2(actual,prediction):
        r2= r2_score(actual,prediction)
        return r2

    def MAE(actual,prediction):
        mae= mean_absolute_error(actual,prediction)
        return mae

    def RMSE(actual,prediction):
        rmse= np.sqrt(mean_squared_error(actual,prediction))
        return rmse

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .90 ))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []

    for i in range(90, len(train_data)):
        x_train.append(train_data[i-90:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    def rnn_model(mod):
        model = Sequential()

        if mod == 'GRU':
            model = Sequential()
            model.add(GRU(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(GRU(64, return_sequences=False))

        elif mod == 'LSTM':
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))

        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=1, epochs=1)


        test_data = scaled_data[training_data_len - 90: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(90, len(test_data)):
            x_test.append(test_data[i-90:i, 0])

        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = RMSE(y_test,predictions)
        mape =MAPE(y_test,predictions)
        mae = MAE(y_test,predictions)
        r2 = R2(y_test,predictions)

        print("RMSE:" ,rmse, "\nMAPE:" ,mape, "\nMAE:" ,mae, "\nR2:" ,r2)

        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        trace_train = go.Scatter(
            x=train.index,
            y=train['Close'],
            mode='lines',
            name='Train',
            line=dict(color='blue')
        )

        trace_valid_close = go.Scatter(
            x=valid.index,
            y=valid['Close'],
            mode='lines',
            name='Val',
            line=dict(color='blue')
        )

        trace_valid_pred = go.Scatter(
            x=valid.index,
            y=valid['Predictions'],
            mode='lines',
            name='Predictions',
            line=dict(color='red', dash='dash')
        )

        layout = go.Layout(
            title='Model',
            xaxis_title='Date',
            yaxis_title='Close Price USD ($)',
            legend=dict(x=0, y=-0.2)
        )

        fig = go.Figure(data=[trace_train, trace_valid_close, trace_valid_pred], layout=layout)

        return (rmse, mape, mae, r2, fig)
    
    return (rnn_model('LSTM'), rnn_model('GRU'))

