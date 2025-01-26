import subprocess
import sys
import importlib.util
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import LinearRegressionModel, RandomForest, XGBModel
from darts.metrics import rmse, mape, r2_score, mae
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
from utils import install_requirements, parse_args, load_data, data_split, fill_data
from visualizations import plot_df_prices, volatility, seasonal_decomposition, dicky_fuller, plot_moving_averages, dividend_yield
from mlmodels import ml_output
from dlmodels import dl_output

def install_requirements(mod):
    """
    Installs required Python libraries from a string of module names, but only if they are not already installed.

    Args:
        modules_string (str): A space-separated string of library names to install.
    """
    # Split the string into a list of module names
    modules = mod.split()

    for module in modules:
        # Check if the module is already installed
        if importlib.util.find_spec(module) is None:
            try:
                # Run pip install for each module
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"Successfully installed {module}.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while installing {module}: {e}")
                sys.exit(1)

modules = "yfinance pandas numpy plotly statsmodels matplotlib darts scikit-learn keras"
install_requirements(modules)

def parse_args():
    """
    Parse command-line arguments to get a string input (ticker), start date, and end date.
    """
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    
    # Adding arguments for ticker, start date, and end date
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('start_date', type=str, help='Start date for stock data (format: YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='End date for stock data (format: YYYY-MM-DD)')
    
    return parser.parse_args()

def load_data(ticker, start_date, end_date):
    df=pd.DataFrame(ticker.history(start=start_date, end=end_date))
    df.index=pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    return df

def data_split(df):
    df_prices = df[['Open', 'Close', 'High', 'Low', 'Volume']]
    df_corporate_actions = df[['Dividends', 'Close']]
    return (df_prices, df_corporate_actions)

def fill_data(df):
    df.index=pd.to_datetime(df.index.strftime(r'%Y/%m/%d'))
    df = df.reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df.set_index('Date', inplace=True)
    df = df.reindex(full_date_range)
    for i in df.columns[:]:
      df[i] = df[i].interpolate()
    return df

def plot_df_prices(df):
    """ Plot stock price and volume over time """
    fig_prices = go.Figure()
    fig_vol = go.Figure()

    for col in df.columns[:5]:
        fig_prices.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=str(col)))

    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volume'], mode='lines', name='Volume'))

    return (fig_prices, fig_vol)

def volatility(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
    df['Volatility_90'] = df['Daily_Return'].rolling(window=90).std()
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_30'], mode='lines', name='30-Day Rolling Volatility', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility_90'], mode='lines', name='90-Day Rolling Volatility', line=dict(color='red')))

    fig.update_layout(
        title='NVIDIA Volatility Over Time',
        xaxis_title='Date',
        yaxis_title='Volatility',
    )

    return fig

def daily_return(df):
    df['Daily_Return'] = df['Close'].pct_change()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Daily_Return'], mode='lines', name='Daily Return', line=dict(color='red')))
    fig.update_layout(
        title='Daily Return Over Time',
        xaxis_title='Date',
        yaxis_title='%Change',
    )
    return fig

def seasonal_decomposition(df):
    cl = df['Close']
    result = seasonal_decompose(cl, model='additive', period=90)

    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    trace_close = go.Scatter(x=df.index, y=cl, mode='lines', name='Close Price')
    trace_trend = go.Scatter(x=df.index, y=trend, mode='lines', name='Trend')
    trace_seasonal = go.Scatter(x=df.index, y=seasonal, mode='lines', name='Seasonal')
    trace_residual = go.Scatter(x=df.index, y=residual, mode='lines', name='Residual')

    # Create figure and add traces
    fig = go.Figure()
    fig.add_trace(trace_close)
    fig.add_trace(trace_trend)
    fig.add_trace(trace_seasonal)
    fig.add_trace(trace_residual)

    # Update layout for better visualization
    fig.update_layout(title='Quaterly Seasonal Decomposition of Close Price',
                    xaxis_title='Date',
                    yaxis_title='Price')

    # Show the plot
    return fig

def moving_averages(df):
    cl = df['Close']
    window_30 = 30
    moving_avg_30 = cl.rolling(window=window_30).mean()

    window_90 = 90
    moving_avg_90 = cl.rolling(window=window_90).mean()

    window_120 = 120
    moving_avg_120 = cl.rolling(window=window_120).mean()

    window_365 = 365
    moving_avg_365 = cl.rolling(window=window_365).mean()

    trace_close = go.Scatter(x=df.index, y=cl, mode='lines', name='Close Price')
    trace_moving_avg_30 = go.Scatter(x=df.index, y=moving_avg_30, mode='lines', name=f'Monthly Moving Average')
    trace_moving_avg_90 = go.Scatter(x=df.index, y=moving_avg_90, mode='lines', name=f'Quaterly Moving Average')
    trace_moving_avg_120 = go.Scatter(x=df.index, y=moving_avg_120, mode='lines', name=f'Half-Yearly Moving Average')
    trace_moving_avg_365= go.Scatter(x=df.index, y=moving_avg_365, mode='lines', name=f'Yearly Moving Average')


    fig = go.Figure()
    fig.add_trace(trace_close)
    fig.add_trace(trace_moving_avg_30)
    fig.add_trace(trace_moving_avg_90)
    fig.add_trace(trace_moving_avg_120)
    fig.add_trace(trace_moving_avg_365)

    fig.update_layout(title='Moving Averages of Close Price',
                    xaxis_title='Date',
                    yaxis_title='Price')

    return fig

def dicky_fuller(df):
    close = df['Close']
    differenced_close = close.diff().dropna()
    dicftest = adfuller(close)
    diff_dicftest = adfuller(differenced_close)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=differenced_close.index,
        y=differenced_close.values,
        mode='lines',
        name='Differenced Close Prices'
    ))

    # Update layout
    fig.update_layout(
        title="Differenced Close Prices",
        xaxis_title="Date",
        yaxis_title="Differenced Close Price",
    )

    # Show the plot
    return (dicftest, diff_dicftest, fig)


def ml_output(df, mod):

    ts_series = TimeSeries.from_dataframe(df[['Close']])
    train, test = ts_series.split_before(0.7)

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


    predictions_df = pd.DataFrame({
        'Date': test.time_index,
        'Predicted':   forecasted_values

    })
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df.set_index('Date', inplace=True)
    predictions_df['Predicted'] = predictions_df['Predicted'].astype(float)

    pred=TimeSeries.from_dataframe(predictions_df, value_cols='Predicted')
    rmseval =   (test, pred)
    mapeval=mape(test, pred)
    r2=r2_score(test,pred)
    maeval=mae(test, pred)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Plot predicted values
    fig.add_trace(go.Scatter(
        x=predictions_df.index,
        y=predictions_df['Predicted'],
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title='LR Predictions vs Actual Values',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        legend_title='Legend',
        template='plotly_white'
    )

    return (rmseval, mapeval, r2, maeval, fig)

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
    training_data_len = int(np.ceil( len(dataset) * .90 ))
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

def work(ticker_symbol, start_date, end_date):
    install_requirements("yfinance pandas numpy plotly statsmodels darts scikit-learn keras")

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


        