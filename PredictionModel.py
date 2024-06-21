import streamlit as st
import plotly.express as px
from Historic_Crypto import HistoricalData, Cryptocurrencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import date

# Set up Streamlit for user inputs
st.title('Cryptocurrency Price Prediction')

# Function to get all available cryptocurrencies and their pairs
def get_available_currencies():
    try:
        all_cryptos_df = Cryptocurrencies().find_crypto_pairs()
        crypto_options = sorted(set([pair.split('-')[0] for pair in all_cryptos_df['id']]))
        return crypto_options
    except Exception as e:
        st.error(f"Error fetching cryptocurrencies: {e}")
        return []

# Function to get historical data
def get_data(cryptos, currency):
    pair = f'{cryptos}-{currency}'
    try:
        # Check if pair exists
        all_cryptos_df = Cryptocurrencies().find_crypto_pairs()
        if pair not in all_cryptos_df['id'].values:
            st.error(f"{pair} not found in available cryptocurrency pairs. Please choose a different pair.")
            return pd.DataFrame()
        
        TodaysDate = date.today()
        tmp = HistoricalData(pair, 60*60*24, '2020-01-01-00-00', f'{TodaysDate}-00-00', verbose=False).retrieve_data()
        coinprices = pd.DataFrame({pair: tmp['close']})  # Selecting 'close' price
        coinprices = coinprices.ffill()  # Fill missing values
        return coinprices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to prepare data for XGBoost
def prepare_data(data, time_step=60):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        return X, y, scaler
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None, None

# Function to make future predictions
def predict_future(model, data, scaler, time_step=60, steps=180):  # Predict for 6 months (180 days)
    try:
        data = scaler.transform(data)
        future_inputs = data[-time_step:]
        predictions = []
        for _ in range(steps):
            pred = model.predict(np.reshape(future_inputs, (1, time_step)))
            future_inputs = np.append(future_inputs[1:], pred)
            predictions.append(pred[0])
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions
    except Exception as e:
        st.error(f"Error predicting future prices: {e}")
        return None

# Main process
crypto_options = get_available_currencies()

if crypto_options:

    # User selects mode: Historical Data or Future Predictions
    mode = st.selectbox('Select Mode', ['Historical Data', 'Future Predictions'])

    st.header("Available Cryptocurrencies")

    # User selects cryptocurrency and currency
    cryptos = st.selectbox('Select Coin', crypto_options)
    currency = st.selectbox('Select Currency', ['EUR', 'USD', 'USDT', 'GBP', 'JPY', 'KRW'])

    # Main process for each selected cryptocurrency
    if cryptos and currency and st.button('Show Predictions'):
        st.header(f'{cryptos}-{currency}')

        coinprices = get_data(cryptos, currency)
        if not coinprices.empty:

            # Use the column name directly for selected cryptocurrency
            selected_column = f'{cryptos}-{currency}'

            if selected_column in coinprices.columns:

                if mode == 'Historical Data':
                    # Plot historical data using Plotly Express
                    fig = px.line(
                        x=coinprices.index, y=coinprices[selected_column],
                        labels={"x": "Date", "y": "Price"},
                        title=f'{cryptos}-{currency} Historical Prices'
                    )
                    # Update layout for dark theme
                    fig.update_layout(
                        template='plotly_dark',
                        xaxis=dict(
                            gridcolor='rgb(75, 75, 75)',
                            tickfont=dict(color='white'),
                            title=dict(text='Date', font=dict(color='white'))
                        ),
                        yaxis=dict(
                            gridcolor='rgb(75, 75, 75)',
                            tickfont=dict(color='white'),
                            title=dict(text='Price', font=dict(color='white'))
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig)

                elif mode == 'Future Predictions':
                    # Prepare data
                    data = coinprices[selected_column].values.reshape(-1, 1)
                    X, y, scaler = prepare_data(data)

                    if X is not None and y is not None and scaler is not None:
                        # Create and train model
                        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)

                        try:
                            with st.spinner('Training the model, please wait...'):
                                model.fit(X, y)

                            # Make future predictions
                            future_predictions = predict_future(model, data, scaler)

                            if future_predictions is not None:
                                # Plot historical and future predictions using Plotly Express
                                future_dates = pd.date_range(start=coinprices.index[-1], periods=len(future_predictions)+1, freq='D')[1:]
                                historical_prices = coinprices[selected_column].values.flatten()
                                combined_prices = np.concatenate((historical_prices, future_predictions.flatten()))
                                combined_dates = pd.concat([coinprices.index, future_dates])

                                fig = px.line(
                                    x=combined_dates,
                                    y=combined_prices,
                                    labels={"x": "Date", "y": "Price"},
                                    title=f'{cryptos}-{currency} Price Prediction'
                                )
                                # Update layout for dark theme
                                fig.update_layout(
                                    template='plotly_dark',
                                    xaxis=dict(
                                        gridcolor='rgb(75, 75, 75)',
                                        tickfont=dict(color='white'),
                                        title=dict(text='Date', font=dict(color='white'))
                                    ),
                                    yaxis=dict(
                                        gridcolor='rgb(75, 75, 75)',
                                        tickfont=dict(color='white'),
                                        title=dict(text='Price', font=dict(color='white'))
                                    ),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white')
                                )
                                st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"Error during model training: {e}")
            else:
                st.error(f"No data found for {cryptos}-{currency}")
