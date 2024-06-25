from flask import Flask, request, jsonify
from flask_cors import CORS
from Historic_Crypto import HistoricalData, Cryptocurrencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import date, timedelta
import plotly.graph_objs as go

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Cryptocurrency Price Prediction API"

def get_available_currencies():
    try:
        all_cryptos_df = Cryptocurrencies().find_crypto_pairs()
        crypto_options = sorted(set([pair.split('-')[0] for pair in all_cryptos_df['id']]))
        return crypto_options
    except Exception as e:
        return str(e), 500

def get_data(cryptos, currency):
    pair = f'{cryptos}-{currency}'
    try:
        all_cryptos_df = Cryptocurrencies().find_crypto_pairs()
        if pair not in all_cryptos_df['id'].values:
            return None, f"{pair} not found in available cryptocurrency pairs."

        coinprices = pd.DataFrame()
        start_date = date(2020, 1, 1)
        end_date = date.today()
        delta = timedelta(days=100)

        while start_date < end_date:
            try:
                tmp = HistoricalData(pair, 60*60*24, start_date.strftime('%Y-%m-%d-00-00'), (start_date + delta).strftime('%Y-%m-%d-00-00'), verbose=False).retrieve_data()
                if tmp.empty:
                    break
                coinprices = pd.concat([coinprices, tmp[['close']]], axis=0)  # Concatenate along rows (axis=0)
            except Exception as e:
                return None, f"Error fetching data for {pair} between {start_date} and {start_date + delta}: {str(e)}"

            start_date += delta

        if coinprices.empty:
            return None, f"No data available for {pair} from {date(2020, 1, 1)} to {date.today()}"

        coinprices.index = pd.to_datetime(coinprices.index)
        coinprices = coinprices.ffill()
        
        return coinprices, None
    
    except Exception as e:
        return None, str(e)

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
        return None, None, str(e)

def predict_future(model, data, scaler, time_step=60, steps=180):
    try:
        data = scaler.transform(data)
        future_inputs = data[-time_step:].reshape(1, time_step)

        predictions = []
        for _ in range(steps):
            pred = model.predict(future_inputs)
            predictions.append(pred[0])
            future_inputs = np.roll(future_inputs, -1)
            future_inputs[0, -1] = pred[0]

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        return predictions.flatten()

    except Exception as e:
        return None, str(e)

@app.route('/available_currencies', methods=['GET'])
def available_currencies():
    try:
        currencies = get_available_currencies()
        return jsonify(currencies)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical_data', methods=['POST'])
def historical_data():
    data = request.json
    cryptos = data.get('cryptos')
    currency = data.get('currency')

    coinprices, error = get_data(cryptos, currency)
    if error:
        return jsonify({'error': error}), 500

    try:
        # Plotting using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=coinprices.index, y=coinprices['close'], mode='lines', name=f'{cryptos}-{currency} Historical Prices'))
        fig.update_layout(
            title=f'{cryptos}-{currency} Historical Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            showlegend=True,
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

        return jsonify({'graph': fig.to_json()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cryptos = data.get('cryptos')
    currency = data.get('currency')
    steps = data.get('steps', 180)

    coinprices, error = get_data(cryptos, currency)
    if error:
        return jsonify({'error': error}), 500

    data = coinprices['close'].values.reshape(-1, 1)
    X, y, scaler, error = prepare_data(data)
    if error:
        return jsonify({'error': error}), 500

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    future_predictions, error = predict_future(model, data[-60:], scaler, steps=steps)
    if error:
        return jsonify({'error': error}), 500

    future_dates = pd.date_range(start=coinprices.index[-1], periods=len(future_predictions)+1, freq='D')[1:]
    return jsonify({
        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
        'predictions': future_predictions.tolist()
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
