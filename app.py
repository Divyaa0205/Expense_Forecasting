from flask import Flask, request, jsonify
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import requests

app = Flask(__name__)

def load_model():
    try:
        with open("model_SARIMAX_fit.pkl", "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        return None

def fetch_data(token):
    url = "https://your-api-url/transaction/list"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df['Expense'] = df['amount']
    df.set_index("Date", inplace=True)
    df = df.groupby(df.index).sum()
    return df

def retrain_model(new_data):
    historical_data = pd.read_csv("expenses.csv")
    df = pd.concat([historical_data, new_data])
    df = df.drop_duplicates()
    sarimax_model = SARIMAX(df['Expense'], order=(0, 0, 1), seasonal_order=(0, 1, 1, 12))
    sarimax_fitted = sarimax_model.fit(disp=False)
    with open("model_SARIMAX_fit.pkl", "wb") as model_file:
        pickle.dump(sarimax_fitted, model_file)
    df.to_pickle("historical_data.pkl")
    return sarimax_fitted

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        request_data = request.get_json()
        token = request_data.get("token")
        
        if not token:
            return jsonify({"error": "Token is required"}), 400
        
        new_data = fetch_data(token)
        
        if new_data.empty:
            return jsonify({"error": "No data found"}), 404
        
        model = load_model()
        if model is None:
            model = retrain_model(new_data)
        else:
            model = retrain_model(new_data)
        
        next_day_forecast = model.forecast(steps=1).tolist()
        next_week_forecast = model.forecast(steps=7).tolist()
        next_month_forecast = model.forecast(steps=30).tolist()
        
        next_month_sum = sum(next_month_forecast)
        
        return jsonify({
            "next_day_forecast": next_day_forecast,
            "next_week_forecast": next_week_forecast,
            "next_month_forecast": next_month_sum
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

