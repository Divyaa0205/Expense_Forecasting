from flask import Flask, request, jsonify
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_model():
    try:
        with open("model_SARIMAX_fit.pkl", "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        return None

def fetch_data():
    url = "https://your-api-url/transaction/list"
    headers = {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY3NTFlY2JjY2JiMWU2NGMwMDEyMzFhYyIsImVtYWlsIjoiZGl2eWFuc2gyMzE2OTAxNEBha2dlYy5hYy5pbiIsImlhdCI6MTczMzU0ODE1MywiZXhwIjoxNzY1MTA1NzUzfQ.LbqbcColEee3wui_YZ1CLf3GqbtIVVZrNC_OQ-WqE6g'}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.groupby(df.index).sum()
    return df

def retrain_model(new_data):
    historical_data = pd.read_csv("expenses.csv")
    df = pd.concat([historical_data, new_data])
    df = df.drop_duplicates()
    sarimax_model = SARIMAX(df['Expense'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarimax_fitted = sarimax_model.fit(disp=False)
    with open("model_SARIMAX_fit.pkl", "wb") as model_file:
        pickle.dump(sarimax_fitted, model_file)
    df.to_pickle("historical_data.pkl")
    return sarimax_fitted

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        new_data = fetch_data()
        model = load_model()
        if model is None:
            model = retrain_model(new_data)
        else:
            model = retrain_model(new_data)
        next_day_forecast = model.forecast(steps=1).tolist()
        next_week_forecast = model.forecast(steps=7).tolist()
        next_month_forecast = model.forecast(steps=30).tolist()
        return jsonify({
            "next_day_forecast": next_day_forecast,
            "next_week_forecast": next_week_forecast,
            "next_month_forecast": next_month_forecast
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
