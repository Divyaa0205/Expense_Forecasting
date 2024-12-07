from flask import Flask, request, jsonify
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

def load_model():
    try:
        with open("model_SARIMAX_fit.pkl", "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        return None

def retrain_model(new_data):
    try:
        historical_data = pd.read_csv("expenses.csv")
        historical_data["Date"] = pd.to_datetime(historical_data["Date"], errors='coerce')
        historical_data["Expense"] = pd.to_numeric(historical_data["Expense"], errors='coerce')
        historical_data = historical_data.dropna().set_index("Date").sort_index()
        df = pd.concat([historical_data, new_data])
        df = df.drop_duplicates().asfreq("D").fillna(method="ffill")
        sarimax_model = SARIMAX(df['Expense'], order=(0, 0, 1), seasonal_order=(0, 1, 1, 12), 
                                enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fitted = sarimax_model.fit(disp=False, maxiter=50)
        with open("model_SARIMAX_fit.pkl", "wb") as model_file:
            pickle.dump(sarimax_fitted, model_file)
        df.to_pickle("historical_data.pkl")
        return sarimax_fitted
    except Exception:
        return None

@app.route("/", methods=["POST"])
def forecast():
    try:
        request_data = request.get_json()
        expenses = request_data.get("expenses")
        if not expenses:
            return jsonify({"error": "No expense data provided."}), 400

        df = pd.DataFrame(expenses)
        df["Date"] = pd.to_datetime(df["date"], errors='coerce')
        df["Expense"] = pd.to_numeric(df["amount"], errors='coerce')
        df = df.dropna().set_index("Date").sort_index()

        if df.empty:
            return jsonify({"error": "Expense data is empty."}), 400

        model = load_model()
        if model is None:
            model = retrain_model(df)
            if model is None:
                return jsonify({"error": "Failed to train SARIMAX model."}), 500
        else:
            model = retrain_model(df)
            if model is None:
                return jsonify({"error": "Failed to retrain SARIMAX model."}), 500

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



