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
        print("Model file not found. Retraining required.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def retrain_model(new_data):
    try:
        past_data = pd.read_csv("expenses.csv")  # Ensure this file exists and is correctly formatted
        print("Past data loaded successfully.")
        
        df = pd.concat([past_data, new_data])
        df = df.drop_duplicates()
        df = df.sort_index()

        sarimax_model = SARIMAX(df['Expense'], order=(0, 0, 1), seasonal_order=(0, 1, 1, 12))
        sarimax_fitted = sarimax_model.fit(disp=False)
        print("Model retrained successfully.")

        with open("model_SARIMAX_fit.pkl", "wb") as model_file:
            pickle.dump(sarimax_fitted, model_file)

        df.to_pickle("past_data.pkl")
        print("Model saved successfully.")
        return sarimax_fitted
    except Exception as e:
        print(f"Error retraining model: {e}")
        return None

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"error": "No data received"}), 400
        
        new_data = pd.DataFrame(request_data)
        new_data.rename(columns={"date": "Date", "amount": "Expense"}, inplace=True)
        new_data["Date"] = pd.to_datetime(new_data["Date"])
        new_data.set_index("Date", inplace=True)
        new_data = new_data.groupby(new_data.index).sum()

        model = load_model()
        if model is None:
            print("Model not found, retraining...")
            model = retrain_model(new_data)
        
        if model is None:
            return jsonify({"error": "Model could not be trained or loaded."}), 500

        next_day_forecast = model.forecast(steps=1).tolist()
        next_week_forecast = model.forecast(steps=7).tolist()
        next_month_forecast = model.forecast(steps=30).tolist()

        next_month_sum = sum(next_month_forecast)
        next_week_sum = sum(next_week_forecast)

        return jsonify({
            "next_day_forecast": next_day_forecast,
            "next_week_forecast_sum":next_week_sum,
            "next_week_forecast": next_week_forecast,
            "next_month_forecast": next_month_sum
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
