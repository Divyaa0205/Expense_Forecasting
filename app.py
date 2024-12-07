from flask import Flask, request, jsonify
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def load_model():
    try:
        with open("model_SARIMAX_fit.pkl", "rb") as model_file:
            logging.info("Model loaded successfully.")
            return pickle.load(model_file)
    except FileNotFoundError:
        logging.warning("Model file not found. Retraining required.")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def retrain_model(new_data):
    try:
        # Load historical data
        try:
            historical_data = pd.read_csv("expenses.csv")
        except FileNotFoundError:
            logging.warning("Historical data file not found. Using only new data.")
            historical_data = pd.DataFrame()

        # Combine historical and new data
        df = pd.concat([historical_data, new_data])
        df = df.drop_duplicates()
        df["Date"] = pd.to_datetime(df.index)
        df = df.sort_index()

        # Train SARIMAX model
        sarimax_model = SARIMAX(df["Expense"], order=(0, 0, 1), seasonal_order=(0, 1, 1, 12))
        sarimax_fitted = sarimax_model.fit(disp=False)

        # Save the trained model
        with open("model_SARIMAX_fit.pkl", "wb") as model_file:
            pickle.dump(sarimax_fitted, model_file)
        df.to_csv("expenses.csv", index=True)
        logging.info("Model retrained and saved successfully.")
        return sarimax_fitted
    except Exception as e:
        logging.error(f"Error retraining model: {str(e)}")
        return None

@app.route("/", methods=["POST"])
def forecast():
    try:
        # Parse JSON request
        request_data = request.get_json()
        if not request_data or "expenses" not in request_data:
            return jsonify({"error": "No expense data provided."}), 400

        # Create DataFrame from input
        expenses = request_data["expenses"]
        df = pd.DataFrame(expenses)
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df["Expense"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df[["Date", "Expense"]].dropna().set_index("Date")

        if df.empty:
            return jsonify({"error": "Expense data is empty or invalid."}), 400

        # Load or retrain the model
        model = load_model()
        if model is None:
            model = retrain_model(df)
            if model is None:
                return jsonify({"error": "Failed to train the model."}), 500
        else:
            model = retrain_model(df)

        # Forecast future expenses
        next_day_forecast = model.forecast(steps=1).tolist()
        next_week_forecast = model.forecast(steps=7).tolist()
        next_month_forecast = model.forecast(steps=30).tolist()
        next_month_sum = sum(next_month_forecast)

        # Return predictions
        return jsonify({
            "next_day_forecast": next_day_forecast,
            "next_week_forecast": next_week_forecast,
            "next_month_forecast_sum": next_month_sum
        })

    except Exception as e:
        logging.error(f"Error in forecast endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)




