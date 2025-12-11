from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler at startup
MODEL_PATH = "ai_model/flood_prediction_lstm_model.h5"
SCALER_PATH = "ai_model/flood_risk_scaler.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

TIME_STEPS = 5
NUM_FEATURES = 7
BUFFER = np.zeros((TIME_STEPS, NUM_FEATURES))


def prepare_lstm_input(new_row):
    global BUFFER
    BUFFER = np.roll(BUFFER, -1, axis=0)
    BUFFER[-1] = new_row
    scaled = scaler.transform(BUFFER)
    return scaled.reshape(1, TIME_STEPS, NUM_FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = np.array([
            data["max_temp"],
            data["min_temp"],
            data["rainfall"],
            data["humidity"],
            data["wind_speed"],
            data["cloud_cover"],
            data["water_level"]
        ])

        X = prepare_lstm_input(features)
        flood_prob = float(model.predict(X, verbose=0)[0][0])

        return jsonify({
            "flood_probability": flood_prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return "Flood Prediction Cloud API Running"


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

