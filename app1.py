from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('demand_forecasting_model.pkl')

# Extract features the model expects (from training)
model_features = model.feature_names_in_.tolist()

@app.route('/predict/demand', methods=['POST'])
def predict_demand():
    try:
        # Parse input JSON
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # One-hot encode
        df_encoded = pd.get_dummies(df)

        # Align with training columns
        df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

        # Predict
        prediction = model.predict(df_encoded)[0]

        return jsonify({'predicted_demand_units': round(float(prediction), 0)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Coconut Demand Forecasting API is running!"

if __name__ == '__main__':
    app.run(debug=True)
