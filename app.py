from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    price_model = pickle.load(f)

# Define the expected input features (before encoding)
price_raw_features = ['year', 'month', 'day', 'province', 'district', 'market']

# Train-time feature list after encoding
price_model_features = price_model.feature_names_in_.tolist()

# Preprocess function for price prediction
def preprocess_price_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=price_model_features, fill_value=0)
    return df

@app.route('/predict/price', methods=['POST'])
def predict_price():
    try:
        data = request.json
        processed = preprocess_price_input(data)
        prediction = price_model.predict(processed)[0]
        return jsonify({'predicted_price': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Coconut Price Forecasting API is running!"

if __name__ == '__main__':
    app.run(debug=True)
