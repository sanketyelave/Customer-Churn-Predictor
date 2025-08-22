# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
# Initialize Flask app
app = Flask(__name__)

# Load the saved Random Forest model
model = joblib.load("churn_model.pkl")

# All features the model was trained on
feature_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Categorical features and their possible values
categorical_features = {
    'gender': ['Female', 'Male'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

# Home route
@app.route('/')
def home():
    return "Customer Churn Prediction API is Running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check content type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all model features are present
        for col in feature_columns:
            if col not in input_df.columns:
                # Fill missing columns with default values
                if col in categorical_features:
                    input_df[col] = categorical_features[col][0]  # first category as default
                else:
                    input_df[col] = 0  # numeric default

        # Encode categorical features
        for col, categories in categorical_features.items():
            input_df[col] = pd.Categorical(input_df[col], categories=categories).codes

        # Convert numeric columns to float
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numeric_cols] = input_df[numeric_cols].astype(float)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1]  # probability of churn

        # Return response
        response = {
            "prediction": int(prediction),
            "churn_probability": float(prediction_prob)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
