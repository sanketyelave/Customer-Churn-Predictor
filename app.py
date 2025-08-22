# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved Random Forest model
model = joblib.load("churn_model.pkl")

# Define the expected feature columns
feature_columns = [
    'Contract', 'tenure', 'MonthlyCharges', 'TotalCharges', 
    'OnlineSecurity', 'TechSupport', 'InternetService', 
    'PaymentMethod', 'OnlineBackup', 'PaperlessBilling'
]

# Define possible categories for categorical features
categorical_features = {
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'PaperlessBilling': ['Yes', 'No']
}

# Home route
@app.route('/')
def home():
    return "Customer Churn Prediction API is Running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data], columns=feature_columns)

        # Encode categorical variables
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
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
