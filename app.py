from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import os

app = Flask(__name__)

# Initialize global variables
data = None
model = None



# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


@app.route('/')
def home():
    """Home endpoint displaying project name and available endpoints"""
    project_info = {
        "Project": "Predictive Analysis for Manufacturing Operations",
        "Available Endpoints": {
            "/upload": "Upload a CSV file containing manufacturing data (POST)",
            "/train": "Train the model on uploaded data and return metrics (POST)",
            "/predict": "Make predictions based on input features (POST)"
        }
    }
    return jsonify(project_info), 200


@app.route('/upload', methods=['POST'])
def upload():
    """Endpoint to upload a CSV file"""
    file = request.files.get('file')
    if file:
        filepath = os.path.join("data", file.filename)
        file.save(filepath)

        global data
        data = pd.read_csv(filepath)
        return jsonify({"message": "File uploaded successfully!"}), 200
    return jsonify({"error": "No file provided"}), 400


@app.route('/train', methods=['POST'])
def train():
    """Endpoint to train the model"""
    global data, model
    if data is None:
        return jsonify({"error": "No data uploaded"}), 400

    try:
        # Prepare data
        X = data[['Temperature', 'Run_Time']]
        y = data['Downtime_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Save the model
        model_path = os.path.join("models", "model.pkl")
        joblib.dump(model, model_path)

        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions"""
    global model
    if model is None:
        return jsonify({"error": "Model not trained"}), 400

    try:
        # Get input JSON
        input_data = request.get_json()
        X_new = pd.DataFrame([input_data])

        # Make predictions
        prediction = model.predict(X_new)
        confidence = max(model.predict_proba(X_new)[0])

        # Return result
        result = {
            "Downtime": "Yes" if prediction[0] == 1 else "No",
            "Confidence": round(confidence, 2)
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
