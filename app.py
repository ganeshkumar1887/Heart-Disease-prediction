from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("heart_model.pkl")

# 🔹 Home Page (Form UI)
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Prediction from Form
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)