from preprocessing import preprocess
from flask import Flask, render_template, request
from waitress import serve
import numpy as np
from flask import Flask, request, render_template
import pickle

# Creating a Flask app instance
app = Flask(__name__)
model = pickle.load(open("../model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Getting user input from form
    user_input = [np.array([str(x) for x in request.form.values()])]
    # Preprocessing the input
    preprocessed_input = preprocess(user_input)
    # Predicting
    prediction = np.rint(model.predict(preprocessed_input))
    prediction_text = f"PREDICTION \n"
    prediction_scores = f"{user_input[0][0]}: {int(prediction[0][0])}  " \
                        f"\n{user_input[0][1]}: {int(prediction[0][1])}"

    print(prediction_text, prediction_scores)
    # Rendering the result on the page
    return render_template("index.html", prediction_text=prediction_text, predicted_scores=prediction_scores)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port="8080")
