from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
import json
import random

app = Flask(__name__)
model = tf.keras.models.load_model("chatbot_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
lbl_encoder = pickle.load(open("label_encoder.pkl", "rb"))

with open("responses.json") as file:
    responses = json.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    msg = request.form["msg"]
    result = tokenizer.texts_to_sequences([msg])
    result = tf.keras.preprocessing.sequence.pad_sequences(result, maxlen=20, truncating='post')
    prediction = model.predict(result)
    tag = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]
    return random.choice(responses[tag])

if __name__ == "__main__":
    app.run(debug=False)

