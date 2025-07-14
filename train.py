import json
import numpy as np
import tensorflow as tf
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open("dataset.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels_encoded = lbl_encoder.transform(training_labels)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', maxlen=20)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, np.array(training_labels_encoded), epochs=300)

model.save("chatbot_model.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(lbl_encoder, open("label_encoder.pkl", "wb"))
with open("responses.json", "w") as f:
    json.dump(responses, f)
