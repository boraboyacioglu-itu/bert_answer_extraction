import json
import numpy as np

import random
import pickle

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


with open('intents.json') as file:
    data = json.load(file)

# Define the colours for terminal output.
cr = "\033[91m"
cg = "\033[92m"
cb = "\033[94m"
ss = "\033[0m"

def chat():
    # Load trained model.
    model = keras.models.load_model('data/chat_model')

    # Load tokenizer object.
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load label encoder object.
    with open('data/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # Parameters.
    max_len = 20

    while True:
        print(f"{cr}You : {ss}", end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(f"{cb}Mira:{ss}", np.random.choice(i['responses']))

print(f"{cg}You can start your conversation with Mira. Type 'quit' to end the sessiomn.{ss}")
chat()