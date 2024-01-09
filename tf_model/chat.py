import json
import numpy as np

import random
import pickle

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


with open('intents.json') as file:
    data = json.load(file)

# Define the ANSI escape characters for terminal output.
cr = "\033[91m"  # red
cg = "\033[92m"  # green
cb = "\033[94m"  # blue
cc = "\033[96m"  # cyan
sb = "\033[1m"   # bold
si = "\033[3m"   # italic
s0 = "\033[0m"   # reset


def predict_intent(text, model, tokenizer, lbl_encoder, max_len=20, threshold=0.3):
    # Preprocessing the input.
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)

    # Getting predictions.
    predictions = model.predict(padded_sequence)
    best_match = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence < threshold:
        return "fallback"
    else:
        return lbl_encoder.inverse_transform([best_match])[0]

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
        print(f"{cr}You : {s0}", end="")
        
        # Take the user input.
        inp = input()
        
        # Developement notes.
        if inp.lower() == "what next":
            print(f"{cb}Mira:{s0} Currently I'm in alpha version. Next steps on my developement process:",
                  f"{cc}{sb}1.{s0} Combining multiple inputs, introducing a conversation flow.",
                  f"{cc}{sb}2.{s0} Adding interactive results.",
                  f"{cc}{sb}3.{s0} Turkish language support.",
                  f"{cc}{sb}4.{s0} Boosting accuracy.",
                  f"{cc}{sb}5.{s0} Active learning from the live conversations.",
                  f"{cc}{sb}6.{s0} Expanding the knowledge base to necessary topics.",
                  f"{cc}{sb}7.{s0} Ability to listen and speak.",
                  sep="\n  ", end="\n\n"
            )
            continue
        
        # End the conversation if the user types 'quit'.
        if inp.lower() == "quit":
            break
        
        # Predict the intent.
        tag = predict_intent(inp, model, tokenizer, lbl_encoder, max_len)

        # Find the response.
        for i in data['intents']:
            if i['tag'] == tag:
                print(f"{cb}Mira:{s0}", random.choice(i['responses']))
                
                if tag == 'goodbye':
                    print(f" You can type {si}'quit'{s0} to end the conversation.")


if __name__ == "__main__":
    print(f"{cg}You can start your conversation with Mira. Type {si}'quit'{s0}{cg} to end the session.{s0}")
    chat()