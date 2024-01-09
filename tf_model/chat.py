import numpy as np
import random

from tensorflow.keras.preprocessing.sequence import pad_sequences

from utilities.chat_utils import load_model_and_data, display_development_notes, print_welcome_message, print_user_prompt, print_response, is_quit_command


# Constants
MAX_LEN = 20
THRESHOLD = 0.3


def predict_intent(text, model, tokenizer, lbl_encoder, max_len=MAX_LEN, threshold=THRESHOLD):
    # Preprocessing the input.
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Getting predictions.
    predictions = model.predict(padded_sequence)
    best_match = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence < threshold:
        return "fallback"
    else:
        return lbl_encoder.inverse_transform([best_match])[0]
    

def chat():
    model, tokenizer, lbl_encoder, data = load_model_and_data()

    while True:
        print_user_prompt()
        inp = input()

        if inp.lower() == "what next":
            display_development_notes()
            continue

        if is_quit_command(inp):
            break

        tag = predict_intent(inp, model, tokenizer, lbl_encoder, MAX_LEN)

        for i in data['intents']:
            if i['tag'] == tag:
                print_response(random.choice(i['responses']), is_goodbye=(tag == 'goodbye'))


if __name__ == "__main__":
    print_welcome_message()
    chat()