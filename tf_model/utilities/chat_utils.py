import json
import pickle

from tensorflow.keras.models import load_model


# Constants
MODEL_NAME = "Mira"

# ANSI escape characters
CR = "\033[91m"  # red
CG = "\033[92m"  # green
CB = "\033[94m"  # blue
CC = "\033[96m"  # cyan
SB = "\033[1m"   # bold
SI = "\033[3m"   # italic
S0 = "\033[0m"   # reset


def load_model_and_data():
    # Load trained model.
    model = load_model('data/chat_model')

    # Load tokenizer object.
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load label encoder object.
    with open('data/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # Load intents data.
    with open('intents.json') as file:
        data = json.load(file)

    return model, tokenizer, lbl_encoder, data


def display_development_notes():
    print(f"{CB}{MODEL_NAME}:{S0} Currently I'm in alpha version. Next steps on my development process:",
          f"{CC}{SB}1.{S0} Combining multiple inputs, introducing a conversation flow.",
          f"{CC}{SB}2.{S0} Adding interactive results.",
          f"{CC}{SB}3.{S0} Turkish language support.",
          f"{CC}{SB}4.{S0} Boosting accuracy.",
          f"{CC}{SB}5.{S0} Active learning from the live conversations.",
          f"{CC}{SB}6.{S0} Expanding the knowledge base to necessary topics.",
          f"{CC}{SB}7.{S0} Ability to listen and speak.",
          sep="\n  ", end="\n\n"
        )


def print_welcome_message():
    print(f"{CB}{MODEL_NAME}:{S0} Hi, I'm {MODEL_NAME}. I'm here to help you with your questions about the company.")


def print_user_prompt():
    print(f"{CR}You : {S0}", end="")


def print_response(response, is_goodbye=False):
    print(f"{CB}{MODEL_NAME}:{S0}", response)

    if is_goodbye:
        print(f" You can type {SI}'quit'{S0} to end the conversation.")


def is_quit_command(user_input):
    return user_input.lower() == "quit"
