import json
import pickle

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Constants
MAX_LEN = 20
VOCAB_SIZE = 10000


def load_data() -> tuple:
    """ Load the data from the knowledge base.
    Returns:
        tuple: A tuple containing the training sentences, training labels,
            labels, and responses.
    """

    # Open and read the knowledge base.
    with open('intents.json') as f:
        data = json.load(f)

    # Initialize the arrays.
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    return training_sentences, training_labels, labels, responses


def preprocess_data(labels, training_sentences, training_labels) -> tuple:
    """ Preprocess the data.
    Args:
        training_sentences (list): A list of sentences.
        training_labels (list): A list of labels.
    Returns:
        tuple: A tuple containing the padded sequences, training labels,
            tokenizer, label encoder, and word index.
    """

    # Number of different topics.
    num_classes = len(labels)

    # Initialize and fit the encoder to the training labels.
    label_encoder = LabelEncoder()
    label_encoder.fit(training_labels)
    training_labels = label_encoder.transform(training_labels)

    # Tokenize the sentences.
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Pad the sequences.
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=MAX_LEN)

    training_labels = to_categorical(training_labels, num_classes=num_classes)

    return padded_sequences, training_labels, tokenizer, label_encoder, word_index


def save_model(model, tokenizer, label_encoder, data_loc):
    """ Save the model.
    Args:
        model (Sequential): The trained model.
        tokenizer (Tokenizer): The fitted tokenizer.
        label_encoder (LabelEncoder): The fitted label encoder.
        data_loc (str): The path to the data folder.
    """

    # Data locations.
    print(model.summary())

    # Save the trained model.
    model.save(data_loc + 'chat_model')

    # Save the fitted tokenizer.
    with open(data_loc + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the fitted label encoder.
    with open(data_loc + 'label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)