import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

from utilities.train_utils import load_data, preprocess_data, save_model

# Constants
EMBEDDING_DIM = 100
MAX_LEN = 20
RANDOM_STATE = 6472  # "MIRA" in T9

EPOCHS = 5000
LEARNING_RATE = 0.001

PATIENCE  = 500


def build_model(embedding_matrix, num_classes) -> Sequential:
    """ Build the model.
    Args:
        embedding_matrix (np.array): An array containing the GloVe vectors.
        num_classes (int): The number of different topics.
    Returns:
        Sequential: The built model.
    """

    # Create embedding layer
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_LEN,
                                trainable=False)

    # Build the model.
    model = Sequential()

    # Add layers.
    model.add(embedding_layer)
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    # Add an Adam optimizer.
    optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(model, train_data, val_data) -> tuple:
    """ Train the model.
    Args:
        model (Sequential): The built model.
        train_data (tuple): A tuple containing the padded sequences and training labels.
        val_data (tuple): A tuple containing the padded sequences and validation labels.
    Returns:
        tuple: A tuple containing the trained model and the history of the training process.
    """

    # Early stopping.
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    # Fit the model.
    history = model.fit(*train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[early_stop])

    return model, history


if __name__ == "__main__":
    training_sentences, training_labels, labels, responses = load_data()
    padded_sequences, training_labels, tokenizer, label_encoder, word_index = preprocess_data(labels, training_sentences, training_labels)

    # Split data into training and validation sets.
    train_data, val_data = train_test_split(
        (padded_sequences, training_labels),
        test_size=0.1, random_state=RANDOM_STATE, shuffle=True
    )

    # Load GloVe vectors
    embeddings_index = {}
    glove_data_loc = f'/Users/wndpzr/Downloads/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt'
    with open(glove_data_loc, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Build, train and save the model.
    model = build_model(embedding_matrix, len(labels))
    trained_model, history = train_model(model, train_data, val_data)
    save_model(trained_model, tokenizer, label_encoder, 'data/')
