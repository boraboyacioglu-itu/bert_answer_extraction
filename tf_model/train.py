import numpy as np
import tensorflow as tf

import json
import pickle

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


glove_id = 100
glove_data_loc = f'/Users/wndpzr/Downloads/glove.6B/glove.6B.{glove_id}d.txt'

# Open and read the knowledge base.
with open('intents.json') as f:
    data = json.load(f)

# Initialise the arrays.
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

# Number of different topics.
num_classes = len(labels)

# Initialise and fit the encoder to the training labels.
label_encoder = LabelEncoder()
label_encoder.fit(training_labels)
training_labels = label_encoder.transform(training_labels)

# Define the parameters.
vocab_size = 10000
embedding_dim = 100
max_len = 20
oov_token = "<OOV>"

# Tokenise the sentences.
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Pad the sequences.
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_len)

# Load GloVe vectors
embeddings_index = {}
with open(glove_data_loc, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Prepare embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Create embedding layer
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=20,
                            trainable=False)

training_labels = to_categorical(training_labels, num_classes=num_classes)

# Build the model.
model = Sequential()

# Add layers.
model.add(embedding_layer)
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model.
epochs = 500
history = model.fit(padded_sequences, training_labels, epochs=epochs)

# Data locations.
data_loc = 'data/'

# Save the trained model.
model.save(data_loc + 'chat_model')

# Save the fitted tokenizer.
with open(data_loc + 'tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the fitted label encoder.
with open(data_loc + 'label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)