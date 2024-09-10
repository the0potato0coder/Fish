import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding

def build_discriminator(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_length))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(SimpleRNN(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
