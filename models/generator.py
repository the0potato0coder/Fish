import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


def build_generator(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(max_length, activation='sigmoid'))
    return model
