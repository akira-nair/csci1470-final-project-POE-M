import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, SimpleRNN, GRU, Dropout, Bidirectional
from metrics import Perplexity

def train_model(x: np.ndarray, y: np.ndarray, vocab_size:int , n_epochs:int=200, embedding_size:int=50, learning_rate:float=0.001, batch_size:float=256, hidden_dim:int=128, lstm_units:int = 10)->tuple:
    """
    Trains an LSTM RNN model

    Args:
        x (np.ndarray): x
        y (np.ndarray): y
        vocab_size (int): size of vocab
        n_epochs (int, optional): number of epochs. Defaults to 50.
        embedding_size (int, optional): dimension of embedding. Defaults to 50.
        learning_rate (float, optional): learning rate for training model. Defaults to 0.01.
        batch_size (float, optional): batch size for training model. Defaults to 512.
        hidden_dim (int, optional): hidden dimension of model's dense layer. Defaults to 128.
        lstm_units (int, optional): number of LSTM units to use. Defaults to 10.
        output (str, optional): output directory to save model. Defaults to None.

    Returns:
        tuple[tf.keras.models.Sequential, tf.keras.callbacks.History]: A tuple of model and its training history
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    # Construct model
    model = Sequential([
        # Embed input sequence
        Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=x.shape[1]),
        # Using bidirectional LSTM units
        Bidirectional(LSTM(lstm_units)),
        Dense(hidden_dim, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    # Set an optimizer for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', Perplexity()])
    # Train the model
    history = model.fit(tf.convert_to_tensor(x), tf.convert_to_tensor(y), batch_size = batch_size, epochs=n_epochs, verbose=2)
    return model, history