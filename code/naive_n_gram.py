#!/usr/bin/env python
'''
File        :   train_model.py
Author      :   Akira Nair, Sedong Hwang, Christine Jeong
Description :   Trains a language model to generate haikus
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, SimpleRNN, GRU, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys, os
import datetime
import joblib
from utils.metrics import Perplexity
import preprocessing
from training import train_model
import shutil
sns.set(rc={"figure.dpi": 300})
tf.keras.backend.clear_session()

"""
Credits:
https://youtu.be/ZMudJXhsUpY
This Tensorflow tutorial walked us through creating a basic
generative RNN model.
"""

def get_x_y_sequences(corpus: list, vocab_size: int, tokenizer: tf.keras.preprocessing.text.Tokenizer, max_length: int = None) -> tuple:
    """
    Get x and y sequences. To train a RNN generator that can adapt to 
    varying lengths of context sequences, we take each haiku and append a 
    context sequence comprised of the first n words of each haiku, for 
    all n in the range of the length of haiku. The model will be trained
    to predict context sequences' final words. For example, consider an element
    in the corpus: "The boy eats an apple." All the following samples are added:
    x -> y
    [The] -> boy
    [The, boy] -> eats
    [The, boy, eats] -> an
    [The boy, eats, an] -> apple
    Of course, instead of strings, the words are encoded to numbers using the tokenizer.

    Args:
        corpus (list): List of haikus
        vocab_size (int): The number of words in the vocab
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer fit to corpus
        max_length (max_length, optional): Pad sequences to this length. If None, calculates max length across input sequences. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple x and y, represented with numpy arrays
    """
    sequences = []
    for haiku in corpus:
        # convert haiku text to a numerical vector using tokenizer
        vectorized_haiku = tokenizer.texts_to_sequences([haiku])[0]
        # loop through each possible context length n, starting with 1 to avoid 
        # empty sequences
        for n in range(1, len(vectorized_haiku)):
            # get haiku up to that length and add to sequences list
            n_gram_seq = vectorized_haiku[:n+1]
            sequences.append(n_gram_seq)
    if max_length is None:
        max_sequence_len = max([len(x) for x in sequences])
    else:
        max_sequence_len = max_length
    # pad sequences with 0 to desired length
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
    # x is all tokens except last one, y is the last token in the sequence
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def generate_poem(model, tokenizer, starter_poem, max_len, length = 20):
    for _ in range(length):
        token_list = tokenizer.texts_to_sequences([starter_poem])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        starter_poem += " " + output_word
    return starter_poem

def plot_convergence(history):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
    sns.lineplot(history.history['loss'], ax = ax[0])
    sns.lineplot(history.history['accuracy'], ax = ax[1])
    fig.savefig("models/convergence.png")

def main(args):
    # if os.path.exists(output):
    #     shutil.rmdir(output)
    output = os.path.join("models", "three_architectures", "naive_n_gram")
    if not os.path.exists(output):
        os.mkdir(output)
    # ----
    corpus, data = preprocessing.get_corpus()
    tokenizer = preprocessing.get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 17
    # ---
    x, y = get_x_y_sequences(corpus = corpus, vocab_size = vocab_size, tokenizer = tokenizer, max_length=max_length)
    model, history = train_model(x, y, vocab_size=vocab_size)
    model.save(os.path.join(output, "model"))
    joblib.dump(history, os.path.join(output, f"history"))


if __name__ == "__main__":
    main(sys.argv[1:])
