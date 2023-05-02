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
sns.set(rc={"figure.dpi": 300})
tf.keras.backend.clear_session()

"""
Credits:
https://youtu.be/ZMudJXhsUpY
This Tensorflow tutorial walked us through creating a basic
generative RNN model.
"""

def get_corpus(filepath = '../data/haiku.csv') -> tuple:
    """
    Reads haiku csv data and constructs a list of strings, representing processed haikus
    This list of strings forms the corpus of the dataset, through which we derive our vocabulary

    Args:
        filepath (str, optional): File path to data. Defaults to '../data/haiku.csv'.

    Returns:
        list: List of strings, each string being a haiku
    """
    data = pd.read_csv(filepath)
    # we use the '/' to represent new lines within each haiku
    # we want to tokenize the slash, so we add spaces
    data = data.replace("/", " / ", regex=True)
    
    # drop any row with missing values
    data = data.dropna()
    # the column processed_title contains all the haiku data
    # we apply the 'process raw haiku' function to that column
    data["processed_title"] = data["processed_title"].apply(lambda x: process_raw_haiku(x))
    split_by_row = data["processed_title"].str.split(" / ", n = 3, expand = True)
    data["row_1"] = split_by_row[0]
    data["row_2"] = split_by_row[1]
    data["row_3"] = split_by_row[2]
    # convert the pandas column to a list
    corpus = data["processed_title"].to_list()
    return corpus, data

def process_line1(corpus, data):
    l1_haikus = data["row_1"].to_list()
    tokenizer = get_tokenizer(corpus)
    input_seqs = []
    for l1 in l1_haikus:
        vectorized_haiku = tokenizer.texts_to_sequences([l1])[0]
        for n in range(1, len(vectorized_haiku)):
            n_gram_seq = vectorized_haiku[:n+1]
            input_seqs.append(n_gram_seq)
    # print(input_seqs)[:5]
    sequences = np.array(pad_sequences(input_seqs, maxlen=5, padding='pre'))
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def process_line2(corpus, data):
    l1_haikus = data["row_1"].to_list()
    l2_haikus = data["row_2"].to_list()
    tokenizer = get_tokenizer(corpus)
    input_sequences = []
    for i, l1 in enumerate(l1_haikus):
        l2 = l2_haikus[i]
        l1_vectorized = tokenizer.texts_to_sequences([l1])[0]
        l2_vectorized = tokenizer.texts_to_sequences([l2])[0]
        for n in range(0, len(l2_vectorized)):
            n_gram_seq = l2_vectorized[: n + 1]
            input_sequences.append(l1_vectorized + n_gram_seq)
    sequences = np.array(pad_sequences(input_sequences, maxlen=12, padding='pre'))
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def process_line3(corpus, data):
    l1_l2_haikus = [i.strip()+" "+j.strip() for i,j in zip(data["row_1"].to_list(),data["row_2"].to_list())]
    l3_haikus = data["row_3"].to_list()
    tokenizer = get_tokenizer(corpus)
    input_sequences = []
    for i, l1_l2 in enumerate(l1_l2_haikus):
        l3 = l3_haikus[i]
        l1_l2_vectorized = tokenizer.texts_to_sequences([l1_l2])[0]
        l3_vectorized = tokenizer.texts_to_sequences([l3])[0]
        for n in range(0, len(l3_vectorized)):
            n_gram_seq = l3_vectorized[: n + 1]
            input_sequences.append(l1_l2_vectorized + n_gram_seq)
    sequences = np.array(pad_sequences(input_sequences, maxlen=17, padding='pre'))
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def get_tokenizer(corpus: list) -> tf.keras.preprocessing.text.Tokenizer:
    """
    Creates a tokenizer object that is fit to the corpus. The tokenizer 
    creates a mapping from words to unique integers and vice versa.

    Args:
        corpus (list): A list of haikus that form the corpus

    Returns:
        tf.keras.preprocessing.text.Tokenizer: A keras tokenizer
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    return tokenizer

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

def train_model(x: np.ndarray, y: np.ndarray, vocab_size:int , n_epochs:int=50, embedding_size:int=50, learning_rate:float=0.01, batch_size:float=512, hidden_dim:int=128, lstm_units:int = 10)->tuple:
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
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Train the model
    history = model.fit(tf.convert_to_tensor(x), tf.convert_to_tensor(y), batch_size = batch_size, epochs=n_epochs, verbose=1)
    model.save("models/lstm")
    return model, history

def process_raw_haiku(haiku):
    tokens = tokenize(haiku)
    return " ".join(tokens)

def tokenize(sentence: str):
    tokens = sentence.lower().split()
    tokens = [process_token(t) for t in tokens]
    return tokens

def process_token(token: str):
    if token.strip() == "/":
        return token
    return re.sub(r'[^\w\s]', '', token.strip())

def syllable_count(word):
    # "source https://stackoverflow.com/questions/46759492/syllable-count-in-python"
    if word.strip() == "":
        return 0
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count

def generate_poem(model1, model2, model3, tokenizer, starter_poem):
    line1 = starter_poem
    while syllable_count(line1) < 5:
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=5-1, padding='pre')
        probs = model1.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line1 += " " + output_word
    line2 = ""
    while syllable_count(line2) < 7:
        token_list = tokenizer.texts_to_sequences([line1 + line2])[0]
        token_list = pad_sequences([token_list], maxlen=12-1, padding='pre')
        probs = model2.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line2 += " " + output_word
    line3 = ""
    while syllable_count(line3) < 5:
        token_list = tokenizer.texts_to_sequences([line1 + line2 + line3])[0]
        token_list = pad_sequences([token_list], maxlen=17-1, padding='pre')
        probs = model3.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line3 += " " + output_word
    return "\n".join([line1.strip(), line2.strip(), line3.strip()])

# def generate_poem(model, tokenizer, starter_poem, max_len, length = 20):
#     for _ in range(length):
#         token_list = tokenizer.texts_to_sequences([starter_poem])[0]
#         token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
#         predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
#         output_word = ""
#         for word, index in tokenizer.word_index.items():
#             if index == predicted:
#                 output_word = word
#                 break
#         starter_poem += " " + output_word
#     print(starter_poem)

def plot_convergence(history):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
    sns.lineplot(history.history['loss'], ax = ax[0])
    sns.lineplot(history.history['accuracy'], ax = ax[1])
    fig.savefig("models/convergence.png")

def main(args):
    model_id = int(args[0])
    corpus, data = get_corpus()
    if model_id == 1:
        x, y = process_line1(corpus, data)
    elif model_id == 2:
        x, y = process_line2(corpus, data)
    elif model_id == 3:
        x, y = process_line3(corpus, data)
    else:
        raise("not valid id.")
    tokenizer = get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    model, history = train_model(x, y, \
        vocab_size=vocab_size, 
        n_epochs = 50, 
        learning_rate=0.001, 
        batch_size = 256)
    model.save(f"model_{model_id}")


if __name__ == "__main__":
    main(sys.argv[1:])
