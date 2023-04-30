import tensorflow as tf
import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, SimpleRNN, GRU, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys, os
sns.set(rc={"figure.dpi": 300})
tf.keras.backend.clear_session()

def get_corpus(filepath = '../data/haiku.csv') -> list:
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
    # convert the pandas column to a list
    corpus = data["processed_title"].to_list()
    return corpus
     
def get_tokenizer(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    
    return tokenizer

def get_x_y_sequences(corpus, vocab_size, tokenizer, max_length = None):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    if max_length is None:
        max_sequence_len = max([len(x) for x in input_sequences])
    else:
        max_sequence_len = max_length
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    x, y = input_sequences[:,:-1],input_sequences[:,-1]
    return x, y

def train_model(x, y, vocab_size, n_epochs=50, embedding_size=50, learning_rate=0.01, batch_size=512, hidden_dim=128):
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=x.shape[1]-1),
        Bidirectional(LSTM(4)),
        Dense(hidden_dim, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(tf.convert_to_tensor(x), tf.convert_to_tensor(y), batch_size = batch_size, epochs=n_epochs, verbose=1)
    model.save("lstm")
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
    print(starter_poem)

def plot_convergence(history):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
    sns.lineplot(history.history['loss'], ax = ax[0])
    sns.lineplot(history.history['accuracy'], ax = ax[1])
    fig.savefig("convergence.png")

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--learning_rate', default=0.001)
    args = parser.parse_args(args)
    # ----
    corpus = get_corpus()
    tokenizer = get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    x, y = get_x_y_sequences(corpus = corpus, vocab_size = vocab_size, tokenizer = tokenizer, max_length=25)

    model, history = train_model(x, y, vocab_size=vocab_size, n_epochs = args.epochs, learning_rate = args.learning_rate, batch_size=args.batch_size)

if __name__ == "__main__":
    main(sys.argv)
