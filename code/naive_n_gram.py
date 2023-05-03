#!/usr/bin/env python
'''
File        :   train_model.py
Author      :   Akira Nair, Sedong Hwang, Christine Jeong
Description :   Trains a language model to generate haikus
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys, os
import joblib
import preprocessing
from training import train_model
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

def main(args):
    # set output directory
    output = os.path.join("models", "three_architectures", "naive_n_gram")
    if not os.path.exists(output):
        os.mkdir(output)
    # get the corpus and tokenizer
    corpus, _ = preprocessing.get_corpus()
    tokenizer = preprocessing.get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 17
    # get x and y and train the model
    x, y = get_x_y_sequences(corpus = corpus, vocab_size = vocab_size, tokenizer = tokenizer, max_length=max_length)
    model, history = train_model(x, y, vocab_size=vocab_size)
    model.save(os.path.join(output, "model"))
    joblib.dump(history, os.path.join(output, f"history"))

if __name__ == "__main__":
    main(sys.argv[1:])
