#!/usr/bin/env python
'''
File        :   line_by_line_no_stops.py
Author      :   Akira Nair, Christine Jeong, Sedong Hwang
Description :   Model 2: Three LSTM models trained in parallel
                on each line of a haiku
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys, os
import joblib
import preprocessing
from training import train_model
tf.keras.backend.clear_session()

def process_line1(corpus: list, data: pd.DataFrame, len_structure: tuple = (5, 7, 5)) -> tuple:
    """
    Processes the first line of all haikus to train model 1

    Args:
        corpus (list): corpus of all words
        data (pd.DataFrame): dataframe that has column
        "row_1" consisting of the first line of all haikus
        len_structure (tuple): a tuple of three integers, corresponding
        to the max lengths for three models. Defaults to (5, 7, 5).

    Returns:
        tuple: x and y to train model 1
    """
    # get line 1 data as a list
    l1_haikus = data["row_1"].to_list()
    tokenizer = preprocessing.get_tokenizer(corpus)
    input_seqs = []
    # construct n-grams, similar to the naive n-gram
    # preprocessing pipeline
    for l1 in l1_haikus:
        vectorized_haiku = tokenizer.texts_to_sequences([l1])[0]
        for n in range(1, len(vectorized_haiku)):
            n_gram_seq = vectorized_haiku[:n+1]
            input_seqs.append(n_gram_seq)
    # to adopt the haiku syllabic structure (5 + 7 + 5),
    # we pad to max length of 5, 
    # which occurs if each word is one syllable
    sequences = np.array(pad_sequences(input_seqs, maxlen=len_structure[0], padding='pre'))
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def process_line2(corpus: list, data: pd.DataFrame, len_structure: tuple = (5, 7, 5)) -> tuple:
    """
    Processes the second line of all haikus to train model 2

    Args:
        corpus (list): corpus of all words
        data (pd.DataFrame): dataframe that has column
        "row_2" consisting of the second line of all haikus
        len_structure (tuple): a tuple of three integers, corresponding
        to the max lengths for three models. Defaults to (5, 7, 5).

    Returns:
        tuple: x and y to train model 2
    """
    # get line 1 and line 2 data as lists
    l1_haikus = data["row_1"].to_list()
    l2_haikus = data["row_2"].to_list()
    tokenizer = preprocessing.get_tokenizer(corpus)
    input_sequences = []
    # we want to incorporate the full l1 sequences
    # in the context of all the l2 n-grams
    # we loop through all the l1 sequences
    for i, l1 in enumerate(l1_haikus):
        # get the corresponding l2
        l2 = l2_haikus[i]
        # convert to numeric sequences
        l1_vectorized = tokenizer.texts_to_sequences([l1])[0]
        l2_vectorized = tokenizer.texts_to_sequences([l2])[0]
        # construct n-grams for l2
        for n in range(0, len(l2_vectorized)):
            n_gram_seq = l2_vectorized[: n + 1]
            # merge l1 to l2's n-gram
            input_sequences.append(l1_vectorized + n_gram_seq)
    # to adopt the haiku syllabic structure (5 + 7 + 5),
    # we pad to max length of 12, 
    # which occurs if each word is one syllable
    sequences = np.array(pad_sequences(input_sequences, len_structure[0]+len_structure[1], padding='pre'))
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def process_line3(corpus: list, data: pd.DataFrame, len_structure: tuple = (5, 7, 5)) -> tuple:
    """
    Processes the third line of all haikus to train model 3

    Args:
        corpus (list): corpus of all words
        data (pd.DataFrame): dataframe that has column
        "row_3" consisting of the third line of all haikus
        len_structure (tuple): a tuple of three integers, corresponding
        to the max lengths for three models. Defaults to (5, 7, 5).

    Returns:
        tuple: x and y to train model 3
    """
    # construct a list of concatenations of lines 1 and 2 across all haikus
    l1_l2_haikus = [i.strip()+" "+j.strip() for i,j in \
                    zip(data["row_1"].to_list(),data["row_2"].to_list())]
    # get line 3 data as list
    l3_haikus = data["row_3"].to_list()
    tokenizer = preprocessing.get_tokenizer(corpus)
    input_sequences = []
    for i, l1_l2 in enumerate(l1_l2_haikus):
        # for each l1+l2 concatenated lines, get the corresponding l3
        l3 = l3_haikus[i]
        # convert to numeric sequences
        l1_l2_vectorized = tokenizer.texts_to_sequences([l1_l2])[0]
        l3_vectorized = tokenizer.texts_to_sequences([l3])[0]
        # construct n-grams for l3
        for n in range(0, len(l3_vectorized)):
            n_gram_seq = l3_vectorized[: n + 1]
            # merge l1+l2 to l3's n-gram
            input_sequences.append(l1_l2_vectorized + n_gram_seq)
    # to adopt the haiku syllabic structure (5 + 7 + 5),
    # we pad to max length of 17, 
    # which occurs if each word is one syllable
    sequences = np.array(pad_sequences(input_sequences, len_structure[0]+len_structure[1]+len_structure[2], padding='pre'))
    x, y = sequences[:,:-1],sequences[:,-1]
    return x, y

def main(args):
    # get model id, must be either 1, 2, or 3, corresponding
    # to which line of the haiku to train on
    model_id = int(args[0])
    # set output directory
    output = os.path.join("models", "three_architectures", "line_by_line")
    if not os.path.exists(output):
        os.mkdir(output)
    # get the corpus and tokenizer
    corpus, data = preprocessing.get_corpus()
    tokenizer = preprocessing.get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    # get x and y and train the model
    if model_id == 1:
        x, y = process_line1(corpus, data)
    elif model_id == 2:
        x, y = process_line2(corpus, data)
    elif model_id == 3:
        x, y = process_line3(corpus, data)
    else:
        raise ValueError("Not valid id. Must be 1, 2, or 3.")
    model, history = train_model(x, y, vocab_size=vocab_size)
    model.save(os.path.join(output, f"model_{model_id}"))
    joblib.dump(history, os.path.join(output, f"history_{model_id}"))

if __name__ == "__main__":
    main(sys.argv[1:])
