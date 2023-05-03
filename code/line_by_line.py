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
from metrics import Perplexity
import preprocessing
import shutil
from training import train_model
sns.set(rc={"figure.dpi": 300})
tf.keras.backend.clear_session()

"""
Credits:
https://youtu.be/ZMudJXhsUpY
This Tensorflow tutorial walked us through creating a basic
generative RNN model.
"""

def process_line1(corpus, data):
    l1_haikus = data["row_1"].to_list()
    tokenizer = preprocessing.get_tokenizer(corpus)
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
    tokenizer = preprocessing.get_tokenizer(corpus)
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
    tokenizer = preprocessing.get_tokenizer(corpus)
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

def main(args):
    model_id = int(args[0])
    output = os.path.join("models", "three_architectures", "line_by_line")
    if not os.path.exists(output):
        os.mkdir(output)
    # # if not os.path.exists(output):
    # os.mkdir(output)
    corpus, data = preprocessing.get_corpus()
    if model_id == 1:
        x, y = process_line1(corpus, data)
    elif model_id == 2:
        x, y = process_line2(corpus, data)
    elif model_id == 3:
        x, y = process_line3(corpus, data)
    else:
        print("not valid id.")
        return
    tokenizer = preprocessing.get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    model, history = train_model(x, y, vocab_size=vocab_size)
    model.save(os.path.join(output, f"model_{model_id}"))
    joblib.dump(history, os.path.join(output, f"history_{model_id}"))

if __name__ == "__main__":
    main(sys.argv[1:])
