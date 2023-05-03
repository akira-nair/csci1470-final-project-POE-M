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
import line_by_line
import preprocessing
from training import train_model
import shutil
import joblib

sns.set(rc={"figure.dpi": 300})
tf.keras.backend.clear_session()

"""
Credits:
https://youtu.be/ZMudJXhsUpY
This Tensorflow tutorial walked us through creating a basic
generative RNN model.
"""

def generate_poem_using_stops(model1, model2, model3, tokenizer, starter_poem):
    line1 = starter_poem
    output_word = ""
    while output_word != "newline":
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
    output_word = ""
    while output_word != "newline":
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
    output_word = ""
    while output_word != "newline":
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
    fulllines = "\n".join([line1.strip(), line2.strip(), line3.strip()])
    fulllines = fulllines.replace("newline", "")
    return fulllines


def main(args):
    model_id = int(args[0])
    output = os.path.join("models", "three_architectures", "line_by_line_with_stops")
    if not os.path.exists(output):
        os.mkdir(output)
    corpus, data = preprocessing.get_corpus(new_line_tokens=True)
    if model_id == 1:
        x, y = line_by_line.process_line1(corpus, data)
    elif model_id == 2:
        x, y = line_by_line.process_line2(corpus, data)
    elif model_id == 3:
        x, y = line_by_line.process_line3(corpus, data)
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
