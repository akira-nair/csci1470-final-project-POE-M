import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
tf.keras.backend.clear_session()

"""
Credits:
https://youtu.be/ZMudJXhsUpY
This Tensorflow tutorial walked us through creating a basic
generative RNN model.
"""

def get_corpus(filepath = '../data/haiku.csv', new_line_tokens = False) -> list:
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
    if new_line_tokens:
        data = data.replace("/", " <newline> ", regex=True)
    # drop any row with missing values
    data = data.dropna()
    # the column processed_title contains all the haiku data
    # we apply the 'process raw haiku' function to that column
    data["processed_title"] = data["processed_title"].apply(lambda x: process_raw_haiku(x))
    if new_line_tokens:
        split_by_row = data["processed_title"].str.split(" newline ", n = 3, expand = True)
    else:
        split_by_row = data["processed_title"].str.split(" / ", n = 3, expand = True)
    data["row_1"] = split_by_row[0]
    data["row_2"] = split_by_row[1]
    data["row_3"] = split_by_row[2]
    if new_line_tokens:
        newline_tok = " <newline> "
        data["row_1"] = data["row_1"].apply(lambda x: x + newline_tok)
        data["row_2"] = data["row_2"].apply(lambda x: x + newline_tok)
        data["row_3"] = data["row_3"].apply(lambda x: x + newline_tok)
    # convert the pandas column to a list
    corpus = data["processed_title"].to_list()
    return corpus, data
     
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