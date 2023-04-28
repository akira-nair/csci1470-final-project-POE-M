import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_data(filename):
    data = pd.read_csv(filename)
    data = data.replace("/", " / ", regex=True)
    # print(data.head())
    # print(data["processed_title"])

    # data = data.rename(columns={'0': "line1", '1':"line2", '2':"line3"})
    # data["full_haiku"] = data["line1"] + " / " + data["line2"] + " / " + data["line3"]
    # data.replace(r"[!#$%&'()*+,-.:;<=>?@\[\\\]^_`{|}~]", '', regex=True, inplace=True)
    data = data.dropna()

    all_text = " ".join(data["processed_title"].to_list())
    tokens = tokenize(all_text)
    vocab_map = vectorize(tokens)

    data["vectorized"] = data["processed_title"].apply(lambda x: [vocab_map[t.strip()] for t in x.lower().split()])
    data = data[[a.count(4) <= 2 for a in data['vectorized']]]
    data = data[data['vectorized'].apply(lambda x: len(x) <= 19)]
    max_length = find_max_length(data["vectorized"].to_list())
    print(f"Max length found {max_length}")
    print(data["vectorized"])

    data['vectorized_pad'] = pad_sequences(vectorized_array, maxlen=max_length, padding='post')
    print(data['vectorized_pad'])

    

def tokenize(sentence: str):
    return list(sentence.lower().split())

def vectorize(tokens):
    vocab, index = {}, 1  # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for token in tokens:
        token = token.strip()
        if token not in vocab:
            vocab[token] = index
            index += 1
    return vocab

def find_max_length(vectorized_poems):
    max_length = 0
    for poem in vectorized_poems:
        # if len(poem)
        max_length = max(max_length, len(poem))   
    return max_length


def main():
    get_data("./data/haiku.csv")

if __name__ == "__main__":
    main()