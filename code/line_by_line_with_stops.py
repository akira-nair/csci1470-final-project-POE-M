#!/usr/bin/env python
'''
File        :   line_by_line_with_stops.py
Author      :   Akira Nair, Christine Jeong, Sedong Hwang
Description :   Model 3: Three LSTM models trained in parallel
                on each line of a haiku, but a stop token
                is added to each line of a haiku
'''

import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys, os
import line_by_line_no_stops
import preprocessing
from training import train_model
import joblib
tf.keras.backend.clear_session()

def main(args):
    # get model id, must be either 1, 2, or 3, corresponding
    # to which line of the haiku to train on
    model_id = int(args[0])
    # set output directory
    output = os.path.join("models", "three_architectures", "line_by_line_with_stops")
    if not os.path.exists(output):
        os.mkdir(output)
    # get the corpus and tokenizer
    # here, we set new_line_tokens to True
    # to get a special new-line symbol added
    # to the line by line columns in the data
    corpus, data = preprocessing.get_corpus(new_line_tokens=True)
    tokenizer = preprocessing.get_tokenizer(corpus)
    vocab_size = len(tokenizer.word_index) + 1
    # get x and y and train the model
    # we increase the maximum length for padding
    # because we incorporated the new line
    # symbol to each line
    if model_id == 1:
        x, y = line_by_line_no_stops.process_line1(corpus, data, 
                                                   len_structure=(6, 8, 6))
    elif model_id == 2:
        x, y = line_by_line_no_stops.process_line2(corpus, data, 
                                                   len_structure=(6, 8, 6))
    elif model_id == 3:
        x, y = line_by_line_no_stops.process_line3(corpus, data, 
                                                   len_structure=(6, 8, 6))
    else:
        raise ValueError("Not valid id. Must be 1, 2, or 3.")
    model, history = train_model(x, y, vocab_size=vocab_size)
    model.save(os.path.join(output, f"model_{model_id}"))
    joblib.dump(history, os.path.join(output, f"history_{model_id}"))


if __name__ == "__main__":
    main(sys.argv[1:])
