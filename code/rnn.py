import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace

class MyRNN(tf.keras.Model):
    
    def __init__(self, vocab_size, rnn_size=256, embed_size=128):
        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        # TODO: input layers here

    def call(self, inputs):
        pass



def main():
    data = get_data("./data/all_haiku.csv")




if __name__ == '__main__':
    main()


