#!/usr/bin/env python
'''
File        :   __metrics.py
Author      :   Akira Nair, Christine Jeong, Sedong Hwang
Description :   Perplexity metric
'''

import tensorflow as tf
class Perplexity(tf.keras.losses.SparseCategoricalCrossentropy):
    """
    Taken from RNN assignment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Perplexity"

    def call(self, *args, **kwargs):
        return tf.math.exp(tf.reduce_mean(super().call(*args, **kwargs)))