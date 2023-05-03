import tensorflow as tf
class Perplexity(tf.keras.losses.SparseCategoricalCrossentropy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = "Perplexity"

        def call(self, *args, **kwargs):
            return tf.math.exp(tf.reduce_mean(super().call(*args, **kwargs)))