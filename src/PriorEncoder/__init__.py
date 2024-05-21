import tensorflow as tf
from tensorflow import keras
from TransformerEncoder import TextEncoder


class PriorEncoder(keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int):
        super().__init__(name="PriorEncoder")
        self.text_encoder = TextEncoder(d_model, dff, num_heads)

        self.projection_layer = keras.layers.Dense(2, activation=None)

        
