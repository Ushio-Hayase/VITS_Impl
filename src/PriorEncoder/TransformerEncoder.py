import tensorflow as tf
from tensorflow import keras



class TextEncoder(keras.layers.Layer):
    """
    트랜스포머 인코더를 사용한 텍스트 인코더

    relative positional encoding 사용
    """
    def __init__(self, d_model: int, dff : int ,num_heads : int):
        super().__init__(name="TransformerEncoder")

        self.norm = keras.layers.LayerNormalization()

        # first floor
        self.MHA = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        # second floor
        self.FFNN1 = keras.layers.Dense(dff, activation="relu")
        self.FFNN2 = keras.layers.Dense(d_model)

    def call(self, inputs):
        data, mask = inputs["input"], inputs["mask"]

        attn = self.MHA(data, data, data, attention_mask=mask)
        attn = self.norm(data + attn)

        output = self.FFNN1(attn)
        output = self.FFNN2(output)
        return self.norm(attn + output)