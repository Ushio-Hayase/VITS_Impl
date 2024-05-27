import math
import tensorflow as tf
from tensorflow import keras
from TransformerEncoder import TextEncoder
from NormalizingFlow import NormalizingFlow
from utils import pad_mask, monotonic_alignment_search
import monotonic_align


class PriorEncoder(keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int,\
                  num_layers : int, input_vocab_size : int,\
                    maximum_positional_encoding :int, affine_stack : int ,\
                        pad_id : int):
        super().__init__(name="PriorEncoder")

        self.pad_id = pad_id

        self.text_encoder = TextEncoder(num_layers, d_model, num_heads, dff,input_vocab_size, maximum_positional_encoding) # 트랜스포머 인코더

        self.normalizing_flow = [NormalizingFlow() for _ in range(affine_stack)] # Normalizing flow 

        self.mu = keras.layers.Dense(d_model, activation=None) # 평균 생성 
        self.logvar = keras.layers.Dense(d_model, activation=None) # 분산 생성 

    def call(self, inputs):
        z, text, condition = inputs["latentInput"], inputs["textInput"], inputs["global"]

        mask = pad_mask(text, self.pad_id) # 패딩 마스크 생성 

        h_text = self.text_encoder({"input": text,"mask": mask})

        text_mu : tf.Tensor = self.mu(h_text) # 평균 
        text_logvar = self.logvar(h_text) # 분산 

        for i in self.normalizing_flow:
            z = i(z, condition)

        context_vector = monotonic_alignment_search(z, text_mu, text_logvar)
        
        return context_vector

            




        
