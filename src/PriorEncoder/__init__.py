import tensorflow as tf
from tensorflow import keras
from TransformerEncoder import TextEncoder
from NormalizingFlow import NormalizingFlow
from utils import pad_mask, relative_position_encoding, monotonic_alignment_search
import monotonic_align


class PriorEncoder(keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, num_heads: int, pad_id : int):
        super().__init__(name="PriorEncoder")

        self.pad_id = pad_id

        self.text_encoder = TextEncoder(d_model, dff, num_heads) # 트랜스포머 인코더

        self.normalizing_flow = NormalizingFlow() # Normalizing flow 

        self.mu = keras.layers.Dense(1, activation=None) # 평균 생성 
        self.logvar = keras.layers.Dense(1, activation=None) # 분산 생성 

    def call(self, inputs):
        z, text, condition = inputs["latentInput"], inputs["textInput"], inputs["global"]

        mask = pad_mask(text, self.pad_id) # 패딩 마스크 생성 
        text = text + relative_position_encoding(text) # TODO

        h_text = self.text_encoder(text, mask)

        text_mu = self.mu(h_text) # 평균 
        text_logvar = self.logvar(h_text) # 분산 

        latent = self.normalizing_flow(z, condition)

        attn = monotonic_alignment_search()




        
