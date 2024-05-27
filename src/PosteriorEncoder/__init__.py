import tensorflow as tf
from tensorflow import keras
from ..ResidualBlock import ResidualBlock

class PosteriorEncoder(keras.layers.Layer):
    def __init__(self, latent_size):
        super().__init__(name="PoseriorEncoder")
        self.residual_block = ResidualBlock()
        self.mu = keras.layers.Dense(latent_size, activation=None) # 평균 생성 레이어
        self.sigma = keras.layers.Dense(latent_size, activation=None) # 분산 생성 레이어
    
    def call(self, inputs):
        data, condition = inputs["input"], inputs["global"] # 들어온 데이터 분리

        output = self.residual_block({"input": data,"global": condition}) # 데이터 넣기

        mu, sigma = self.mu(output), self.sigma(output) # 평균, 분산 각각 구하기
        return mu, sigma
        