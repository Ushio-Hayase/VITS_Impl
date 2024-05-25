import tensorflow as tf
from tensorflow import keras
from ResidualBlock import ResidualBlock

class NormalizingFlow(keras.layers.Layer):
    def __init__(self):
        super().__init__(name="normalizing_flow")
        self.residual_1 = ResidualBlock()
        self.residual_2 = ResidualBlock()

    def call(self, inputs):
        data, condition = inputs["input"], inputs["global"] # 들어온 데이터 분리
        front, end = tf.split(data, num_or_size_splits=2, axis=-1) # affine transformation을 이용하기 위해 데이터를 반으로 나눔

        front = self.residual_1({"input": front,"global": condition}) # 앞쪽 데이터 먼저 Residual Block에 넣기 

        b1 = front
        s1 = tf.math.exp(front)

        output1 = tf.math.multiply(s1, end) + b1 # 첫번째 계산 결과

        end = self.residual_2({"input": end,"global": condition}) # 뒤쪽 데이터 Residual Block에 넣기 

        b2 = end
        s2 = tf.math.exp(end)

        output2 = tf.math.multiply(s2, front) + b2 # 두번째 계산 결과 

        return tf.concat(output1, output2, axis=-1) # 합치기
        