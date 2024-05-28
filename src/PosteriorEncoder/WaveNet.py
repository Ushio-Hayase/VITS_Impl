import tensorflow as tf
from tensorflow import keras

class WaveNet(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dilated_conv = keras.layers.Conv1D(5, 5, dilation_rate=2) # Dilated convolution

        # Filter
        self.filter = keras.layers.Conv1D(3, 3)
        self.filter_activation = keras.layers.Activation(keras.activations.tanh) 

        # Gate 
        self.gate = keras.layers.Conv1D(3, 3)
        self.gate_activation = keras.layers.Activation(keras.activations.sigmoid) 

        # 전역적 조건 가중치
        self.global_filter = keras.layers.Conv1D(1,1)
        self.global_gate = keras.layers.Conv1D(1,1)

        self.multiply = keras.layers.Multiply() # Gate Activation Units

        self.conv = keras.layers.Conv1D(1,1) # Convolution Layer
    
    def call(self, inputs):
        data, h = inputs["input"], inputs["global"] # 데이터 분리
        data = self.dilated_conv(data) # Dliated convolution 실행 
        
        f = self.filter(data) + self.global_filter(h) # 전역적 조건과 필터 거친 데이터 더하기
        g = self.gate(data) + self.global_gate(h) # 전역적 조건과 게이트 거친 데이터 더하기

        tanh = self.filter_activation(f)
        sig = self.gate_activation(g)
        
        output = self.multiply(tanh, sig)
        
        output = self.conv(output)
        
        return data + output
        