import tensorflow as tf
from tensorflow import keras

class PosteriorEncoder(keras.layers.Layer):
    def __init__(self):
        super().__init__(name="PoseriorEncoder")
        self.dilated_conv = keras.layers.Conv1D(5, 5, dilation_rate=2) # Dilated convolution

        # Filter
        self.filter = keras.layers.Conv1D(3, 3)
        self.filter_activation = keras.layers.Activation(keras.activations.tanh) 

        # Gate 
        self.gate = keras.layers.Conv1D(3, 3)
        self.gate_activation = keras.layers.Activation(keras.activations.sigmoid) 

        # Projection Layer
        self.filter_linear = keras.layers.Dense(2, activation=None)
        self.gate_linear = keras.layers.Dense(2, activation=None)

        self.multiply = keras.layers.Multiply() # Gate Activation Units

        self.conv = keras.layers.Conv1D(1,1) # Convolution Layer
    
    def call(self, inputs):
        data, h = inputs["input"], inputs["global"]
        data = self.dilated_conv(data)
        
        f = self.filter(data)
        g = self.gate(data)

        tanh = self.filter_activation(f + self.filter_linear(h))
        sig = self.gate_linear(g + self.gate_linear(g))
        
        output = self.multiply(tanh, sig)
        
        output = self.conv(output)
        
        return data + output
        