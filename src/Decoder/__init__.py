import tensorflow as tf
from tensorflow import keras
from HiFIGAN import HifiGAN

class Decoder(keras.layers.Layer):
    def __init__(self, resblock_kernel_size : list, resblock_dilation_sizes, 
                 upsample_kernel_size : list, upsample_rates : list, upsample_initial_channel : int):
        super().__init__(name="decoder")
        
        self.decoder = HifiGAN(resblock_kernel_size, resblock_dilation_sizes, upsample_kernel_size, upsample_rates, upsample_initial_channel)

    def call(self, inputs):
        x = inputs["input"]
        return self.decoder(x)


