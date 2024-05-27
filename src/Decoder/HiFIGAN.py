import tensorflow as tf
from tensorflow import keras


class ResBlock(keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, diliation=(1,3,5)):
        super().__init__()

        padding = "same"

        self.conv1 = keras.Sequential([
            keras.layers.Conv1D(channels, kernel_size, diliation_rate=diliation[0], padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(channels, kernel_size, diliation_rate=diliation[1], padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(channels, kernel_size, diliation_rate=diliation[2], padding=padding),
            keras.layers.BatchNormalization()
        ])

        self.conv2 = keras.Sequential([
            keras.layers.Conv1D(channels, kernel_size, diliation_rate=1, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(channels, kernel_size, diliation_rate=1, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(channels, kernel_size, diliation_rate=1, padding=padding),
            keras.layers.BatchNormalization()
        ])
 
    def call(self, inputs):
        x = inputs["input"]
        for c1, c2 in zip(self.conv1, self.conv2):
            xt = keras.activations.leaky_relu(x)
            xt = c1(xt)
            xt = keras.activations.leaky_relu(xt)
            xt = c2(xt)
            x = xt + x
        return x

class HifiGAN(keras.layers.Layer):
    def __init__(self, resblock_kernel_size : list, resblock_dilation_sizes, 
                 upsample_kernel_size : list, upsample_rates : list, upsample_initial_channel : int):
        super().__init__()

        self.num_kernels = len(resblock_kernel_size)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = keras.layers.Conv1D(upsample_initial_channel, 7, 1, padding="same")
        
        resblock = ResBlock

        ups = []
        for i ,(u, k) in enumerate(zip(upsample_rates, upsample_kernel_size)):
            ups.append(keras.layers.Conv1DTranspose(upsample_initial_channel//(2**(i+1)), k, u, padding="same"))

        self.ups = keras.Sequential(ups)

        resblocks = []
        for i in range(len(ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_size, resblock_dilation_sizes)):
                resblocks.append(resblock(ch, k, d))
        
        self.resblocks = keras.Sequential(resblocks)

        self.conv_post = keras.layers.Conv1D(1, 7, 1)

    def call(self, inputs):
        x = inputs["input"]

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = keras.activations.leaky_relu(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)

            x = xs / self.num_kernels

        x = keras.activations.leaky_relu(x)
        x = self.conv_post(x)
        x = tf.tanh(x)

        return x

