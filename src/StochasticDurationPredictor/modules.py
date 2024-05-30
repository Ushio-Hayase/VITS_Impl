import math
import tensorflow as tf
from tensorflow import keras

from utils import piecewise_rational_quadratic_transform

class ResBlock(keras.layers.Layer):
    def __init__(self, channels: int, kernel_size: int, n_layers : int, dropout = 0.1):
        super().__init__()

        self.drop = keras.layers.Dropout(dropout)

        self.n_layers = n_layers

        conv_sep = []
        conv_1x1 = []
        norms_1 = []
        norms_2 = []

        for i in range(n_layers):
            dilation = kernel_size ** i
            conv_sep.append(keras.layers.Conv1D(channels, kernel_size, groups=channels, dilation_rate=dilation, padding="same"))
            conv_1x1.append(keras.layers.Conv1D(channels, 1))
            norms_1.append(keras.layers.LayerNormalization())
            norms_2.append(keras.layers.LayerNormalization())

        self.conv_sep = keras.Sequential(conv_sep)
        self.conv_1x1 = keras.Sequential(conv_1x1)
        self.norms_1 = keras.Sequential(norms_1)
        self.norms_2 = keras.Sequential(norms_2)

    def call(self, inputs):
        x = inputs

        for i in range(self.n_layers):
            y = self.conv_sep[i]()
            y = self.norms_1[i][y]
            y = keras.activations.gelu(y)
            y = self.conv_1x1[i](y)
            y = self.norms_2[i](y)
            y = keras.activations.gelu(y)
            y = self.drop(y)
            x = x + y
        return x

    
class Log(keras.Layer):
    def call(self, inputs, training=True):
        x = inputs
        if training:
            y = tf.math.log(tf.maximum(x, 1e-5))
            logdet = keras.backend.sum(-y, 1)
            logdet = keras.backend.sum(y, axis=2)
            return y, logdet
        else:
            x = tf.exp(x)
            return x
        
class ElementwiseAffine(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()

        self.m = keras.Variable(keras.backend.zeros((channels, 1)))
        self.logs = keras.Variable(keras.backend.zeros((channels, 1)))

    def call(self, inputs, training = True):
        x = inputs
        
        if training:
            y = self.m + tf.exp(self.logs) * x
            logdet = keras.backend.sum(self.logs, 1)
            logdet(logdet, 2)
            return y, logdet
        else:
            x = (x - self.m) * tf.exp(-self.logs)
            return x
    

class ConvFlow(tf.keras.Model):
  def __init__(self, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
    super(ConvFlow, self).__init__()
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.num_bins = num_bins
    self.tail_bound = tail_bound

    self.pre = keras.layers.Conv1D(filter_channels, 1, input_shape=(None, self.half_channels))
    self.convs = ResBlock(filter_channels, kernel_size, n_layers, p_dropout=0.)
    self.proj = keras.layers.Conv1D(self.half_channels * (num_bins * 3 - 1), 1, kernel_initializer='zeros', bias_initializer='zeros')

  def call(self, inputs, training=True):
    x = inputs
    x0, x1 = tf.split(x, num_or_size_splits=2, axis=2)
    h = self.pre(x0)
    h = self.convs(h)
    h = self.proj(h)

    b, t, c = tf.shape(x0)
    h = tf.reshape(h, [b, t, -1, c])
    h = tf.transpose(h, [0, 1, 3, 2]) # [b, t, c, ?]

    unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_derivatives = h[..., 2 * self.num_bins:]

    x1, logabsdet = piecewise_rational_quadratic_transform(x1,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=not training,
        tails='linear',
        tail_bound=self.tail_bound
    )

    x = tf.concat([x0, x1], axis=2)
    logdet = tf.reduce_sum(logabsdet, axis=[1, 2])
    if not training:
        return x, logdet
    else:
        return x 

