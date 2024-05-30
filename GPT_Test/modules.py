import copy
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Conv1D

import commons
from commons import init_weights, get_padding
from transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1

class LayerNorm(tf.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = tf.Variable(tf.ones(channels), trainable=True)
        self.beta = tf.Variable(tf.zeros(channels), trainable=True)

    def __call__(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.keras.layers.LayerNormalization(epsilon=self.eps)(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x


class ConvReluNorm(tf.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = [layers.Conv1D(hidden_channels, kernel_size, padding='same') for _ in range(n_layers)]
        self.norm_layers = [LayerNorm(hidden_channels) for _ in range(n_layers)]
        self.relu_drop = tf.keras.Sequential([
            layers.ReLU(),
            layers.Dropout(p_dropout)
        ])
        self.proj = layers.Conv1D(out_channels, 1)
        self.proj.build((None, hidden_channels, out_channels))
        self.proj.kernel_initializer = tf.zeros_initializer()
        self.proj.bias_initializer = tf.zeros_initializer()

    def __call__(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(tf.Module):
    """
    Dialted and Depth-Separable Convolution
    """
    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = layers.Dropout(p_dropout)
        self.convs_sep = [layers.Conv1D(channels, kernel_size, padding='same', groups=channels, dilation_rate=kernel_size ** i) for i in range(n_layers)]
        self.convs_1x1 = [layers.Conv1D(channels, 1) for _ in range(n_layers)]
        self.norms_1 = [LayerNorm(channels) for _ in range(n_layers)]
        self.norms_2 = [LayerNorm(channels) for _ in range(n_layers)]

    def __call__(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = tf.nn.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = tf.nn.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(tf.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = [layers.Conv1D(2 * hidden_channels, kernel_size, dilation_rate=dilation_rate ** i, padding='same') for i in range(n_layers)]
        self.res_skip_layers = [layers.Conv1D(2 * hidden_channels if i < n_layers - 1 else hidden_channels, 1) for i in range(n_layers)]
        self.drop = layers.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = layers.Conv1D(2 * hidden_channels * n_layers, 1)
            self.cond_layer = tf.keras.utils.weight_norm(self.cond_layer)

    def __call__(self, x, x_mask, g=None, **kwargs):
        output = tf.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = tf.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class ResBlock1(tf.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = [tf.keras.layers.Conv1D(channels, kernel_size, padding='same', dilation_rate=d, kernel_initializer=init_weights) for d in dilation]
        self.convs2 = [tf.keras.layers.Conv1D(channels, kernel_size, padding='same', dilation_rate=1, kernel_initializer=init_weights) for _ in dilation]

    def __call__(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = tf.nn.leaky_relu(x, alpha=LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = tf.nn.leaky_relu(xt, alpha=LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class ResBlock2(tf.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = [tf.keras.layers.Conv1D(channels, kernel_size, padding='same', dilation_rate=d, kernel_initializer=init_weights) for d in dilation]

    def __call__(self, x, x_mask=None):
        for c in self.convs:
            xt = tf.nn.leaky_relu(x, alpha=LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class Log(tf.Module):
    def __call__(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = tf.math.log(tf.clip_by_value(x, clip_value_min=1e-5)) * x_mask
            logdet = tf.reduce_sum(-y, axis=[1, 2])
            return y, logdet
        else:
            x = tf.math.exp(x) * x_mask
            return x


class Flip(tf.Module):
    def __call__(self, x, *args, reverse=False, **kwargs):
        x = tf.reverse(x, axis=[1])
        if not reverse:
            logdet = tf.zeros_like(x[:, 0, 0])
            return x, logdet
        else:
            return x
        
class ElementwiseAffine(tf.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = tf.Variable(tf.zeros([channels, 1]), trainable=True)
        self.logs = tf.Variable(tf.zeros([channels, 1]), trainable=True)

    def __call__(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + tf.exp(self.logs) * x
            y = y * x_mask
            logdet = tf.keras.backend.sum(self.logs * x_mask, [1,2])
            return y, logdet
        else:
            x = (x - self.m) * tf.exp(-self.logs) * x_mask
            return x
        
class ResidualCouplingLayer(tf.keras.layers.Layer):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        super(ResidualCouplingLayer, self).__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = Conv1D(hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = Conv1D(self.half_channels * (2 - int(mean_only)), 1)

    def call(self, x, x_mask, g=None, reverse=False):
        x0, x1 = tf.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = tf.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = tf.zeros_like(m)

        if not reverse:
            x1 = m + x1 * tf.exp(logs) * x_mask
            x = tf.concat([x0, x1], 1)
            logdet = tf.reduce_sum(logs, [1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * tf.exp(-logs) * x_mask
            x = tf.concat([x0, x1], 1)
            return x


class ConvFlow(tf.keras.layers.Layer):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super(ConvFlow, self).__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = Conv1D(filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
        self.proj = Conv1D(self.half_channels * (num_bins * 3 - 1), 1)

    def call(self, x, x_mask, g=None, reverse=False):
        x0, x1 = tf.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = tf.reshape(h, [b, c, -1, t]) # [b, cx?, t] -> [b, c, t, ?]
        h = tf.transpose(h, [0, 1, 3, 2])

        unnormalized_widths = h[..., :self.num_bins] / tf.sqrt(float(self.filter_channels))
        unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / tf.sqrt(float(self.filter_channels))
        unnormalized_derivatives = h[..., 2 * self.num_bins:]

        x1, logabsdet = piecewise_rational_quadratic_transform(x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails='linear',
            tail_bound=self.tail_bound
        )

        x = tf.concat([x0, x1], 1) * x_mask
        logdet = tf.reduce_sum(logabsdet * x_mask, [1,2])
        if not reverse:
            return x, logdet
        else:
            return x

