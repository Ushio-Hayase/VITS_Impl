import math
import numpy as np
import tensorflow as tf

def init_weights(layer, mean=0.0, std=0.01):
    if isinstance(layer, tf.keras.layers.Conv1D):
        layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=std)

def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (tf.exp(2. * logs_p) + tf.square((m_p - m_q))) * tf.exp(-2. * logs_q)
    return kl

def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = tf.random.uniform(shape) * 0.99998 + 0.00001
    return -tf.math.log(-tf.math.log(uniform_samples))

def rand_gumbel_like(x):
    g = rand_gumbel(x.shape)
    return g

def shift_1d(x):
    pad_shape = convert_pad_shape([[0, 0], [0, 0], [1, 0]])
    x = tf.pad(x, pad_shape)[:, :, :-1]
    return x

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = tf.reduce_max(length)
    x = tf.range(max_length, dtype=length.dtype)
    return x[tf.newaxis, :] < length[:, tf.newaxis]

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = tf.cumsum(duration, axis=-1, reverse=False)

    mask_flat = tf.reshape(mask, (b * t_y, t_x))
    cum_duration_flat = tf.reshape(cum_duration, (b * t_x,))
    path = sequence_mask(cum_duration_flat, tf.shape(mask_flat)[1])
    path = tf.reshape(path, (b, t_x, t_y))
    path = path - tf.pad(path, [[0, 0], [1, 0], [0, 0]])[:, :-1]
    path = tf.expand_dims(path, axis=1)
    path = tf.transpose(path, perm=[0, 1, 3, 2])
    path = path * mask
    return path

@tf.function
def clip_grad_value_(gradients, clip_value, norm_type=2):
    if isinstance(gradients, tf.Tensor):
        gradients = [gradients]
    gradients = [g for g in gradients if g is not None]
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for grad in gradients:
        param_norm = tf.norm(grad, ord=norm_type)
        total_norm += tf.pow(param_norm, norm_type)
        if clip_value is not None:
            grad = tf.clip_by_value(grad, -clip_value, clip_value)
    total_norm = tf.pow(total_norm, 1. / norm_type)
    return total_norm
