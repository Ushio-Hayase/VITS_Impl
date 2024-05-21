import tensorflow as tf
from tensorflow import keras

def pad_mask(x, pad_id : int):
    pad = [x!=pad_id][:, tf.newaxis, tf.newaxis, :]
    return pad