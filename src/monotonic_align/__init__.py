import tensorflow as tf
import numpy as np
from core import maximum_path_c

def maximum_path(neg_cent, mask):
    """ TensorFlow optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    neg_cent_np = neg_cent.numpy().astype(np.float32)
    path = tf.zeros_like(neg_cent, dtype=tf.int32)

    t_t_max = tf.reduce_sum(mask, axis=1)[:, 0]
    t_s_max = tf.reduce_sum(mask, axis=2)[:, 0]
    
    maximum_path_c(path.numpy(), neg_cent_np, t_t_max.numpy(), t_s_max.numpy())
    return path