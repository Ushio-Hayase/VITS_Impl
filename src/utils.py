import tensorflow as tf
from tensorflow import keras

def pad_mask(x, pad_id : int):
    pad = [x!=pad_id][:, tf.newaxis, tf.newaxis, :]
    return pad

def relative_position_encoding(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(tf.int64) * num_buckets
        n = tf.abs(n)
    else:
        n = tf.max(n, tf.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            tf.math.log(n.float() / max_exact) / tf.math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(tf.int64)
    val_if_large = tf.min(val_if_large, tf.fill(val_if_large, num_buckets - 1))
    ret += tf.where(is_small, n, val_if_large)
    return ret