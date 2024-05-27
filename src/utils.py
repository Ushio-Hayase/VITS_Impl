import tensorflow as tf
from tensorflow import keras

def pad_mask(x, pad_id : int):
    pad = [x!=pad_id][:, tf.newaxis, tf.newaxis, :]
    return pad

def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = keras.backend.arange(max_length, dtype=length.dtype, device=length.device)
  return x[tf.newaxis, :] < length[:, tf.newaxis]

def log_likelihood(z, mu, sigma):
    with tf.device("/CPU:0"):
        return -0.5 * tf.reduce_sum(tf.square((z - mu) / sigma), axis=-1) - tf.reduce_sum(tf.math.log(sigma), axis=-1)

# MAS 동적 계획법 구현
def monotonic_alignment_search(z, mu, sigma):
    with tf.device("/CPU:0"):
        batch_size, seq_len = tf.shape(z)[0], tf.shape(z)[1]
        dp = tf.TensorArray(tf.float32, size=seq_len+1, dynamic_size=False, clear_after_read=False)
        dp = dp.write(0, tf.zeros((batch_size,), dtype=tf.float32))
        
        for j in tf.range(1, seq_len+1):
            current_ll = log_likelihood(z[:, j-1], mu[j-1], sigma[j-1])
            max_prev_ll = tf.maximum(dp.read(j-1), dp.read(j-1))
            current_q = max_prev_ll + current_ll
            dp = dp.write(j, current_q)
        
        # backtracking
        alignment = []
        i = seq_len
        while i > 0:
            alignment.append(i)
            if dp.read(i) == dp.read(i-1):
                i -= 1
            i -= 1
        
        alignment = tf.reverse(alignment, axis=[0])
        return alignment