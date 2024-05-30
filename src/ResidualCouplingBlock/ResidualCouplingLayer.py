import tensorflow as tf
from tensorflow.keras import layers

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

        self.pre = layers.Conv1D(hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = layers.Conv1D(self.half_channels * (2 - mean_only), 1)
        self.post.build((None, None, hidden_channels))
        self.post.set_weights([tf.zeros_like(w) for w in self.post.get_weights()])

    def call(self, inputs, training=True):
        x, x_mask = inputs["input"], x_mask["mask"]
        x0, x1 = tf.split(x, num_or_size_splits=2, axis=-1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        
        if not self.mean_only:
            m, logs = tf.split(stats, num_or_size_splits=2, axis=-1)
        else:
            m = stats
            logs = tf.zeros_like(m)

        if training:
            x1 = m + x1 * tf.exp(logs) * x_mask
            x = tf.concat([x0, x1], axis=-1)
            logdet = tf.reduce_sum(logs, axis=[1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * tf.exp(-logs) * x_mask
            x = tf.concat([x0, x1], axis=-1)
            return x

class WN(tf.keras.layers.Layer):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = []
        self.res_skip_layers = []
        self.drop = layers.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = layers.Conv1D(2 * hidden_channels * n_layers, 1, padding='same')
            self.cond_layer = tf.keras.utils.WeightNormalization(self.cond_layer)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) // 2)
            in_layer = layers.Conv1D(2 * hidden_channels, kernel_size, dilation_rate=dilation, padding='same')
            in_layer = tf.keras.utils.WeightNormalization(in_layer)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = layers.Conv1D(res_skip_channels, 1, padding='same')
            res_skip_layer = tf.keras.utils.WeightNormalization(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def call(self, x, x_mask, g=None):
        output = tf.zeros_like(x)
        n_channels_tensor = self.hidden_channels

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, :, cond_offset:cond_offset + 2 * self.hidden_channels]
            else:
                g_l = tf.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :, :self.hidden_channels]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, :, self.hidden_channels:]
            else:
                output = output + res_skip_acts

        return output * x_mask

def fused_add_tanh_sigmoid_multiply(x, y, n_channels):
    """
    Helper function to perform the fused add, tanh, sigmoid, and multiply operations.
    """
    assert x.shape[-1] == 2 * n_channels
    x_tanh = tf.math.tanh(x[:, :, :n_channels] + y[:, :, :n_channels])
    x_sigmoid = tf.math.sigmoid(x[:, :, n_channels:] + y[:, :, n_channels:])
    return x_tanh * x_sigmoid

# Example usage:
# wn_layer = WN(hidden_channels=64, kernel_size=3, dilation_rate=2, n_layers=6, gin_channels=0, p_dropout=0.1)
# output = wn_layer(input_tensor, input_mask, g=None)

