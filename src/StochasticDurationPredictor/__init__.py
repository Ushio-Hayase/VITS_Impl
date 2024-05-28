import tensorflow as tf
from tensorflow import keras
from StochasticDurationPredictor.modules import Log, ElementwiseAffine, ConvFlow, Flip, ResBlock
import numpy as np

class StochasticDurationPredictor(keras.layers.Layer):
    def __init__(self, filter_channels, kernel_size, dropout, n_flows=4, gin_channels=0):
        super().__init__()

        self.log_flow = Log()
        flows = []
        flows.append(ElementwiseAffine(2))

        for _ in range(n_flows):
            flows.append(ConvFlow(filter_channels, kernel_size, n_layers=3))
            flows.append(Flip())

        self.flow = keras.Sequential(flows)

        self.post_pre = keras.layers.Conv1D(filter_channels, 1)
        self.post_proj = keras.layers.Conv1D(filter_channels, 1)
        self.post_convs = ResBlock(filter_channels, kernel_size, n_layers=3, p_dropout=dropout)
        self.post_flows = []

        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = keras.layers.Conv1D(filter_channels, 1)
        self.proj = keras.layers.Conv1D(filter_channels, 1)
        self.convs = ResBlock(filter_channels, kernel_size, n_layers=3, p_dropout=dropout)
        if gin_channels != 0:
            self.cond = keras.layers.Conv1D(filter_channels, 1)

    def call(self, inputs, training=True,):
        x, x_mask, w, g, noise_scale = inputs["input"], inputs["mask"], \
            inputs["w"], inputs["g"], inputs["noise_scale"]
        x = tf.stop_gradient(x)
        x = self.pre(x)
        if g is not None:
            g = tf.stop_gradient(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if training:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0 
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = tf.random.normal(w.shape) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = tf.split(z_q, num_or_size_splits=2, axis=2)
            u = tf.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += tf.reduce_sum((tf.math.log_sigmoid(z_u) + tf.math.log_sigmoid(-z_u)) * x_mask, axis=[1,2])
            logq = tf.reduce_sum(-0.5 * (np.log(2 * np.pi) + (e_q**2)) * x_mask, axis=[1,2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = tf.concat([z0, z1], axis=2)
            for flow in flows:
                z, logdet = flow({"input": z, "mask":x_mask, "global":x}, training)
                logdet_tot += logdet
            nll = tf.reduce_sum(0.5 * (np.log(2 * np.pi) + (z**2)) * x_mask, axis=[1,2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = tf.random.normal(x.shape) * noise_scale
            for flow in flows:
                z = flow({"input": z, "mask":x_mask, "global":x}, training)
            z0, z1 = tf.split(z, num_or_size_splits=2, axis=2)
            logw = z0
            return logw
