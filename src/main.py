import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model 
from .ResidualCouplingBlock import ResidualCouplingBlock
from .PriorEncoder import PriorEncoder
from .Decoder import Decoder
from .PosteriorEncoder import PosteriorEncoder
from .StochasticDurationPredictor import StochasticDurationPredictor

class SynthesizerTrn(Model):
    def __init__(self, n_vocab, spec_channels, inter_channels, hidden_channels, 
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, 
                 **kwargs):
        super(SynthesizerTrn, self).__init__()

        self.embd = keras.layers.Embedding(n_vocab, inter_channels)

        self.pri_enc = PriorEncoder(128, 256, n_heads, n_layers, n_vocab, 256, inter_channels, hidden_channels, kernel_size, )
        self.dec = Decoder(resblock_kernel_sizes, 
                             resblock_dilation_sizes, upsample_rates, upsample_initial_channel, 
                             upsample_kernel_sizes, gin_channels=gin_channels)
        self.post_enc = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 
                                      16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, 
                                          gin_channels=gin_channels)

        self.sdp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, 
                                                  gin_channels=gin_channels)


    def call(self, inputs, training=True):
        x_lin, c_text = inputs["x_lin"], inputs["text"]
        mu, logvar = self.post_enc({"input" : x_lin, "global" : None})
        z = self.reparameterization(mu, logvar)

        duration, h_text = self.pri_enc({"latentInput": z, "textInput":c_text, "global":None})

        asdf = self.sdp({"input": h_text, "d":duration})

        return self.dec({"input" : z})

    def reparameterization(self, mu, logvar):
        std = tf.exp(logvar/2)
        return mu + std
