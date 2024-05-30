import tensorflow as tf
from tensorflow import keras
from .ResidualCouplingLayer import ResidualCouplingLayer
from ..utils import Flip

class ResidualCouplingBlock(keras.layers.Layer):
    def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4):
        super().__init__()
        
        flows = []
        for _ in range(n_flows):
            flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers))
            flows.append(Flip())
        
        self.flows = keras.Sequential(flows)

    def call(self, inputs, training=True):
        x, x_mask = inputs["input"], inputs["mask"]
        if training:
            for flow in self.flows:
                x, _ = flow({"inputs" : x, "mask" : x_mask}, training)
        else:
            for flow in reversed(self.flows):
                x = flow({"inputs" : x, "mask" : x_mask}, training)

        return x

