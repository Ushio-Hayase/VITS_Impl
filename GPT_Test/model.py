import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, Embedding, Dense, Layer, Dropout, LeakyReLU
from tensorflow.keras import initializers, Model
import numpy as np

from GPT_Test.commons import generate_path, sequence_mask

class StochasticDurationPredictor(Layer):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = []
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = Conv1D(filter_channels, 1)
        self.post_proj = Conv1D(filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = []
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = Conv1D(filter_channels, 1)
        self.proj = Conv1D(filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = Conv1D(filter_channels, 1)

    def call(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = tf.stop_gradient(x)
        x = self.pre(x)
        if g is not None:
            g = tf.stop_gradient(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0 
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = tf.random.normal((w.shape[0], 2, w.shape[2]), dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = tf.split(z_q, 2, axis=1) 
            u = tf.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += tf.reduce_sum((tf.math.log_sigmoid(z_u) + tf.math.log_sigmoid(-z_u)) * x_mask, [1,2])
            logq = tf.reduce_sum(-0.5 * (np.log(2*np.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = tf.concat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = tf.reduce_sum(0.5 * (np.log(2*np.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = tf.random.normal((x.shape[0], 2, x.shape[2]), dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = tf.split(z, 2, axis=1)
            logw = z0
            return logw

class DurationPredictor(Layer):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = Dropout(p_dropout)
        self.conv_1 = Conv1D(filter_channels, kernel_size, padding='same')
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = Conv1D(filter_channels, kernel_size, padding='same')
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = Conv1D(1, 1)

        if gin_channels != 0:
            self.cond = Conv1D(in_channels, 1)

    def call(self, x, x_mask, g=None):
        x = tf.stop_gradient(x)
        if g is not None:
            g = tf.stop_gradient(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = tf.nn.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = tf.nn.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

class TextEncoder(Layer):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = Embedding(n_vocab, hidden_channels, embeddings_initializer=initializers.RandomNormal(0.0, hidden_channels**-0.5))
        self.encoder = (hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = Conv1D(out_channels * 2, 1)

    def call(self, x, x_lengths):
        x = self.emb(x) * tf.math.sqrt(tf.cast(self.hidden_channels, tf.float32))  # [b, t, h]
        x = tf.transpose(x, perm=[0, 2, 1])  # [b, h, t]
        x_mask = tf.expand_dims(sequence_mask(x_lengths, x.shape[2]), 1)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = tf.split(stats, 2, axis=1)
        return x, m, logs, x_mask

class ResidualCouplingBlock(Layer):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = []
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def call(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class PosteriorEncoder(Layer):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = Conv1D(hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = Conv1D(out_channels * 2, 1)

    def call(self, x, x_lengths, g=None):
        x_mask = tf.expand_dims(sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = tf.split(stats, 2, axis=1)
        z = (m + tf.random.normal(m.shape) * tf.exp(logs)) * x_mask
        return z, m, logs, x_mask

class Generator(Layer):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1D(upsample_initial_channel, 7, padding='same')

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(Conv1DTranspose(upsample_initial_channel//(2**(i+1)), k, strides=u, padding='same'))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(resblock, ch, k, d))

        self.conv_post = Conv1D(1, 7, padding='same')
        self.ups = tf.keras.Sequential(self.ups)
        self.resblocks = tf.keras.Sequential(self.resblocks)

    def call(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            x_f = self.resblocks[i*self.num_kernels](x)
            for j in range(1, self.num_kernels):
                x_f += self.resblocks[i*self.num_kernels+j](x)
            x = x_f / self.num_kernels
        x = self.conv_post(x)
        x = tf.nn.tanh(x)
        return x

class DiscriminatorP(Layer):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = tf.keras.layers.LayerNormalization

        self.convs = [
            norm_f(Conv2D(32, (kernel_size, 1), (stride, 1), padding='same')),
            norm_f(Conv2D(128, (kernel_size, 1), (stride, 1), padding='same')),
            norm_f(Conv2D(512, (kernel_size, 1), (stride, 1), padding='same')),
            norm_f(Conv2D(1024, (kernel_size, 1), (stride, 1), padding='same')),
            norm_f(Conv2D(1024, (kernel_size, 1), 1, padding='same')),
        ]
        self.conv_post = Conv2D(1, (3, 1), 1, padding='same')

    def call(self, x):
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = tf.pad(x, [[0, 0], [0, 0], [0, n_pad]], mode='reflect')
        x = tf.reshape(x, [b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x)
            x = LeakyReLU(0.1)(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = tf.reshape(x, [b, -1])
        return x, fmap

class MultiPeriodDiscriminator(Layer):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        self.discriminators = [DiscriminatorP(2, use_spectral_norm=use_spectral_norm),
                               DiscriminatorP(3, use_spectral_norm=use_spectral_norm),
                               DiscriminatorP(5, use_spectral_norm=use_spectral_norm),
                               DiscriminatorP(7, use_spectral_norm=use_spectral_norm),
                               DiscriminatorP(11, use_spectral_norm=use_spectral_norm)]

    def call(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    

class SynthesizerTrn(Model):
    def __init__(self, 
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock, 
                 resblock_kernel_sizes, 
                 resblock_dilation_sizes, 
                 upsample_rates, 
                 upsample_initial_channel, 
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):
        super(SynthesizerTrn, self).__init__()
        
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        
        self.enc_p = TextEncoder(n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        
        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
        
        if n_speakers > 1:
            self.emb_g = Embedding(n_speakers, gin_channels)
    
    def call(self, x, x_lengths, y, y_lengths, sid=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = tf.expand_dims(self.emb_g(sid), -1) # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        
        s_p_sq_r = tf.exp(-2 * logs_p) # [b, d, t]
        neg_cent1 = tf.reduce_sum(-0.5 * np.log(2 * np.pi) - logs_p, axis=1, keepdims=True) # [b, 1, t_s]
        neg_cent2 = tf.matmul(-0.5 * tf.transpose(z_p ** 2, perm=[0, 2, 1]), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
        neg_cent3 = tf.matmul(tf.transpose(z_p, perm=[0, 2, 1]), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
        neg_cent4 = tf.reduce_sum(-0.5 * (m_p ** 2) * s_p_sq_r, axis=1, keepdims=True) # [b, 1, t_s]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
        
        attn_mask = tf.expand_dims(x_mask, 2) * tf.expand_dims(y_mask, -1)
        attn = maximum_path(neg_cent, tf.squeeze(attn_mask, axis=1)).unsqueeze(1)
        
        w = tf.reduce_sum(attn, axis=2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / tf.reduce_sum(x_mask)
        else:
            logw_ = tf.math.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = tf.reduce_sum((logw - logw_)**2, axis=[1,2]) / tf.reduce_sum(x_mask)
        
        m_p = tf.matmul(tf.squeeze(attn, axis=1), tf.transpose(m_p, perm=[0, 2, 1]))
        m_p = tf.transpose(m_p, perm=[0, 2, 1])
        logs_p = tf.matmul(tf.squeeze(attn, axis=1), tf.transpose(logs_p, perm=[0, 2, 1]))
        logs_p = tf.transpose(logs_p, perm=[0, 2, 1])
        
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
    
    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = tf.expand_dims(self.emb_g(sid), -1) # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = tf.exp(logw) * x_mask * length_scale
        w_ceil = tf.math.ceil(w)
        y_lengths = tf.clip_by_value(tf.reduce_sum(w_ceil, axis=[1, 2]), 1, np.inf).astype(tf.int32)
        y_mask = tf.expand_dims(sequence_mask(y_lengths, None), axis=1)
        attn_mask = tf.expand_dims(x_mask, 2) * tf.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = tf.matmul(tf.squeeze(attn, axis=1), tf.transpose(m_p, perm=[0, 2, 1]))
        m_p = tf.transpose(m_p, perm=[0, 2, 1]) # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = tf.matmul(tf.squeeze(attn, axis=1), tf.transpose(logs_p, perm=[0, 2, 1]))
        logs_p = tf.transpose(logs_p, perm=[0, 2, 1]) # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + tf.random.normal(tf.shape(m_p)) * tf.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = tf.expand_dims(self.emb_g(sid_src), -1)
        g_tgt = tf.expand_dims(self.emb_g(sid_tgt), -1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
    
