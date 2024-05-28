import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model # type: ignore
from 

class SynthesizerTrn(Model):
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, 
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
                 upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0, 
                 use_sdp=True, **kwargs):
        super(SynthesizerTrn, self).__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
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
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, 
                                 n_heads, n_layers, kernel_size, p_dropout)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, 
                             resblock_dilation_sizes, upsample_rates, upsample_initial_channel, 
                             upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 
                                      16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, 
                                          gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, 
                                                  gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = layers.Embedding(n_speakers, gin_channels)

    def call(self, x, x_lengths, y, y_lengths, sid=None, training=False):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = tf.expand_dims(self.emb_g(sid), -1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        # negative cross-entropy
        s_p_sq_r = tf.exp(-2 * logs_p)  # [b, d, t]
        neg_cent1 = tf.reduce_sum(-0.5 * tf.math.log(2 * math.pi) - logs_p, axis=1, keepdims=True)  # [b, 1, t_s]
        neg_cent2 = tf.matmul(-0.5 * tf.transpose(z_p, perm=[0, 2, 1]) ** 2, s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
        neg_cent3 = tf.matmul(tf.transpose(z_p, perm=[0, 2, 1]), m_p * s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
        neg_cent4 = tf.reduce_sum(-0.5 * m_p ** 2 * s_p_sq_r, axis=1, keepdims=True)  # [b, 1, t_s]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        attn_mask = tf.expand_dims(x_mask, 2) * tf.expand_dims(y_mask, -1)
        attn = monotonic_align.maximum_path(neg_cent, tf.squeeze(attn_mask, 1))[:, tf.newaxis, :]

        w = tf.reduce_sum(attn, axis=2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g, training=training)
            l_length = l_length / tf.reduce_sum(x_mask)
        else:
            logw_ = tf.math.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g, training=training)
            l_length = tf.reduce_sum((logw - logw_) ** 2, axis=[1, 2]) / tf.reduce_sum(x_mask)  # for averaging 

        # expand prior
        m_p = tf.matmul(tf.squeeze(attn, 1), tf.transpose(m_p, perm=[0, 2, 1]))
        logs_p = tf.matmul(tf.squeeze(attn, 1), tf.transpose(logs_p, perm=[0, 2, 1]))

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g, training=training)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = tf.expand_dims(self.emb_g(sid), -1)  # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = tf.exp(logw) * x_mask * length_scale
        w_ceil = tf.math.ceil(w)
        y_lengths = tf.clip_by_value(tf.reduce_sum(w_ceil, axis=[1, 2]), clip_value_min=1, clip_value_max=tf.int64.max)
        y_mask = tf.expand_dims(commons.sequence_mask(y_lengths, None), 1)
        attn_mask = tf.expand_dims(x_mask, 2) * tf.expand_dims(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = tf.matmul(tf.squeeze(attn, 1), tf.transpose(m_p, perm=[0, 2, 1]))
        logs_p = tf.matmul(tf.squeeze(attn, 1), tf.transpose(logs_p, perm=[0, 2, 1]))

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
