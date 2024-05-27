import tensorflow as tf


class RelativePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=10000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.relative_position_embeddings = self.add_weight(
            shape=(2 * max_len - 1, d_model),
            initializer='random_uniform',
            trainable=True,
            name='relative_position_embeddings'
        )
    
    def call(self, length):
        positions = tf.range(length)
        rel_pos = positions[:, None] - positions[None, :]
        rel_pos = rel_pos + self.max_len - 1  # Shift to 0-based index
        return tf.nn.embedding_lookup(self.relative_position_embeddings, rel_pos)

class TextEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding):
        super(TextEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.relative_position_encoding = RelativePositionalEncoding(d_model, maximum_position_encoding)
        self.enc_layers = [tf.keras.layers.TransformerEncoder(num_layers, d_model, num_heads, dff) for _ in range(num_layers)]
    
    def call(self, x):
        x, mask = x["input"], x["mask"]
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Add relative positional encoding
        pos_encoding = self.relative_position_encoding(seq_len)
        x += pos_encoding
        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x, padding_mask = mask)
        
        return x
