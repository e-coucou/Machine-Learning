import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D

from model.attention import MultiHeadAttention, AddNormalization, FeedForward

# Implementing Encoder
class EncoderLayer(Layer):
    def __init__(self, h, d_model, rate, d_ff, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.h = h
        # self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        # self.sequence_length = sequence_length
        self.multihead_attention = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, training=False):
        out,_ = self.multihead_attention(x, None)
        out = self.dropout1(out, training=training)
        out = self.add_norm1(x, out)
        out2 = self.feed_forward(out)
        out2 = self.dropout2(out2, training=training)
        out2 = self.add_norm2(out,out2)
        return out2

class Encoder(Layer):
    def __init__(self, h, d_model, rate, d_ff, N, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        # self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_model, rate, d_ff) for _ in range(N)]

    def call(self, x, training=False):
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x,training)
        return x