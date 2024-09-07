import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D, ELU,BatchNormalization,MaxPooling1D

from model.attention import MultiHeadAttention, AddNormalization, FeedForward


# Conv Layer
class ConvLayer(Layer):
    def __init__(self, seq_len):
        super(ConvLayer, self).__init__()
        self.downConv = Conv1D( filters=seq_len, kernel_size=3, padding="causal")
        self.norm = BatchNormalization(axis=0)
        self.activation = ELU()
        self.maxPool = MaxPooling1D(pool_size=3, strides=2, padding='same')

    def call(self, x):
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = tf.transpose(x,perm=(0,2,1))
        x = self.maxPool(x)
        return x

# Implementation Enocder for Informer
class EncoderInfLayer(Layer):
    def __init__(self,attention, d_model, rate, d_ff, **kwargs):
        super(EncoderInfLayer, self).__init__(**kwargs)
        self.attention = attention
        self.conv1 = Conv1D(filters=d_ff,kernel_size=3,padding="causal", kernel_initializer="he_uniform") # vs "same"
        self.conv2 = Conv1D(filters=d_model,kernel_size=3,padding="causal", kernel_initializer="he_uniform") # vs "same"
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout = Dropout(rate)
        self.activation = ReLU()

    def call(self, x, attn_mask=None, training=False):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x,training=training)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)), training=training)
        y = self.dropout(self.conv2(y), training=training)

        return self.norm2(x+y), attn    
        
class EncoderInf(Layer):
    def __init__(self, attn_layers, conv_layers=None, N=2):
        super(EncoderInf, self).__init__()
        self.attn_layers = list(attn_layers)
        self.conv_layers = list(conv_layers if conv_layers is not None else None)
        self.norm = LayerNormalization()

    def call(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
   
# Implementing Encoder
class EncoderLayer(Layer):
    def __init__(self, h, d_model, rate, d_ff, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        # self.build(input_shape=[None, sequence_length, d_model])
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