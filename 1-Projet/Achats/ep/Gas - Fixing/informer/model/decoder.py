import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D

from model.utils import lookahead_mask
from model.attention import MultiHeadAttention, AddNormalization, FeedForward

#implemente Decoder
class DecoderLayer(Layer):
    def __init__(self, h, d_model, rate, d_ff, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        # self.build(input_shape=[None, sequence_length, d_model])
        self.multihead_attention1 = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()
        self.training=False

    def call(self, x, lookahead_mask, encoder_out):
        out,_ = self.multihead_attention1(x, lookahead_mask)
        out = self.dropout1(out, training=self.training)
        out = self.add_norm1(x, out)
        out2 = self.multihead_attention2(out, value=encoder_out)
        out2 = self.dropout2(out2, training=self.training)
        out2 = self.add_norm1(out, out2) 
        out3 = self.feed_forward(out2)
        out3 = self.dropout3(out3, training=self.training)
        out3 = self.add_norm3(out2,out3)
        return out3

class Decoder(Layer):
    def __init__(self, h, d_model, rate, d_ff, N, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, d_model)
        # self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(h, d_model, rate, d_ff) for _ in range(N)]

    def call(self, x, encoder_out):
        # Generate the positional encoding
        # pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)
        training= False
        # Add in a dropout layer
        x = self.dropout(x, training=training)
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, lookahead_mask, encoder_out)
 
        return x