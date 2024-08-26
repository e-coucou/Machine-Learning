import numpy as np
# from keras import Input
# from keras import layers, Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D

# Fixe Positional Embedding - google paper    
class PositionEmbedding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        position_embedding_matrix = self.get_position_encoding(seq_len, d_model)                                          
        self.position_embedding_layer = Embedding(
            input_dim=seq_len, output_dim=d_model,
            weights=[position_embedding_matrix],
            trainable=False
        )
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
 
    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[1])
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_indices
    

# Projection Embedding using Conv1D
class projEmbedding(Layer):
    def __init__(self, seq_len, d_model, rate, **kwargs):
        super(projEmbedding, self).__init__(**kwargs)
        self.conv1D = Conv1D(d_model,kernel_size=3,padding="causal", kernel_initializer="he_uniform") # vs "same"
        self.dropout = Dropout(rate)
        self.Norm = LayerNormalization()

    def call(self, x, training=False):
        x = self.conv1D(x)
        x = self.dropout(x, training=training)
        x = self.Norm(x)
        return x
    


# My data Embedding     
class myEmbedding(Layer):
    def __init__(self,seq_len, d_model, rate, **kwargs):
        super(myEmbedding, self).__init__(**kwargs)
        self.PositionEmb = PositionEmbedding(seq_len, d_model)
        # self.DataEmb = DataEmbedding(d_model, rate)
        self.ProjEmb = projEmbedding(seq_len,d_model,rate)
        self.Norm = LayerNormalization()

    def call(self, x, conv=False, training=False):
        # x = tf.cast(x, dtype=tf.float32)
        # x = tf.reshape(x,[x.shape[0],x.shape[1],1])
        x1 = self.PositionEmb(x)
        x2 = self.ProjEmb(x)
        out = (x1+x2)
        out = self.Norm(out)
        return out
    
