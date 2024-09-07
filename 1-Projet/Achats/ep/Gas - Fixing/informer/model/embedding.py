import numpy as np
# from keras import Input
# from keras import layers, Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D
from tensorflow.keras.initializers import HeUniform
import math

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
    
class FixedEmbedding(Layer):
    def __init__(self, c_in, d_model, **kwargs):
        super(FixedEmbedding, self).__init__(**kwargs)
        w = np.zeros((c_in, d_model),dtype=np.float32)

        position = np.arange(0, c_in,dtype=np.float32)[:,np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        w[:, 0::2] = np.sin(position * div_term)
        w[:, 1::2] = np.cos(position * div_term)

        self.emb = Embedding(input_dim=c_in, output_dim=d_model, weights=[w], trainable=False)

    def call(self, x):
        y = self.emb(x)
        return y

# Projection Embedding using Conv1D
class projEmbedding(Layer):
    def __init__(self, d_model, rate, **kwargs):
        super(projEmbedding, self).__init__(**kwargs)
        self.conv1D = Conv1D(filters=d_model,kernel_size=3,padding="causal", kernel_initializer="he_uniform") # vs "same"
        self.dropout = Dropout(rate)
        self.Norm = LayerNormalization()

    def call(self, x, training=False):
        x = self.conv1D(x)
        x = self.dropout(x, training=training)
        x = self.Norm(x)
        return x
    
class TokenEmbedding(Layer): # CHECK OK
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = Conv1D(filters=d_model,kernel_size=3,padding="causal", kernel_initializer="he_uniform") # vs "same"

    def call(self, x):
        out = self.tokenConv(x)
        return out
    
class TimeFeatureEmbedding(Layer):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        # d_inp = freq_map[freq]
        self.embed = Dense(d_model,bias_initializer=HeUniform())
    
    def call(self, x):
        y = self.embed(x)
        return y

class TemporalEmbedding(Layer):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def call(self, x):
        x = tf.cast(x, dtype=tf.int64)
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

# My data Embedding     
class myEmbedding(Layer):
    def __init__(self,seq_len, d_model, rate, **kwargs):
        super(myEmbedding, self).__init__(**kwargs)
        self.PositionEmb = PositionEmbedding(seq_len, d_model)
        # self.DataEmb = DataEmbedding(d_model, rate)
        self.ProjEmb = projEmbedding(d_model,rate)
        self.Norm = LayerNormalization()

    def call(self, x, training=False):
        # x = tf.cast(x, dtype=tf.float32)
        # x = tf.reshape(x,[x.shape[0],x.shape[1],1])
        x1 = self.PositionEmb(x)
        x2 = self.ProjEmb(x)
        out = (x1+x2)
        out = self.Norm(out)
        return out
    
class DataEmbedding(Layer):
    def __init__(self, seq_len, d_model, rate=0.1, embed_type='fixed', freq='h', timeF=True, token = True):
        super(DataEmbedding, self).__init__()
        self.timeF = timeF
        self.token_embedding = TokenEmbedding(d_model=d_model) if token else projEmbedding(d_model=d_model,rate=rate)
        self.position_embedding = PositionEmbedding(seq_len=seq_len,d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = Dropout(rate)

    def call(self, x, x_mark=0, training=False):
        self.a = self.token_embedding(x)
        self.b = self.position_embedding(x)
        self.c = self.temporal_embedding(x_mark) if self.timeF else 0
        self.out = self.a + self.b + self.c
        self.outD = self.dropout(self.out, training=training)
        
        return self.outD