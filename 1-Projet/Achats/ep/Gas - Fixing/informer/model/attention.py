# import numpy as np
# from keras import Input
# from keras import layers, Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32,reduce_max, reduce_min
from keras.backend import softmax



# Implementing Dot Product - return Values/Weights
class DotProduct(Layer):
    def __init__(self, **kwargs):
        super(DotProduct, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, mask=None):
        d_k=queries.shape[-1]
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
        # Apply mask to the attention scores
        if mask is not None:
            scores += mask
        # Computing the weights by a softmax operation
        weights = softmax(scores)
        out = matmul(weights, values)
        # Computing the attention by a weighted sum of the value vectors
        return out, weights

# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProduct()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.head_dim = int(d_model / h)
        self.d_model = d_model  # Dimensionality of the model
        self.qkv_layer = Dense(3*d_model)  # Learned projection matrix for the queries
        self.dense = Dense(d_model)  # Learned projection matrix for the multi-head output
  
    def call(self, x, mask=None, value=None):
        batch_size = x.shape[0]
        seq_lenght = x.shape[1]
        input_dim = x.shape[2]
        qkv = self.qkv_layer(x)
        qkv = tf.reshape(qkv,shape=[batch_size , seq_lenght , self.heads , int(3*self.head_dim)])
        qkv = transpose(qkv,perm=(0,2,1,3))
        # print(qkv.shape, batch_size , self.heads, seq_lenght , self.head_dim)
        q = tf.slice(qkv,[0,0,0,0],[batch_size,self.heads,seq_lenght,self.head_dim])
        k = tf.slice(qkv,[0,0,0,self.head_dim],[batch_size,self.heads,seq_lenght,self.head_dim])
        if (value==None):
            v = tf.slice(qkv,[0,0,0,int(2*self.head_dim)],[batch_size,self.heads,seq_lenght,self.head_dim])
        else:
            v = value.slice()
 
        # # Compute the multi-head attention output using the reshaped queries, keys and values
        attention, weights = self.attention(q, k, v, mask)
        # print('attention',attention.shape)
        attention = tf.transpose(attention,perm=(0,2,1,3))
        attention = tf.reshape(attention,shape=[batch_size,seq_lenght,int(self.heads*self.head_dim)])

        # # Apply one final linear projection to the output to generate the multi-head attention
        # # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        # return attention
        return attention,weights
    

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
 
    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        x = x + sublayer_x
        x = self.layer_norm(x)
        return x
    
class Normalization(Layer):
    def __init__(self, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.Norm = LayerNormalization()
    def call(self, x):
        x = self.Norm(x)
        return x
    

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.linear1 = Dense(d_ff)  # First fully connected layer
        self.linear2 = Dense(d_model)  # Second fully connected layer
        self.relu = ReLU()  # ReLU activation layer
 
    def call(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x