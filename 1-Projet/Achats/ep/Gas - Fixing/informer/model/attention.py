import numpy as np
import math
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

# implemente fullAttention
class FullAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, rate=0.1, output_attention=False, **kwargs):
        super(FullAttention, self).__init__(**kwargs)
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = Dropout(rate)
        
    def call(self, queries, keys, values, attn_mask):
        print(queries.shape)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(float(E))

        scores = tf.einsum('blhe,bshe->bhls', queries, keys)
        print(scores.shape)
        scores = tf.convert_to_tensor(scores)
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)

        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = self.dropout(tf.nn.softmax(scale * scores, axis=-1)) #,training=training)
        out = tf.einsum("bhls,bshd->blhd", attn, values)
        
        if self.output_attention:
            return out,attn
        else:
            return out,None
        
class ProbAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, rate=0.1, output_attention=False, **kwargs):
        super(ProbAttention, self).__init__(**kwargs)
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = Dropout(rate)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D] /batch/seq_len/head/reste...
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # Ktmp = tf.expand_dims(K,axis=-2)
        # print(Ktmp.shape)
        # Ktmp2 = tf.broadcast_to(Ktmp, shape=(B, H, L_Q, L_K, E))
        # print(Ktmp2.shape)
        K_expand = tf.broadcast_to(tf.expand_dims(K,axis=-3), (B, H, L_Q, L_K, E))
        index_sample = tf.random.uniform(maxval=L_K, shape=(L_Q, sample_k),dtype=tf.dtypes.int64)
        # K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        v = tf.expand_dims(np.arange(L_Q),1)
        K_sample = tf.convert_to_tensor(K_expand.numpy()[:, :, v.numpy(), index_sample.numpy(), :])
        # K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # Q_K_sample = tf.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q,axis=-2),tf.transpose(K_sample,perm=(0,1,2,4,3))),axis=-2)

        # find the Top_k query with sparisty measurement
        # M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # M_top = M.topk(n_top, sorted=False)[1]
        M = tf.reduce_max(Q_K_sample, axis=-1) - tf.divide(tf.reduce_sum(Q_K_sample, axis=-1), L_K)
        M_top = tf.math.top_k(M, k=n_top, sorted=False)[1] # [1] on recupere les indices

        # use the reduced Q to calculate Q_K
        # Q_reduce = Q[torch.arange(B)[:, None, None],
        #              torch.arange(H)[None, :, None],
        #              M_top, :] # factor*ln(L_q)
        Q_reduce = Q.numpy()[np.arange(B)[:, None, None],
                    np.arange(H)[None, :, None],
                    M_top.numpy(), :] # factor*ln(L_q)
        Q_K = tf.matmul(Q_reduce, tf.transpose(K, perm=(0,1,3,2))) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = tf.reduce_mean(V, axis=-2)
            contex = tf.broadcast_to(tf.expand_dims(V_sum,axis=-2), (B, H, L_Q, V_sum.shape[-1])) # .clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            # contex = V.cumsum(dim=-2)
            contex = tf.math.cumsum(V, axis=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)
        attn = tf.nn.softmax(scores, axis=-1)
        context_np = context_in.numpy()
        context_np[np.arange(B)[:, None, None],
                    np.arange(H)[None, :, None],
                    index.numpy(), :] = (tf.matmul(attn, V).numpy())

        # context_in[torch.arange(B)[:, None, None],
        #            torch.arange(H)[None, :, None],
        #            index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            # attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            # attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            attns = (np.ones([B, H, L_V, L_V], dtype=np.float32)/L_V)
            attns[np.arange(B)[:, None, None], np.arange(H)[None, :, None], index.numpy(), :] = attn.numpy()
            return (tf.convert_to_tensor(context_np),  tf.convert_to_tensor(attns))
        else:
            return (tf.convert_to_tensor(context_np),  None)
    
    def call(self, q, k, v, attn_mask):
        B, L_Q, H, D = q.shape # Batch/Seq_len/Head/reste...
        _, L_K, _, _ = k.shape

        q = tf.transpose(q, (0,2,1,3))
        k = tf.transpose(k, (0,2,1,3))
        v = tf.transpose(v, (0,2,1,3))

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./math.sqrt(float(D))
        # scale = 1./math.sqrt(float(D))
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(v, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, v, scores_top, index, L_Q, attn_mask)
        
        return tf.transpose(context,perm=(0,2,1,3)), attn

# AttentionLayer for Informer
class AttentionLayer(Layer):
    def __init__(self, attention, d_model, heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//heads) #64
        d_values = d_values or (d_model//heads) #64
        self.inner_attention = attention
        self.query_projection = Dense(d_keys * heads) # d_model
        self.key_projection = Dense(d_keys * heads)
        self.value_projection = Dense(d_values * heads)
        self.out_projection = Dense(d_model) # d_values * n_heads,
        self.heads = heads
        self.mix = mix

    def call(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.heads

        q = tf.reshape(self.query_projection(queries), shape=(B, L, H, -1))
        k = tf.reshape(self.key_projection(keys), shape=(B, S, H, -1))
        v = tf.reshape(self.value_projection(values), shape=(B, S, H, -1))

        print('attention Layer', q.shape)

        out, attn = self.inner_attention( q, k, v, attn_mask )
        if self.mix:
            out = tf.transpose(out, perm=(0,2,1,3))
        out = tf.reshape(out, shape=(B, L, -1))

        return self.out_projection(out), attn
    
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
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_layer(x)
        qkv = tf.reshape(qkv,shape=[batch_size , seq_len , self.heads , int(3*self.head_dim)])
        qkv = transpose(qkv,perm=(0,2,1,3))
        # print(qkv.shape, batch_size , self.heads, seq_lenght , self.head_dim)
        q = tf.slice(qkv,[0,0,0,0],[batch_size,self.heads,seq_len,self.head_dim])
        k = tf.slice(qkv,[0,0,0,self.head_dim],[batch_size,self.heads,seq_len,self.head_dim])
        if (value==None):
            v = tf.slice(qkv,[0,0,0,int(2*self.head_dim)],[batch_size,self.heads,seq_len,self.head_dim])
        else:
            v = value.slice()
 
        # # Compute the multi-head attention output using the reshaped queries, keys and values
        attention, weights = self.attention(q, k, v, mask)
        # print('attention',attention.shape)
        attention = tf.transpose(attention,perm=(0,2,1,3))
        attention = tf.reshape(attention,shape=[batch_size,seq_len,int(self.heads*self.head_dim)])

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