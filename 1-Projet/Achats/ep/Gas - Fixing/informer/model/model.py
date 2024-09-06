import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D
from keras.losses import sparse_categorical_crossentropy, binary_focal_crossentropy, mean_squared_error,poisson
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from tensorflow.keras import Model

from model.embedding import myEmbedding, DataEmbedding
from model.encoder import Encoder,EncoderInf,EncoderInfLayer, ConvLayer
from model.attention import AttentionLayer, FullAttention, ProbAttention
from model.decoder import DecoderInf, DecoderInfLayer
from model.utils import Normalize


# Defining the loss function
def loss_fcn(y_true, y_pred):
    loss = mean_squared_error(y_true=y_true, y_pred=y_pred)
    # loss = huber_loss(y_true=y_true, y_pred=y_pred)
    # return loss/loss.shape[0]
    return (reduce_sum(loss)/loss.shape[0])

# Defining the accuracy function
def accuracy_fcn(y_true, y_pred):
    accuracy = tf.abs(y_true - y_pred)
    return (1 - reduce_sum(accuracy)/100./accuracy.shape[0])


#Informer Model
class myInformer(Model):
    def __init__(self, seq_len, pred_len, batch_size, d_model, rate , factor, head,d_ff,e_layer, features, **kwargs):
        super(myInformer, self).__init__(**kwargs)
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.Embedding = DataEmbedding(seq_len=seq_len, d_model=d_model,rate=rate,timeF=True )
        self.Encoder = EncoderInf( [ EncoderInfLayer( 
                        AttentionLayer(
                            ProbAttention(False,factor,None,rate,True)
                                ,d_model, head, None, None, False)
                            ,d_model, rate,d_ff) for _ in range(e_layer) 
                        ]                     ,
                    [ ConvLayer(seq_len=seq_len) for _ in range (e_layer - 1)
                    ])
        self.Decoder = DecoderInf( DecoderInfLayer (
                        AttentionLayer(
                            ProbAttention(False,factor,None,rate,True)
                                ,d_model,head,None,None,False),
                        AttentionLayer(
                            FullAttention(False,factor,None,rate,False)
                                ,d_model, head, None, None, False)
                        ,d_model,None,rate,'relu')
                    ,norm_layer= LayerNormalization()
                   )
        self.Projection = Dense(features)

    def call(self, x, x_date , start=0 ):
        L, S, F = x.shape
        _, _, Fd = x_date.shape
        x,scaler = Normalize(tf.reshape(x, [L,-1]))
        x = tf.reshape(x,[L,S,F])
        x = tf.cast(x,tf.float32)
        x_date,scaler = Normalize(tf.reshape(x_date,[L,-1]))
        x_date = tf.reshape(x_date,[L,S,Fd])
        x_date = tf.cast(x_date,tf.float32)

        dec_len = self.seq_len // 2
        x_enc, x_date_enc = [ x[start:start+self.batch_size,:self.seq_len,:] , x_date[start:start+self.batch_size,:self.seq_len,:]]
        x_dec, x_date_dec = [ x[start:start+self.batch_size,:dec_len,:] , x_date[start:start+self.batch_size,:dec_len,:]]
        x_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,x.shape[-1]))
        x_date_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,x_date.shape[-1]))
        x_dec = tf.concat([x_dec,x_dec_padd],1)
        x_date_dec = tf.concat([x_date_dec,x_date_dec_padd],1)

        #Embedding
        encEmb = self.Embedding(x=x_enc, x_mark=x_date_enc, training=True)
        decEmb = self.Embedding(x=x_dec, x_mark=x_date_dec, training =True)
        #Encoder
        outEnc,attnE = self.Encoder(encEmb, training=True)
        #Decoder
        outDec = self.Decoder(decEmb, outEnc,training=True)
        #Projection
        out = self.Projection(outDec)

        return out, outDec, outEnc, decEmb, encEmb, attnE, x, x_date, x_enc

#My Forecasting 
class myForcast(Model):
    def __init__(self,seq_len, h, d_model,rate, d_ff, N, **kwargs):
        super(myForcast, self).__init__(**kwargs)
        # self.MyE = myEmbedding(seq_len,d_model,rate)
        self.MyE = DataEmbedding(seq_len,d_model,rate, 'fixed','h',False)
        self.Encoder = Encoder(h,d_model,rate,d_ff,N)
        self.linear1 = Dense(d_ff/2)#, activation='sigmoid')
        self.linear2 = Dense(d_ff/8)#, activation='sigmoid')
        self.linear3 = Dense(d_ff/32)#, activation='sigmoid')
        # self.linear4 = Dense(d_ff/128)#, activation='sigmoid')
        self.finish = Dense(1)#, activation='sigmoid')
        self.relu = ReLU()
        self.dropout = Dropout(rate)

    def call(self, x, training=False):
        x = self.MyE(x, training=training)
        y = self.Encoder(x,training)
        y = tf.reshape(y, shape=[y.shape[0],y.shape[1]*y.shape[2]])
        y = self.linear1(y)
        y = self.relu(y)
        y = self.dropout(y, training)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.dropout(y, training)
        y = self.linear3(y)
        y = self.relu(y)
        y = self.dropout(y, training)
        # y = self.linear4(y)
        # y = self.dropout(y, training)
        y = self.finish(y)
        # print(y.shape,y.numpy())
        return y