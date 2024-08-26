import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, ReLU, Embedding,Conv1D
from keras.losses import sparse_categorical_crossentropy, binary_focal_crossentropy, mean_squared_error,poisson
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from tensorflow.keras import Model

from model.embedding import myEmbedding
from model.encoder import Encoder


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



#My Forecasting 
class myForcast(Model):
    def __init__(self,seq_len, h, d_model,rate, d_ff, N, **kwargs):
        super(myForcast, self).__init__(**kwargs)
        self.MyE = myEmbedding(seq_len,d_model,rate)
        self.Encoder = Encoder(h,d_model,rate,d_ff,N)
        self.linear1 = Dense(d_ff/2)#, activation='sigmoid')
        self.linear2 = Dense(d_ff/8)#, activation='sigmoid')
        self.linear3 = Dense(d_ff/32)#, activation='sigmoid')
        # self.linear4 = Dense(d_ff/128)#, activation='sigmoid')
        self.finish = Dense(1)#, activation='sigmoid')
        self.relu = ReLU()
        self.dropout = Dropout(rate)

    def call(self, x, training=False):
        x = self.MyE(x, conv=False, training=training)
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