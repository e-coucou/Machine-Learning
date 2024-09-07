import tensorflow as tf
import numpy as np
from time import time

from model.data import getDataETTh1, Normalize
from model.model import myInformer,accuracy_fcn,loss_fcn

from tensorflow.keras.metrics import Mean
# from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD
# from tensorflow.keras.optimizers import AdamW

class Exp():
    def __init__(self, settings, **kwargs):
        super(Exp, self).__init__(**kwargs)
        self.global_size = settings["global_size"]
        self.batch_size = settings["batch_size"]
        self.seq_len = settings["seq_len"]
        self.pred_len = settings["pred_len"]
        self.d_model = settings["d_model"]
        self.rate = settings["rate"]
        self.factor = settings["factor"]
        self.heads = settings["heads"]
        self.d_ff = settings["d_ff"]
        self.e_layer = settings["e_layer"]
        self.d_layer = settings["d_layer"]
        self.features = settings["features"]
        self.dataFct = getDataETTh1(seq_len=self.seq_len,batch_len=self.batch_size,global_size=self.global_size)
        self.modelFct = myInformer(
                seq_len=self.seq_len, pred_len=self.pred_len,batch_size=self.batch_size,d_model=self.d_model,rate=self.rate,factor=self.factor,head=self.heads,d_ff=self.d_ff,
                e_layer=self.e_layer,features=self.features, d_layer=self.d_layer )

    def _getData(self):
        self.x, self.x_date, self.Yt, self.YtDate, self.df_raw = self.dataFct.get()

    def _formatData(self):
        L, S, F = self.x.shape
        _, _, Fd = self.x_date.shape
        self.x,self.scalerX = Normalize(tf.reshape(self.x, [L,-1]))
        self.x = tf.reshape(self.x,[L,S,F])
        self.x = tf.cast(self.x,tf.float32)
        # self.x_date,self.scaler = Normalize(tf.reshape(self.x_date,[L,-1]))
        # self.x_date = tf.reshape(self.x_date,[L,S,Fd])
        self.x_date = tf.cast(self.x_date,tf.float32)

    def _selectData(self, start=0):
        dec_len = self.seq_len // 2
        self.x_enc, self.x_date_enc = [ self.x[start:start+self.batch_size,:self.seq_len,:] , self.x_date[start:start+self.batch_size,:self.seq_len,:]]
        self.x_dec, self.x_date_dec = [ self.x[start:start+self.batch_size,:dec_len,:] , self.x_date[start:start+self.batch_size,:dec_len,:]]
        x_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.x.shape[-1]))
        x_date_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.x_date.shape[-1]))
        self.x_dec = tf.concat([self.x_dec,x_dec_padd],1)
        self.x_date_dec = tf.concat([self.x_date_dec,x_date_dec_padd],1)

    def _pred(self):
        self.pred, self.outDec, self.outEnc, self.decEmb, self.encEmb, attn = self.modelFct(self.x_enc, self.x_date_enc, self.x_dec, self.x_date_dec, training=False)
        return self.pred
            
    def build(self):
        self._getData()
        self._formatData()
        n = self.x.shape[0] if self.global_size==0 else self.global_size
        self.start = np.random.randint(0,n-self.batch_size)
        self._selectData(start=self.start)
    
    def pred(self,x_enc, x_dec,x_date_enc, x_date_dec, training=False):
        self.pred, self.outDec, self.outEnc, self.decEmb, self.encEmb, attn = self.modelFct(x_enc, x_date_enc, x_dec, x_date_dec, training=training)
        return self.pred
    
    def train(self, dataset, datasetVal, epochs):
        # Include metrics monitoring
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy= Mean(name='train_accuracy')
        train_loss_dict = {}
        train_accuracy_dict = {}
        val_loss = Mean(name='val_loss')
        val_accuracy= Mean(name='val_accuracy')
        val_loss_dict = {}
        val_accuracy_dict = {}
        start_time = time()
        for e in range(epochs):
            self.train_loss.reset_states()
            inter=time()
            for i,(btX,btY) in enumerate(dataset):
                btY = tf.reshape(btY,shape=[btY.shape[0],1])
                btX = tf.reshape(btX, shape=[btX.shape[0],self.seq_len,1])
                self.step(btX,btY)
            train_loss_dict[e] = self.train_loss.result()
            train_accuracy_dict[e] = self.train_accuracy.result()

            for i,(btX,btY) in enumerate(datasetVal):
                btY = tf.reshape(btY,shape=[btY.shape[0],1])
                btX = tf.reshape(btX, shape=[btX.shape[0],self.seq_len,1])
                pred = self.model(btX)
                loss = loss_fcn(btY, pred)
                acc = accuracy_fcn(btY, pred)
                val_loss(loss)
                val_accuracy(acc*100.)
            val_loss_dict[e]=val_loss.result()
            val_accuracy_dict[e] = val_accuracy.result()
            print('Epoch:{:>2} -> loss={:>2.2f}/{:>2.2f} -> Accuracy={:>2.2f}%/{:>2.2f}% - [{:.2f}s]'.format(e,self.train_loss.result(),val_loss.result(),self.train_accuracy.result(),val_accuracy.result(),(time() - inter)))

# model = myForcast(SEQ_LEN, h, d_model, RATE, d_ff, N)
# MyE = myEmbedding(SEQ_LEN, d_model,RATE)
# PE = PositionEmbedding(SEQ_LEN,d_model)


        # Save the training loss values
        with open('./train_loss.pkl', 'wb') as file:
            dump(train_loss_dict, file)
        with open('./train_accuracy.pkl', 'wb') as file:
            dump(train_accuracy_dict, file)
        with open('./val_loss.pkl', 'wb') as file:
            dump(val_loss_dict, file)
        with open('./val_accuracy.pkl', 'wb') as file:
            dump(val_accuracy_dict, file)
        print("Total time taken: {:.1f}s".format(time() - start_time))

        return self.model
