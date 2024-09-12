import tensorflow as tf
import numpy as np
from time import time

from model.data import getDataETTh1, Normalize
from model.model import myInformer,accuracy_fcn,loss_fcn

from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD
# from tensorflow.keras.optimizers import AdamW

from pickle import dump

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
        self.timeFeatures = settings["timeFeatures"]
        self.declen = self.seq_len//2

        self.dataFct = getDataETTh1(seq_len=self.seq_len,batch_len=self.batch_size,global_size=self.global_size)
        self.modelFct = myInformer(
                seq_len=self.seq_len, pred_len=self.pred_len,batch_size=self.batch_size,d_model=self.d_model,rate=self.rate,factor=self.factor,head=self.heads,d_ff=self.d_ff,
                e_layer=self.e_layer,features=self.features, d_layer=self.d_layer )

    def _getData(self):
        self.X, self.X_date, self.Yt, self.YtDate, self.df_raw = self.dataFct.get()

    def _formatData(self):
        L, S, F = self.X.shape
        _, _, Fd = self.X_date.shape
        self.Xn,self.scalerX = Normalize(tf.reshape(self.X, [L,-1]))
        self.Xn = tf.reshape(self.Xn,[L,S,F])
        self.Xn = tf.cast(self.Xn,tf.float32)
        self.X_date = tf.cast(self.X_date,tf.float32)

    def _selectData(self, start=0):
        dec_len = self.seq_len // 2
        self.x_enc, self.X_date_enc = [ self.Xn[start:start+self.batch_size,:self.seq_len,:] , self.X_date[start:start+self.batch_size,:self.seq_len,:]]
        self.x_dec, self.X_date_dec = [ self.Xn[start:start+self.batch_size,:dec_len,:] , self.X_date[start:start+self.batch_size,:dec_len,:]]
        x_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.Xn.shape[-1]))
        x_date_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.X_date.shape[-1]))
        self.x_dec = tf.concat([self.x_dec,x_dec_padd],1)
        self.X_date_dec = tf.concat([self.X_date_dec,x_date_dec_padd],1)

    def _pred(self,training=True):
        self.predData, self.outDec, self.outEnc, self.decEmb, self.encEmb, attn = self.modelFct(self.x_enc, self.X_date_enc, self.x_dec, self.X_date_dec, training=training)
        return self.predData
            
    def _build(self):
        self._getData()
        self._formatData()
        n = self.Xn.shape[0] if self.global_size==0 else self.global_size
        self.start = np.random.randint(0,n-self.batch_size)
        self._selectData(start=self.start)

    def getDataset(self, split=0.2):
        self._getData()
        self.dataset, self.dataValset = self.dataFct.buildDataset(split=split)
    
    def pred(self,x_enc, x_dec,X_date_enc, X_date_dec, training=False):
        self.x_enc = x_enc
        self.x_dec = x_dec
        self.X_date_enc = X_date_enc
        self.X_date_dec = X_date_dec
        self.predData, self.outDec, self.outEnc, self.decEmb, self.encEmb, attn = self.modelFct(self.x_enc, self.X_date_enc, self.x_dec, self.X_date_dec, training=training)
        return self.predData
    
    def build(self, split=0.2):
        print('BUILD')
        self.getDataset(split)
        print( self.dataFct.X.shape)
        self.X = self.dataFct.X
        self.y = self.dataFct.y
        self.scaler = self.dataFct.scalerData
        self.X_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.features))
        self.X_date_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.timeFeatures))


    def _buildModel(self, LR=0.01):
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-9
        # Instantiate an Adam optimizer
        # optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
        self.LR=LR
        self.optimizer = Adam(self.LR, self.beta_1, self.beta_2, self.epsilon)

    def _step(self):
        with tf.GradientTape() as tape:
            pred = self._pred(training=True)
            loss = loss_fcn(self.yR, pred)
            accuracy = accuracy_fcn(y_true=self.yR, y_pred=pred)
        # grads = tape.gradient(loss,model.trainable_weights)
        grads = tape.gradient(loss,self.modelFct.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.modelFct.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(accuracy*100.)

    def _step_data(self,dX,dy):
        B,S,F = dX.shape
        self.x_enc = dX[:,:,0:7]
        self.x_dec = self.x_enc[:,0:self.declen,:]
        self.X_date_enc = dX[:,:,7:]
        self.X_date_dec = self.X_date_enc[:,0:self.declen,:]
        self.yR = dX[:,self.declen:self.declen+self.pred_len,0:7]
        # self.ydateR = dy[:,7:]
        x_dec_padd = tf.zeros(shape=(B,self.pred_len,self.features))
        x_date_dec_padd = tf.zeros(shape=(B,self.pred_len,self.timeFeatures))
        self.x_dec = tf.concat([self.x_dec,x_dec_padd],1)
        self.X_date_dec = tf.concat([self.X_date_dec,x_date_dec_padd],1)


    def train(self, epochs):
        # Include metrics monitoring
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy= Mean(name='train_accuracy')
        train_loss_dict = {}
        train_accuracy_dict = {}
        val_loss = Mean(name='val_loss')
        val_accuracy= Mean(name='val_accuracy')
        val_loss_dict = {}
        val_accuracy_dict = {}
        #start Training
        start_time = time()
        for e in range(epochs):
            self.train_loss.reset_states()
            inter=time()
            for i,(btX,btY) in enumerate(self.dataset):
                self._step_data(btX,btY)
                # btY = tf.reshape(btY,shape=[btY.shape[0],1])
                # btX = tf.reshape(btX, shape=[btX.shape[0],self.seq_len,1])
                self._step()
            train_loss_dict[e] = self.train_loss.result()
            train_accuracy_dict[e] = self.train_accuracy.result()

            # for i,(btX,btY) in enumerate(self.datasetVal):
            #     btY = tf.reshape(btY,shape=[btY.shape[0],1])
            #     btX = tf.reshape(btX, shape=[btX.shape[0],self.seq_len,1])
            #     pred = self.modelF(btX)
            #     loss = loss_fcn(btY, pred)
            #     acc = accuracy_fcn(btY, pred)
            #     val_loss(loss)
            #     val_accuracy(acc*100.)
            # val_loss_dict[e]=val_loss.result()
            # val_accuracy_dict[e] = val_accuracy.result()
            print('Epoch:{:>2} -> loss={:>2.2f}/{:>2.2f} -> Accuracy={:>2.2f}%/{:>2.2f}% - [{:.2f}s]'.format(e,self.train_loss.result(),val_loss.result(),self.train_accuracy.result(),val_accuracy.result(),(time() - inter)))

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

        return self.modelFct
