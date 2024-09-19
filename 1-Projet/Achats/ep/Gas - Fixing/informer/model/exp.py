import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time

from model.data import getDataETTh1, Normalize, getGasData
from model.model import myInformer,accuracy_fcn,loss_fcn

from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD
# from tensorflow.keras.optimizers import AdamW

from pickle import dump, load

class Exp():
    def __init__(self, settings, **kwargs):
        super(Exp, self).__init__(**kwargs)
        self.settings=settings
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
        self.freq= settings["freq"]
        self.dataSource = settings["data_source"]
        self.declen = self.seq_len//2

        getData = getDataETTh1 if self.dataSource=='ETTh' else getGasData
        self.dataFct = getData(seq_len=self.seq_len,batch_len=self.batch_size,global_size=self.global_size)
        self.modelFct = myInformer(
                seq_len=self.seq_len, pred_len=self.pred_len,batch_size=self.batch_size,d_model=self.d_model,rate=self.rate,factor=self.factor,head=self.heads,d_ff=self.d_ff,
                e_layer=self.e_layer, freq=self.freq, features=self.features, d_layer=self.d_layer )

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
        # self._getData()  # ca sert à quoi ?
        self.dataset, self.datasetVal = self.dataFct.buildDataset(split=split)
    
    def pred(self,x_enc, x_dec,X_date_enc, X_date_dec, training=False):
        self.x_enc = x_enc
        self.x_dec = x_dec
        self.X_date_enc = X_date_enc
        self.X_date_dec = X_date_dec
        self.predData, self.outDec, self.outEnc, self.decEmb, self.encEmb, attn = self.modelFct(self.x_enc, self.X_date_enc, self.x_dec, self.X_date_dec, training=training)
        return self.predData
    
    def build(self, split=0.2):
        self.getDataset(split)
        self.X = self.dataFct.X
        self.y = self.dataFct.y
        self.X_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.features))
        # self.X_date_dec_padd = tf.zeros(shape=(self.batch_size,self.pred_len,self.timeFeatures))


    def _buildModel(self, LR=0.0001):
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
        grads = tape.gradient(loss,self.modelFct.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        # grads = tape.gradient(loss,self.modelFct.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.modelFct.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(accuracy*100.)

    def _step_data(self,dX,dy):
        B,S,F = dX.shape
        self.x_enc = dX[:,:,0:self.features] # 32 x 96 x 7
        self.x_dec = dy[:,:self.declen,:self.features]   #32 x 48 x 7 les 48 dernières values de x_enc
        self.X_date_enc = dX[:,:,self.features:] # 32 x 96 x 4
        self.X_date_dec = dy[:,:,self.features:] # 32 x 72 x 4
        # self.yR = dX[:,self.declen:self.declen+self.pred_len,0:self.features]
        self.yR = self.x_dec[:,-self.pred_len:,:] # 32 x 24 x 7
        x_dec_padd = tf.zeros(shape=(B,self.pred_len,self.features)) # 32 x 24 x 7 (0)
        # x_date_dec_padd = tf.zeros(shape=(B,self.pred_len,self.timeFeatures))
        self.x_dec = tf.concat([self.x_dec,x_dec_padd],axis=1) # 32 x 72 x 7
        # self.X_date_dec = tf.concat([self.X_date_dec,x_date_dec_padd],axis=1)

    def train(self, epochs):
        # Include metrics monitoring
        self.epochs = epochs
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy= Mean(name='train_accuracy')
        self.train_loss_dict = {}
        self.train_accuracy_dict = {}
        val_loss = Mean(name='val_loss')
        val_accuracy= Mean(name='val_accuracy')
        self.val_loss_dict = {}
        self.val_accuracy_dict = {}
        #start Training
        start_time = time()
        for e in range(epochs):
            self.train_loss.reset_states()
            inter=time()
            for i,(btX,btY) in enumerate(self.dataset):
                self._step_data(btX,btY)
                self._step()
            self.train_loss_dict[e] = self.train_loss.result()
            self.train_accuracy_dict[e] = self.train_accuracy.result()

            for i,(btX,btY) in enumerate(self.datasetVal):
                self._step_data(btX,btY)
                pred = self._pred(training=False)
                loss = loss_fcn(self.yR, pred)
                accuracy = accuracy_fcn(y_true=self.yR, y_pred=pred)
                val_loss(loss)
                val_accuracy(accuracy*100.)
            self.val_loss_dict[e] = val_loss.result()
            self.val_accuracy_dict[e] = val_accuracy.result()
            print('Epoch:{:>2} -> loss={:>2.2f}/{:>2.2f} -> Accuracy={:>2.2f}%/{:>2.2f}% - [{:.2f}s]'.format(e,self.train_loss.result(),val_loss.result(),self.train_accuracy.result(),val_accuracy.result(),(time() - inter)))

        # Save the training loss values
        with open('./train_loss.pkl', 'wb') as file:
            dump(self.train_loss_dict, file)
        with open('./train_accuracy.pkl', 'wb') as file:
            dump(self.train_accuracy_dict, file)
        with open('./val_loss.pkl', 'wb') as file:
            dump(self.val_loss_dict, file)
        with open('./val_accuracy.pkl', 'wb') as file:
            dump(self.val_accuracy_dict, file)
        print("Total time taken: {:.1f}s".format(time() - start_time))

        return self.modelFct
    
    def save_model(self):    
        # SAVE MODEL
        # Create a checkpoint object and manager to manage multiple checkpoints
        ckpt = tf.train.Checkpoint(model=self.modelFct, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)
        # Save a checkpoint after every five epochs
        save_path = ckpt_manager.save()
        print("Saved checkpoint")
        # Save the trained model weights
        self.modelFct.save_weights("weights/wghts" + str(self.epochs) + ".ckpt")
        with open('./scaler/scaler.sav', 'wb') as file:
            dump(self.dataFct.scalerData, file)
        with open('./settings/settings.sav', 'wb') as file:
            dump(self.settings, file)

    def drawTrain(self):
        g_epochs = range(0, self.epochs)
        # Dictionary's values
        train_values = self.train_loss_dict.values()
        train_acc_values = self.train_accuracy_dict.values()

        fig, ax1 = plt.subplots(figsize=(18,3))
        ax2 = ax1.twinx()
        # Plot and label the training loss/accuracy values
        ax1.plot(g_epochs, train_values, label='Training Loss', color='red')
        ax2.plot(g_epochs, train_acc_values, label='Training Accuracy', color='blue')
        
        # Title, Axes labels
        plt.title('/# Training Loss/Accuracy #/')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss',color='red')
        ax2.set_ylabel('Training Accuracy',color='blue')
        # Set tick locations
        plt.xticks(np.arange(0, self.epochs+1, 2))
        # Display plot
        plt.legend(loc='best');
        # plt.show()        

    def drawValid(self):
        g_epochs = range(0, self.epochs)
        # Dictionary's values
        valid_values = self.train_loss_dict.values()
        valid_acc_values = self.train_accuracy_dict.values()

        fig, ax1 = plt.subplots(figsize=(18,3))
        ax2 = ax1.twinx()
        # Plot and label the training loss/accuracy values
        ax1.plot(g_epochs, valid_values, label='Validation Loss', color='red')
        ax2.plot(g_epochs, valid_acc_values, label='Validation Accuracy', color='blue')
        
        # Title, Axes labels
        plt.title('/# Validation Loss/Accuracy #/')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Validation Loss',color='red')
        ax2.set_ylabel('Validation Accuracy',color='blue')
        # Set tick locations
        plt.xticks(np.arange(0, self.epochs+1, 2))
        # Display plot
        plt.legend(loc='best');
        # plt.show()        
