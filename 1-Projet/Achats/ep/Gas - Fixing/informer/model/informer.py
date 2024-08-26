import tensorflow as tf
from time import time

from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import Model

from model.model import myForcast,accuracy_fcn,loss_fcn

from pickle import dump

class Informer(Model):
    def __init__(self,seq_len,model, **kwargs):
        super(Informer, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.model = model
        # Define the training parameters
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-9

    def _build_model(self, LR=0.01):
        # Instantiate an Adam optimizer
        # optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
        self.LR=LR
        self.optimizer = Adam(self.LR, self.beta_1, self.beta_2, self.epsilon)

    def step(self,X,y):
        with tf.GradientTape() as tape:
            pred = self.model(X,training=True)
            loss = loss_fcn(y, pred)
            accuracy = accuracy_fcn(y_true=y, y_pred=pred)
        # grads = tape.gradient(loss,model.trainable_weights)
        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(accuracy*100.)

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
