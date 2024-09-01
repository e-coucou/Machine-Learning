import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model.utils import Normalize


class getMyData :
    def __init__(self, data_size, seq_len, batch_len, **kwargs):
        super(getMyData, self).__init__(**kwargs)
        self.data_size=data_size
        self.seq_len = seq_len
        self.batch_len = batch_len
        # Sinusoide de TEST
        x = np.sin(np.linspace(0, 0.1*self.data_size, self.data_size))
        x_random = np.random.normal(0, 0.05, self.data_size) # add noise
        x_rampe = np.linspace(0,0.1*self.data_size,self.data_size)/self.data_size*10 # add rampe
        x = x+x_random + x_rampe
        x,scaler = Normalize(x)
        # x,scaler,xOrg = conver2diff(x, self.data_size, False)
        # x = Norm(x)
        # Create a dataset
        X = np.array([x[ii:ii+self.seq_len] for ii in range(0, x.shape[0]-self.seq_len)], dtype=np.float32)
        Y = np.array([x[ii+self.seq_len] for ii in range(0, x.shape[0]-self.seq_len)], dtype=np.float32)
        self.X = X.astype('float32')
        self.Y = Y.astype('float32')

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=369, shuffle=True)#, stratify=Y)

        self.dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        self.dataset = self.dataset.batch(self.batch_len)
        self.dataset = self.dataset.shuffle(self.batch_len)
        self.datasetVal = tf.data.Dataset.from_tensor_slices((X_val,y_val))
        self.datasetVal = self.datasetVal.batch(self.batch_len)
        self.datasetVal = self.datasetVal.shuffle(self.batch_len)

    def get(self):
        return self.dataset, self.datasetVal
        

    def display(self):
        fig = plt.figure(figsize=(18,3))
        plt.plot(self.Y);
