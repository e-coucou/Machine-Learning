from keras import Sequential, Model
import numpy as np
from keras import Input
from keras import layers
from tensorflow.keras.optimizers import RMSprop

import sys
sys.path.append('..')

import ep.etools as apt

import yfinance as yf
etf = ['BZ=F','EURUSD=X']
data = yf.Ticker(etf[0])
dataH = data.history(start='2007-01-01',period='1d')
it = yf.download(tickers=etf, period='1d',start='2009-01-01')
# print(it.describe())
# supprime colonnes inutiles
it.drop(['High','Low','Open','Volume','Adj Close'],axis=1,inplace=True)
print(it.columns.to_list())
# it.rename(columns='_'.join,inplace=True)
it.columns = it.columns.map('_'.join)
it.rename(columns={'Close_BZ=F':'brent','Close_EURUSD=X':'FX'},inplace=True)
# it.describe()
it.dropna(axis=0,inplace=True)

import matplotlib.pyplot as plt
it_plot = it.copy()
# normalisation
it_mean = it.mean(axis=0)
it_std = it.std(axis=0)
it_plot = (it_plot - it_mean) /it_std
# # plot
# plt.figure(figsize=(18,10))
# plt.xlabel('Date')
# plt.ylabel('cours en $')
# plt.plot(it_plot,label=it_plot.columns.to_list())
# plt.legend(loc='upper left')
# plt.show()

# parametres
scale            = 1        # % du dataset (1=all)
train_prop       = .8       # ration du train vs test
sequence_len     = 16
batch_size       = 64
epochs           = 15
features         = ['brent','FX']
features_len     = len(features)
# Mise à l'échelle du dataFrame

df = it[:int(scale*len(it))].reset_index()
train_len=int(train_prop*len(df))

print(df)

# ---- Train / Test
dataset_train = df.loc[ :train_len-1, features ]
dataset_test  = df.loc[train_len:,    features ]
apt.subtitle('Train dataset :')
# display(dataset_train.head(15))

# ---- Normalize, and convert to numpy array

mean = dataset_train.mean()
std  = dataset_train.std()
dataset_train = (dataset_train - mean) / std
dataset_test  = (dataset_test  - mean) / std

apt.subtitle('Après normalisation :')
# display(dataset_train.describe().style.format("{0:.2f}"))

dataset_train = dataset_train.to_numpy()
dataset_test  = dataset_test.to_numpy()

apt.subtitle('Shapes :')
print('Dataset       : ',df.shape)
print('Train dataset : ',dataset_train.shape)
print('Test  dataset : ',dataset_test.shape)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

print(tf.config.get_visible_devices())

# ---- Train generator

train_generator = TimeseriesGenerator(dataset_train, dataset_train, length=sequence_len,  batch_size=batch_size)
test_generator  = TimeseriesGenerator(dataset_test,  dataset_test,  length=sequence_len,  batch_size=batch_size)

# ---- About

apt.subtitle('About the splitting of our dataset :')

x,y=train_generator[0]
print(f'Nombre de train batchs disponibles : ', len(train_generator))
print('batch x shape : ',x.shape)
print('batch y shape : ',y.shape)

x,y=train_generator[0]
apt.subtitle('What a batch looks like (x[0]) :')
apt.np_print(x[0] )
apt.subtitle('What a batch looks like (y[0]) :')
apt.np_print(y[0])

run_dir = './run/'
save_dir = f'{run_dir}/best_model.h5'
bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, save_best_only=True)

# le model Keras
input_data = Input(shape=(sequence_len,features_len))
# x = layers.GRU(100, dropout=0.2,return_sequences=True, recurrent_dropout=0.5)(input_data)
x = layers.LSTM(100,activation='relu')(input_data)
x = layers.Dropout(0.2)(x)
x = layers.Dense(100,activation='relu')(x)
x = layers.Dropout(0.2)(x)
# x = layers.GRU(100, dropout=0.2,recurrent_dropout=0.5,activation='relu')(x)
# x = layers.Dense(100,activation='relu')(x)
# x = layers.Dense(32,activation='relu')(x)
output_data = layers.Dense(features_len)(x)
# with tf.device("/cpu:0"):
model = Model(input_data,output_data)
    # model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['mae'])
model.compile(optimizer=RMSprop(),loss='mse',metrics=['mae'])
model.summary()
history = model.fit(train_generator,epochs=epochs,validation_data=test_generator,verbose=1,use_multiprocessing=False,callbacks = [bestmodel_callback])

apt.plot_history(history,plot={'loss':['loss','val_loss'], 'mae':['mae','val_mae']}, save_as='01-history')

import math,random
def denormalize(mean,std,seq):
    nseq = seq.copy()
    for i,s in enumerate(nseq):
        s = s*std + mean
        nseq[i]=s
    return nseq


def get_prediction(dataset, model, iterations=4,sequence_len=16):
    # ---- Initial sequence
    s=random.randint(0,len(dataset)-sequence_len-iterations)
    sequence_pred = dataset[s:s+sequence_len].copy()
    sequence_true = dataset[s:s+sequence_len+iterations].copy()
    # ---- Iterate
    sequence_pred=list(sequence_pred)
    for i in range(iterations):
        sequence=sequence_pred[-sequence_len:]
        pred = model.predict( np.array([sequence]) )
        sequence_pred.append(pred[0])
    # ---- Extract the predictions    
    pred=np.array(sequence_pred[-iterations:])
    # ---- De-normalization
    sequence_true = denormalize(mean,std, sequence_true)
    pred          = denormalize(mean,std, pred)
    return sequence_true,pred

best_model = tf.keras.models.load_model(f'{run_dir}/best_model.h5')
sequence_true, pred = get_prediction(dataset_test, best_model,iterations=5,sequence_len=16)
feat=[0,1]
apt.plot_multivariate_serie(sequence_true, predictions=pred, labels=features,
                            only_features=feat,width=14, height=8)
