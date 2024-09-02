import sys
sys.path.insert(0, '..')
from model.data import getMyData
from model.embedding import DataEmbedding
from model.encoder import Encoder,ConvLayer
from model.attention import AttentionLayer,ProbAttention

import torch_ep.embed as epEmbed
import torch_ep.encoder as epEncoder
import torch

import numpy  as np
import tensorflow as tf
import math

from tensorflow.keras.layers import Dense
from keras.backend import softmax

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


# Get Features from ETTh1
import pandas as pd
import os
import re
root_path = "../../data/"
data_path = "ETTh1.csv"
df_raw = pd.read_csv(os.path.join(root_path,data_path))

SEQ_LEN = 96
BATCH_SIZE = 32
GLOBAL_SIZE=1000

# split = re.compile('-|:| ')
# featuresDate = (df_raw['date'].str.split(split, expand=True)).astype(np.int64).iloc[:,:4]
featuresData = df_raw.iloc[:,1:8].astype(np.float32)

# On ajoute les infos temporelles : mois/jour/joursemain/heure
df_raw['mois'] = df_raw.date.apply(lambda row:int(row[5:7])).astype(np.int64)
df_raw['jour'] = df_raw.date.apply(lambda row:int(row[8:10])).astype(np.int64)
df_raw['heure'] = df_raw.date.apply(lambda row:int(row[11:13])).astype(np.int64)
df_raw['sJour'] = df_raw['date'].astype('datetime64[s]').dt.dayofweek.astype(np.int64)
featuresDate = df_raw[['mois','jour','sJour','heure']].astype(np.int64)
# featuresDate.describe()

# Convert Features to numpy array
featuresData = featuresData.to_numpy()
featuresDate = featuresDate.to_numpy()
# Create a dataset Feature Data
X = np.array([featuresData[i:i+SEQ_LEN] for i in range(0, featuresData.shape[0]-SEQ_LEN)], dtype=np.float32)
Y = np.array([featuresData[i+SEQ_LEN] for i in range(0, featuresData.shape[0]-SEQ_LEN-1)], dtype=np.float32)
Xt = tf.convert_to_tensor(X[:GLOBAL_SIZE,:,:], dtype=tf.float32)
XT = torch.from_numpy(X[:GLOBAL_SIZE,:,:])

# Create a dataset : Features Date
X = np.array([featuresDate[i:i+SEQ_LEN] for i in range(0, featuresDate.shape[0]-SEQ_LEN)], dtype=np.float32)
Y = np.array([featuresDate[i+SEQ_LEN] for i in range(0, featuresDate.shape[0]-SEQ_LEN-1)], dtype=np.float32)
XtDate = tf.convert_to_tensor(X[:GLOBAL_SIZE,:,:], dtype=tf.float32)
XTDate = torch.from_numpy(X[:GLOBAL_SIZE,:,:])


# Final Test with AttentionLayer
d_model = 512
HEAD = 8
RATE = 0.05
SEQ_LEN = 96
FACTOR = 5

tf.keras.backend.clear_session()

#Embedding
encEmb = DataEmbedding(seq_len=SEQ_LEN, d_model=d_model,rate=RATE)
x = encEmb(x=Xt, x_mark=XtDate, training=True)
print(x.shape)

attLayer = AttentionLayer(ProbAttention(False,FACTOR,None,RATE,False),d_model=d_model,heads=HEAD)
newX, attn = attLayer(x,x,x,None)
print(newX.shape)

fig = plt.figure(figsize=(18,2))
sns.heatmap(x[0,:,:], vmin=-10, vmax=10, cmap=sns.color_palette("hls", 256)).set_title('Context ');
fig = plt.figure(figsize=(18,2))
sns.heatmap(newX[0,:,:], vmin=-10, vmax=10, cmap=sns.color_palette("hls", 256)).set_title('Context ');
