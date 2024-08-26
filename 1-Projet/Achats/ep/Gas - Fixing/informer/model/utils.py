from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import tensorflow as tf


# Normalize - conver2diff - diff2org
def Normalize(d):
    min_max_scaler = MinMaxScaler((-1,1))
    d = tf.reshape(d, shape=[-1,1])
    scaler = min_max_scaler.fit(d)
    data = scaler.transform(d)
    return data, min_max_scaler

def conver2diff(d, size, norm=False):
    all = pd.Series(tf.reshape(d,shape=[size]))
    initDiff = all[0]
    allDiff = all.diff()
    allDiff[0]=0.
    xDiff = tf.convert_to_tensor(allDiff, dtype=tf.float32)
    if (norm):
        xDiff, scalerDiff = Normalize(xDiff)
    else:
        scalerDiff = None
        xDiff = xDiff.numpy()
    return xDiff, scalerDiff, initDiff

def diff2org(d, size, scalerDiff,xOrigine):
    if (scalerDiff==None):
        all_N = pd.Series(tf.reshape(d,shape=[size])).to_numpy()
        all_ = all_N.copy()
        all_[0]=xOrigine
        return tf.convert_to_tensor(all_.cumsum())
    else:
        xDiff = scalerDiff.inverse_transform(d)
        all_ = pd.Series(tf.reshape(xDiff,shape=[size])).to_numpy(dtype=np.float32)
        all_[0]=xOrigine
        return tf.convert_to_tensor(all_.cumsum())
    
def padding_mask(input):
    # Create mask which marks the zero padding values in the input by a 1.0
    mask = math.equal(input, 0)
    print(mask)
    mask = cast(mask, float32)
    print(mask)
    # The shape of the mask should be broadcastable to the shape
    # of the attention weights that it will be masking later on
    return mask[:, newaxis, newaxis, :]

def lookahead_mask(shape):
    # Mask out future entries by marking them with a 1.0
    mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)
    return mask*(-1e9)