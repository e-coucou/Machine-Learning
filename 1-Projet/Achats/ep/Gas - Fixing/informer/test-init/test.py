import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3
sys.path.insert(0, '..')

import numpy  as np
import tensorflow as tf
import math
import pandas as pd

from model.data import getDataETTh1,Normalize
from model.model import myInformer
from model.exp import Exp

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


# EXP
d_model = 512
HEADS = 8
RATE = 0.05
SEQ_LEN = 96
FACTOR = 5
BATCH_SIZE = 32
GLOBAL_SIZE = 6400
d_ff= 2048
N=2
PRED_LEN =24
FEATURES= 7
TIME_FEATURES = 4
EPOCHS = 10
settings = {'batch_size':BATCH_SIZE, 'seq_len':SEQ_LEN, 'global_size':GLOBAL_SIZE, 'd_model':d_model, 'd_ff':d_ff,
            'e_layer':N, 'd_layer':1, 'rate':RATE, 'factor':FACTOR, 'heads':HEADS, 'pred_len':PRED_LEN, 'features':FEATURES, 'timeFeatures':TIME_FEATURES }
# print(settings)
# Creation du model Exp
myExpFct = Exp(settings)
myExpFct.build()
myExpFct._buildModel(0.00001)
model = myExpFct.train(EPOCHS)