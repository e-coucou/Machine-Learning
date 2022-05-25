from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import json

# Serialize data into file:
def saveParam(file_,data_):
    np.save( file_,data_,allow_pickle=True )

# Read data from file:
def readParam(file_):
    return np.load( file_, allow_pickle=True).item()

def forward_propagation(X,param):
    activation = {'A0' : X}
    # print('longueur',len(param))
    C = len(param)//2
    for c in range(1,C+1):
        Z = param['W'+str(c)].dot(activation['A'+str(c-1)]) + param['b'+str(c)]
        activation['A'+str(c)] = 1 / (1 + np.exp(-Z))
    return activation
def predict(X, param):
    activation = forward_propagation(X,param)
    C = len(param)//2
    a = activation['A'+str(C)]
    # return a>0.5
    return a

def toNum(pred_):
    i=0
    ret = -1
    for p in pred_:
        # print(p)
        if(p):
            ret =i
        i+=1
    return ret

def toNumMax(pred_):
    return np.argmax(pred_)

##
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28*28)/ 255.0
X_test = X_test.reshape(-1,28*28) /255.0
# X_test = X_test.T
y=np.zeros((y_train.shape[0],10))
j=0
for i in y_train:
    y[j][i]=1
    j+=1
y_train =y.copy().reshape((-1,10)).T #y_train.shape[0]))
print('y train 0:', y_train.T[0])
# print(y_train)
y=np.zeros((y_test.shape[0],10))
# print(y)
j=0
for i in y_test:
    y[j][i]=1
    j+=1
y_test =y.copy().reshape((-1,10)).T

# prediction

W_param  = readParam('mnist_v1.npy')
# print(W_param)
X_ = 20
Y_ = 20
accuracy=0
for xx in tqdm(range(X_)):
    for yy in range(Y_):
        i = Y_*xx + yy
        # print(i)
        image = X_test[i].reshape((-1,1))
        # image = X_train[i].reshape((-1,1))
        image_predict = predict(image,W_param)
        # pred = toNum(image_predict.T[0])
        pred = toNumMax(image_predict.T[0])
        reel = toNum(y_test.T[i])
        # reel = toNum(y_train.T[i])
        # print('prediction',image_predict.T, pred,reel)
        # print(y_test.T[i])
    # g = 451+i
    # print(g)
    # plt.subplot(g)
        image = image.reshape((28, 28))*255.0
        plt.subplot2grid((X_,Y_),(xx,yy))
        couleur = plt.cm.Reds
        # plt.subplot(220+i+1)
        if (pred == reel):
            couleur=plt.cm.gray_r
            accuracy += 1
        # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=couleur, interpolation="nearest")

        # plt.legend()

print('Accuracy:',accuracy/X_/Y_*100.0)

# print('X_test shape:',X_test.shape,'X_train shape',X_train.shape,'y_test shape', y_test.T[0])
# image = X_test.T[0].reshape((-1,1))
# print('image shape',image.shape)
# print('image predict shape:',image_predict.shape)
plt.show()