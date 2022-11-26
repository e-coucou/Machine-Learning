import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from keras.datasets import mnist

class relu():
    @staticmethod
    def activation(_X):
        y = np.maximum(_X,0)
        return y
    def prime(_y):
        _y[_y<=0]=0
        _y[_y>0]=1
        return _y
        
class init():
    #staticmethod
    def rand(_in, _out):
        W = np.random.randn(_in, _out) * np.sqrt(1/_out)
        b = np.zeros((1,_out))
        return W, b

class layer():
    class dense():
        def __init__(self,_in,_out, _activation):
            self.W , self.b = init.rand(_in, _out)
            self.a = _activation
        
        def forward(self,_X):
            self.X = _X
            z = np.dot(self.X,self.W) + self.b
            return z

        def backward(self):
            return self.X.T

class sequentiel():
    def __init__(self):
        self.layers = []
        self.activations = []

    def add(self,_layer):
        self.layers.append(_layer)
        self.activations.append(self.layers[-1].a)

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train.reshape(-1,28*28).T / 255.0
    # X_test = X_test.reshape(-1,28*28).T /255.0

    # print(X_train.shape)
    # img = X_train[0].reshape(28,28)
    # plt.imshow(img,cmap='gray')
    # plt.show()
    model = sequentiel()
    model.add(layer.dense(5,10,'relu'))

    return(0)


if __name__ == '__main__':

    sys.exit(main())