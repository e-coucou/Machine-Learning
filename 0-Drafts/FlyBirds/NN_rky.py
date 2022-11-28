import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.datasets import make_blobs, make_circles
from tqdm import tqdm as ProgressDisplay

class relu():
    @staticmethod
    def activation(_X):
        y = np.maximum(_X,0)
        return y
    @staticmethod
    def prime(_y):
        _y[_y<=0]=0  # type: ignore
        _y[_y>0]=1  # type: ignore
        return _y
class sigmoid():
    @staticmethod
    def activation(_X):
        y = 1 / (1 + np.exp(-_X))
        return y
    @staticmethod
    def prime(_X):
        f = sigmoid.activation(_X)
        y = f * (1 - f)
        return y
class softmax:
    @staticmethod
    def activation(_X):
        e_x = np.exp(_X - np.max(_X,axis=-1).reshape(-1,1)) 
        return e_x / e_x.sum(axis=-1).reshape(-1,1)
class ep:
    @staticmethod
    def activation(_X):
        return (_X>0.5)+0.

class init():
    @staticmethod
    def rand(_in, _out):
        W = np.random.randn(_in, _out) * np.sqrt(1/_in)
        b = np.zeros((1,_out))
        return W, b
    @staticmethod
    def zero(_in, _out):
        W = np.zeros((_in, _out))
        b = np.zeros((1,_out))
        return W, b
    @staticmethod
    def he(_in, _out):
        W = np.random.randn(_in, _out) * np.sqrt(2/_in)
        b = np.zeros((1,_out))
        return W, b

class optimiseur():
    @staticmethod
    def GD(w,b,dw,db,lr):
        w -= dw * lr
        b -= db * lr
        return w,b

class layer():
    class dense():
        def __init__(self,_in,_out, _activation,_initialisation):
            self.W , self.b = eval(_initialisation)(_in, _out)
            self.a = _activation
        
        def forward(self,_X):
            self.X = _X
            z = np.dot(self.X,self.W) + self.b
            return z

        def backward(self):
            return self.X.T

        def predict(self,_X):
            z= self.forward(_X)
            y = eval(self.a).activation(z)
            return y
        
        def update(self, dw, db, lr,  _optimiseur='GD'):
            self.W, self.b = eval('optimiseur.'+_optimiseur)(self.W,self.b,dw,db,lr)

class CategoricalCrossEntropy():
    def __init__(self, a, y_true):
        self.y_predict = a / np.sum(a,axis=-1)
        self.y_true = y_true
    def forward(self):
        m = self.y_predict.shape[0]
        return (-1 / m) * np.sum((self.y_true * np.log(self.y_predict)))
    # def backward(self):
    #     delta = self.y_predict - self.y_true
    #     return delta
class BinaryCrossEntropy:
    def __init__(self, z, y_true):
        self.y_predict = z
        self.y_true = y_true

    def forward(self):
        m = self.y_predict.shape[0]
        return (-1 / m) * (np.sum((self.y_true * np.log(self.y_predict + 1e-8)) + ((1 - self.y_true) * (np.log(1 - self.y_predict + 1e-8)))))

    def backward(self):
        delta = self.y_predict - self.y_true
        return delta

class sequentiel():
    def __init__(self, **kwargs):
        self.layers = []
        self.activations = []
        self.loss = []
        self.accuracy = []
        self.metrics = kwargs.get('metrics',False)

    def add(self,_layer):
        self.layers.append(_layer)
        self.activations.append(self.layers[-1].a)

    def predict(self,_X):
        y = _X
        for _, layer in enumerate(self.layers):
            y = layer.predict(y)
        return y
    
    def compile(self,_lr,_optimiseur,_loss='BinaryCrossEntropy'):
        self.lr=_lr
        self.optimiseur = _optimiseur
        self.lossFCT = _loss

    def fit(self,_X,_y,_batch=32,_epoch=100):
        print('- Fit ------')
        self.loss, self.accuracy = [], []
        m = _X.shape[0]
        nb = m // _batch
        for idx in ProgressDisplay(range(_epoch)):
            for b in range(nb):
                k=b*_batch
                l=k+_batch
                a = _X[k:l]
                y = _y[k:l]
                _A, _Z = [], []
                #forward
                for n, layer in enumerate(self.layers):
                    z = layer.forward(a)
                    a = eval(layer.a).activation(z)
                    _A.append(a)
                    _Z.append(z)
                diff = a - y
                if self.metrics:
                    loss = eval(self.lossFCT)(a,y)
                    cost = loss.forward()
                    accuracy = np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(a, axis=-1))) * 100
                    if idx%10:
                        self.loss.append(cost)
                        self.accuracy.append(accuracy)
                #backward
                for i in reversed(range(1,n+1)):
                    # print(_A[i].shape, _A[i-1].shape, diff.shape)
                    dw = (1 / _batch) * np.dot(_A[i-1].T,diff)
                    db = (1 / _batch) * np.sum(diff)
                    diff =np.dot( diff, self.layers[i].W.T) * eval(self.activations[i-1]).prime(_Z[i])
                    self.layers[i].update(dw,db,self.lr,self.optimiseur)

def main():
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train.reshape(-1,28*28).T / 255.0
    # X_test = X_test.reshape(-1,28*28).T /255.0

    # print(X_train.shape)
    # img = X_train[0].reshape(28,28)
    # plt.imshow(img,cmap='gray')
    # plt.show()

    X,y = make_circles(n_samples=300, noise=0.15, factor=0.35, random_state=0)
    # X,y = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=0)
    X[:,1] = X[:,1]*1
    y = y.reshape((y.shape[0],1))

    print('dimension de X: ', X.shape)
    print('dimension de y: ',y.shape)

    # plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')    

    model = sequentiel(metrics=True)
    model.add(layer.dense(X.shape[1],32,_activation='sigmoid',_initialisation='init.rand'))
    # model.add(layer.dense(64,16,_activation='relu',_initialisation='init.he'))
    # model.add(layer.dense(256,64,_activation='relu',_initialisation='init.he'))
    model.add(layer.dense(32,y.shape[1],_activation='sigmoid',_initialisation='init.rand'))
    model.compile(_lr=0.1,_optimiseur='GD',_loss='BinaryCrossEntropy')

    # y_pred = model.predict(X)
    # print(y,(y_pred>0.5)*1)
    # print(y,y_pred)

    model.fit(X,y,_batch=32,_epoch=10000)

    y_pred = model.predict(X)
    # print(y,(y_pred>0.5)*1)
    # print(y,y_pred)
    # calcul la limite
    h = 100
    W1 = np.linspace(X[:,0].min(),X[:,0].max(),h)
    W2 = np.linspace(X[:,1].min(),X[:,1].max(),h)
    # W1 = np.linspace(-2,2,h)
    # W2 = np.linspace(-2,2,h)
    W11, W22 = np.meshgrid(W1,W2)
    print(W11.shape)
    W_Final = np.c_[W11.ravel(),W22.ravel()]
    print(W_Final.shape)
    Z = (model.predict(W_Final)).reshape(W11.shape)
    plt.figure(figsize=(18,4))
    plt.subplot(131)
    plt.scatter(X[:,0],X[:,1],c=(y),cmap='RdBu') #cmap='summer')    
    plt.contour(W11,W22,Z,1, colors=('yellow','blue','red'),alpha=0.5)
    # plt.colorbar()
    plt.subplot(132)
    plt.plot(model.loss,c='y')
    plt.subplot(133)
    plt.plot(model.accuracy,c='y')
    plt.show()

    return(0)


if __name__ == '__main__':

    sys.exit(main())