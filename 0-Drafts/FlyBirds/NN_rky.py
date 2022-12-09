import numpy as np
import h5py
import math
import sys
import matplotlib.pyplot as plt
# import seaborn as sn
from keras.datasets import mnist
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm as ProgressDisplay

class relu():
    @staticmethod
    def activation(x):
        x[x<0] = 0
        # y = np.maximum(_X,0)
        return x
    @staticmethod
    def prime(y):
        y[y<=0]=0  # type: ignore
        y[y>0]=1  # type: ignore
        return y
class elu():
    @staticmethod
    def activation(x,a=2.):
        return np.where(x <= 0, a*(np.exp(x) - 1), x)
    @staticmethod
    def prime(y,a=2.):
        return np.where(y <= 0, a*np.exp(y), 1)
class prelu():
    a=0.02
    @staticmethod
    def activation(x):
        return np.where(x<0, prelu.a*x, x)
    @staticmethod
    def prime(y):
        return np.where(y<0, prelu.a, 1)
class linear():
    @staticmethod
    def activation(x):
        return x
    @staticmethod
    def prime(y):
        return [1]*len(y)
class sigmoid():
    @staticmethod
    def activation(x):
        y = 1 / (1 + np.exp(-x))
        return y
    @staticmethod
    def prime(y):
        f = sigmoid.activation(y)
        return (f * (1 - f))
class tanh():
    @staticmethod
    def activation(x):
        return np.tanh(x)
    @staticmethod
    def prime(y):
        return (1-np.tanh(y)**2)
class softmax:
    @staticmethod
    def activation(x):
        e_x = np.exp(x - np.max(x,axis=-1).reshape(-1,1)) 
        return e_x / e_x.sum(axis=-1).reshape(-1,1)
class ep:
    @staticmethod
    def activation(_X):
        return (_X>0.5)+0.
class none:
    @staticmethod
    def activation(x):
        return x
    def prime(y):
        return y

class init():
    @staticmethod
    def normal(_in, _out):
        W = np.random.randn(_in, _out) * np.sqrt(1/_in)
        b = np.zeros((1,_out))
        vdw = np.zeros((_in,_out))
        vdb = np.zeros((1,_out))
        return W, b, vdw, vdb
    def uniform(_in, _out):
        # print('uniform',_in,_out)
        W = np.random.uniform(-1,1,(_in, _out)) * np.sqrt(2/_in)
        b = np.zeros((1,_out))
        vdw = np.zeros((_in,_out))
        vdb = np.zeros((1,_out))
        return W, b, vdw, vdb
    def rand(_in, _out):
        W = np.random.randn(_in, _out) * np.sqrt(1/_in)
        b = np.zeros((1,_out))
        vdw = np.zeros((_in,_out))
        vdb = np.zeros((1,_out))
        return W, b, vdw, vdb
    @staticmethod
    def zero(_in, _out):
        W = np.zeros((_in, _out))
        b = np.zeros((1,_out))
        vdw = np.zeros((_in,_out))
        vdb = np.zeros((1,_out))
        return W, b, vdw, vdb
    @staticmethod
    def he(_in, _out):
        W = np.random.randn(_in, _out) * np.sqrt(2/_in)
        b = np.zeros((1,_out))
        vdw = np.zeros((_in,_out))
        vdb = np.zeros((1,_out))
        return W, b, vdw, vdb

class optimiseur():
    @staticmethod
    def GD(w,b,dw,db,lr,_a,_b,_i):
        w -= dw * lr
        b -= db * lr
        return w,b
    @staticmethod
    def RMSprop(w,b,dw,db,lr,vdw,vdb,_):
        gamma = 0.9
        vdw = gamma * vdw + (1 - gamma) * dw ** 2
        vdb = gamma * vdb + (1 - gamma) * db ** 2
        w -= dw * lr / (np.sqrt(vdw + 1e-08))
        b -= db * lr / (np.sqrt(vdb + 1e-08))
        return w,b
    @staticmethod
    def Adam(w,b,dw,db,lr,vdw,vdb,epoch):
        gamma = 0.9
        theta = 0.999
        mdw = gamma * vdw + (1 - gamma) * dw
        mdb = gamma * vdb + (1 - gamma) * db
        vdw = theta * vdw + (1 - theta) * dw ** 2
        vdb = theta * vdb + (1 - theta) * db ** 2
        mdw_corr = mdw / (1 - np.power(gamma, epoch + 1))
        mdb_corr = mdb / (1 - np.power(gamma, epoch + 1))
        vdw_corr = vdw / (1 - np.power(theta, epoch + 1))
        vdb_corr = vdb / (1 - np.power(theta, epoch + 1))
        w -= mdw_corr * lr / (np.sqrt(vdw_corr + 1e-08))
        b -= mdb_corr * lr / (np.sqrt(vdb_corr + 1e-08))
        return w,b

class layer():
    class dense():
        def __init__(self,_in,_out, _activation,_initialisation):
            self.W , self.b, self.vdw, self.vdb = eval(_initialisation)(_in, _out)
            self.a = _activation
            self.back = True
        
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
        
        def update(self, dw, db, lr,  _optimiseur='GD',epoch=1):
            self.W, self.b = eval('optimiseur.'+_optimiseur)(self.W,self.b,dw,db,lr,self.vdw,self.vdb,epoch)

    class flatten():
        def __init__(self,_in,_out,_activation,_initialisation):
            self.W , self.b, self.vdw, self.vdb = [],[],[],[]
            self.a = 'none'
            self.dim = np.prod(_in)
            self.back = False
        def forward(self, _X):
            d=_X.shape[0]
            return _X.reshape(d,-1)
        def predict(self,_X):
            return self.forward(_X)
        def update(self,dw,db,lr,_,epoch=1):
            print('ici ....')

class CategoricalCrossEntropy():
    def __init__(self, a, y_true,_m):
        self.y_predict = a / np.sum(a,axis=-1)
        self.y_true = y_true
        self.m = _m
    def forward(self):
        return (-1 / self.m) * np.sum((self.y_true * np.log(self.y_predict)))
    # def backward(self):
    #     delta = self.y_predict - self.y_true
    #     return delta
class MSE():
    def __init__(self, p, y, _m):
        self.p = p
        self.y = y
        self.m = _m
    def forward(self):
        m = self.p.shape[0]
        return (1 / m) * (np.sum(np.square(self.y - self.p)))
class BinaryCrossEntropy:
    def __init__(self, p, y,_m):
        self.p = p
        self.y = y
        self.m = _m
    def forward(self):
        m = self.p.shape[0]
        return (-1 / m) * (np.sum((self.y * np.log(self.p + 1e-8)) + ((1 - self.y) * (np.log(1 - self.p + 1e-8)))))

    def backward(self):
        delta = self.p - self.y
        return delta

def Shuffle(a,b,c=None):
    p = np.random.permutation(len(a))
    if c==None:
        return a[p], b[p]
    else:
        return a[p], b[p], c[p]

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

    def predict(self,_X,**kwargs):
        y = _X
        for _, layer in enumerate(self.layers):
            y = layer.predict(y)
        return y
    
    def compile(self,_lr,_optimiseur,_loss='BinaryCrossEntropy'):
        self.lr=_lr
        self.optimiseur = _optimiseur
        self.lossFCT = _loss

    def fit(self,_X,_y,batch_size=32,epochs=100, **kwargs):
        # print('- Fit ------')
        shuffle = kwargs.get('shuffle',False)
        sample_weight = kwargs.get('sample_weight',[])
        self.loss, self.accuracy = [], []
        m = _X.shape[0]
        nb = m // batch_size
        a, y = [], []
        for idx in ProgressDisplay(range(epochs)):
            # print('idx',idx)
            for b in range(nb):
                k=b*batch_size
                l=k+batch_size
                a = _X[k:l]
                y = _y[k:l]
                if len(sample_weight)!=0:
                    sw = sample_weight[k:l]
                    if shuffle:
                        a, y ,sw = Shuffle(a, y, sw)
                else:
                    if shuffle:
                        a, y = Shuffle(a, y)
                _A, _Z = [], []
                _A.append(a)
                # forward -----
                for n, layer in enumerate(self.layers):
                    z = layer.forward(a)
                    a = eval(layer.a).activation(z)
                    _A.append(a)
                    _Z.append(z)
                lossFCT=MSE(a,y,0)
                loss= lossFCT.forward()
                diff = (a - y)
                # backward -----
                for i in reversed(range(0,n+1)):
                    if self.layers[i].back:
                        if len(sample_weight)!=0: diff *=sw
                        dw = (1 / m) * np.dot(_A[i].T,diff)
                        db = (1 / m) * np.sum(diff,keepdims=True)
                        if i>0:
                            diff =np.dot( diff, self.layers[i].W.T) * eval(self.activations[i-1]).prime(_Z[i-1])
                        self.layers[i].update(dw,db,self.lr,self.optimiseur,i)
            if self.metrics:
                if idx%10:
                    # loss = eval(self.lossFCT)(a,y,m)
                    # cost = loss.forward()
                    cost = log_loss(y,a)
                    accuracy = np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(a, axis=-1))) * 100
                    # accuracy = accuracy_score(y.flatten(),a.flatten())
                    self.loss.append(cost)
                    self.accuracy.append(accuracy)
        hist=Hist()
        hist.history['loss']=[loss]
        return hist
    def save(self,fichier):
        h5File = h5py.File(fichier, 'w')
        h5File.create_dataset('dataset_1',data=np.array(self.layers,dtype=np.str0))
        h5File.create_dataset('activation',self.activations)
        h5File.create_dataset('loss',self.loss)
        i=0
        for layer in self.layers:
            name = 'layer'+i+'W'
            h5File.create_dataset(name,layer.W)
            name = 'layer'+i+'b'
            h5File.create_dataset(name,layer.b)
            name = 'layer'+i+'vwd'
            h5File.create_dataset(name,layer.vdw)
            name = 'layer'+i+'vdb'
            h5File.create_dataset(name,layer.vdb)
            name = 'layer'+i+'a'
            h5File.create_dataset(name,layer.a)
            name = 'layer'+i+'back'
            h5File.create_dataset(name,layer.back)
            i += 1
        h5File.close()

class Hist():
    def __init__(self):
        self.history = {}

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape,y_train.shape)
    X = X_train.reshape(-1,28*28) / 255.0
    Xt = X_test.reshape(-1,28*28) /255.0
    y = np.eye(10)[y_train]
    yt = np.eye(10)[y_test]

    # print(X_train.shape)
    # img = X_train[0].reshape(28,28)
    # plt.imshow(img,cmap='gray')
    # plt.show()

    # X,y = make_circles(n_samples=300, noise=0.15, factor=0.35, random_state=0)
    # # X,y = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=0)
    # X[:,1] = X[:,1]*1
    # y = y.reshape((y.shape[0],1))

    print('dimension de X: ', X.shape)
    print('dimension de y: ',y.shape)

    # plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')

    model = sequentiel(metrics=True)
    model.add(layer.dense(X.shape[1],100,_activation='prelu',_initialisation='init.uniform'))
    # model.add(layer.dense(50,100,_activation='elu',_initialisation='init.he'))
    # model.add(layer.dense(100,32,_activation='relu',_initialisation='init.he'))
    model.add(layer.dense(100,y.shape[1],_activation='softmax',_initialisation='init.normal'))
    model.compile(_lr=0.006,_optimiseur='RMSprop',_loss='CategoricalCrossEntropy')

    # y_pred = model.predict(X)
    # print(y,(y_pred>0.5)*1)
    # print(y,y_pred)

    model.fit(X,y,batch_size=6000,epochs=100)

    y_pred = model.predict(Xt)
    yp = np.argmax(y_pred,axis=-1)
    # print(y_pred[:10,:],yt[:10,:])
    # print(yp[:10], y_test[:10])
    # print(y,(y_pred>0.5)*1)
    # print(y,y_pred)
    # calcul la limite
    # h = 100
    # W1 = np.linspace(X[:,0].min(),X[:,0].max(),h)
    # W2 = np.linspace(X[:,1].min(),X[:,1].max(),h)
    # # W1 = np.linspace(-2,2,h)
    # # W2 = np.linspace(-2,2,h)
    # W11, W22 = np.meshgrid(W1,W2)
    # # print(W11.shape)
    # W_Final = np.c_[W11.ravel(),W22.ravel()]
    # # print(W_Final.shape)
    # Z = (model.predict(W_Final)).reshape(W11.shape)
    # plt.figure(figsize=(18,4))
    # plt.subplot(131)
    # plt.scatter(X[:,0],X[:,1],c=(y),cmap='RdBu') #cmap='summer')    
    # plt.contour(W11,W22,Z,1, colors=('yellow','blue','red'),alpha=0.5)
    # # plt.colorbar()
    matrix = confusion_matrix(y_test,yp,10)

    print(f'Perte mini : {model.loss[-1]:.5f}')
    print(f'Accuracy   : {model.accuracy[-1]:.5f}')
    # print(matrix)
    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.plot(model.loss,c='y')
    plt.subplot(132)
    plt.plot(model.accuracy,c='y')
    plt.subplot(133)
    plt.imshow(matrix, cmap='GnBu', interpolation='nearest')
    plt.show()

    return(0)

def confusion_matrix(p,y,dim):
    m = np.zeros((dim,dim))
    for i in range(p.shape[0]):
        m[y[i],p[i]] += 1
    return m

def myModel(input_shape, action_space, lr):
    d=np.prod(input_shape)
    Actor = sequentiel(metrics=True)
    Actor.add(layer.flatten(input_shape,d,_activation='none',_initialisation='init.zero'))
    Actor.add(layer.dense(d,512,_activation='elu',_initialisation='init.uniform'))
    Actor.add(layer.dense(512,action_space,_activation='softmax',_initialisation='init.uniform'))
    Actor.compile(_lr=lr,_optimiseur='RMSprop',_loss='CategoricalCrossEntropy')

    Critic = sequentiel(metrics=True)
    Critic.add(layer.flatten(input_shape,d,_activation='none',_initialisation='init.zero'))
    Critic.add(layer.dense(d,512,_activation='elu',_initialisation='init.uniform'))
    Critic.add(layer.dense(512,1,_activation='sigmoid',_initialisation='init.uniform'))
    Critic.compile(_lr=lr,_optimiseur='RMSprop',_loss='BinaryCrossEntropy')

    return Actor, Critic

if __name__ == '__main__':

    sys.exit(main())