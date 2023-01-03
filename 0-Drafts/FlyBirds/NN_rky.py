import numpy as np
import h5py
import pickle
import math
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sn
from keras.datasets import mnist
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm as ProgressDisplay

def plot_image(x,cm='binary', figsize=(4,4),interpolation='lanczos'):
    """
    Draw a single image.
    Image shape can be (lx,ly), (lx,ly,1) or (lx,ly,n)
    args:
        x       : image as np array
        cm      : color map ('binary')
        figsize : fig size (4,4)
    """
    xx=x
    # ---- Draw it
    plt.figure(figsize=figsize)
    plt.imshow(xx,   cmap = cm, interpolation=interpolation)
    plt.show()

def plot_mini_images(x, y, y_pred,indices,c,**kwargs):
    """
    Draw une série d'images de x en fonction la liste indices en affichant la valeur réelle y et y_pred
    """
    cm = 'binary'
    interpolation = kwargs.get('interpolation','lanczos')
    norm = kwargs.get('norm',None)
    fontsize = kwargs.get('fontsize',8)
    n=1
    r=int(len(indices)//c+1)
    fig = plt.figure(figsize=(c,r*1.2))
    for i in indices:
        ax = fig.add_subplot(r,c,n)
        ax.set_yticks([])
        ax.set_xticks([])
        n += 1
        img=ax.imshow(x[i],   cmap = cm, norm=norm, interpolation=interpolation)
        ax.set_xlabel(f'{y_pred[i]} ({y[i]})',fontsize=fontsize)
    plt.show()

EPSILON = 1e-12

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
        # return [1]*len(y)
        return np.ones(y.shape)
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
        # e_x = np.exp(x).reshape(-1,1)
        e_x = np.exp(x - np.max(x,axis=-1).reshape(-1,1))
        return e_x / e_x.sum(axis=-1).reshape(-1,1)
    @staticmethod
    def primeP(y): #marche pas
        n = y.shape[1]
        print(n,y.shape)
        i = np.identity(n)
        un = np.ones((n,y.shape[1]))
        SM = y.reshape((-1,1))
        print(SM.shape)
        jac = np.diagflat(y) - np.dot(SM, SM.T)
        j = np.dot((i* jac),un)
        # x = np.dot(jac,y.T)
        return j.T
    @staticmethod
    def prime(y):
        return(y * (1-y))
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
    @staticmethod
    def uniform(_in, _out):
        # print('uniform',_in,_out)
        W = np.random.uniform(-1,1,(_in, _out)) * np.sqrt(2/_in)
        b = np.zeros((1,_out))
        vdw = np.zeros((_in,_out))
        vdb = np.zeros((1,_out))
        return W, b, vdw, vdb
    @staticmethod
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
        return w,b,0,0
    @staticmethod
    def RMSprop(w,b,dw,db,lr,vdw,vdb,_):
        gamma = 0.9
        vdw = gamma * vdw + (1 - gamma) * dw ** 2
        vdb = gamma * vdb + (1 - gamma) * db ** 2
        w -= dw * lr / (np.sqrt(vdw + 1e-08))
        b -= db * lr / (np.sqrt(vdb + 1e-08))
        return w,b,vdw, vdb
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
        return w,b,vdw,vdb

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
            self.W, self.b, self.vdw, self.vdb = eval('optimiseur.'+_optimiseur)(self.W,self.b,dw,db,lr,self.vdw,self.vdb,epoch)

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
    def __init__(self, a, y_true , _m, **kwargs):
        self.p = a / (np.sum(a,axis=-1,keepdims=True)+EPSILON)
        self.y = y_true
        self.m = _m
        self.sw = kwargs.get('sample_weight',1)
    def normalized(self,a,y_true):
        self.p = a / (np.sum(a,axis=-1,keepdims=True)+EPSILON)
        self.y = y_true
    def metrics(self): # -y.log(p)
        return (1 / self.m) * np.sum(self.forward())
    def forward(self): # -y.log(p)
        return (-self.y * np.log(self.p+EPSILON)) * self.sw
    def backward(self):
        # -y/p
        return (-self.y/(self.p+EPSILON)) * self.sw
class MSE():
    def __init__(self, p, y, _m, **kwargs):
        self.p = p
        self.y = y
        self.m = _m
        self.sw = kwargs.get('sample_weight',1)
    def normalized(self,a,y):
        self.p = a
        self.y = y
        self.c = self.y.shape[-1] #nb de class
    def forward(self):
        return (np.square(self.p - self.y)) / self.c * self.sw
    def backward(self):
        return (2*(self.p - self.y)) * self.sw
    def metrics(self):
        return (1 / self.m) * (np.sum(self.forward()))
        # return 1/self.m *np.sum(np.square(self.p - self.y))
class BinaryCrossEntropy:
    def __init__(self, p, y,_m,**kwargs):
        self.p = p
        self.y = y
        self.m = _m
        self.sw = kwargs.get('sample_weight',1)

    def normalized(self,a,y):
        self.p = a
        self.y = y
        self.c = self.y.shape[-1] #nb de class
    def metrics(self):
        r = (1 / self.m) * np.sum(self.forward()) / self.c
        return r
    def forward(self):
    # (-y.log(p) - (1-y).log(1-p))
        f = ((-self.y * np.log(self.p + EPSILON)) - ((1 - self.y) * (np.log(1 - self.p + EPSILON)))) * self.sw
        return f
    def backward(self):
    # -y/p + (1-y)/(1-p)
        # n = self.p.shape[0]
        return (-self.y/(self.p + EPSILON) + (1 - self.y)/(1 - self.p + EPSILON)) * self.sw 
class ppo:
    def __init__(self,y_true, y_pred):
        self.action_space = 6
        self.p = y_pred
        self.advantages, self.prediction_picks, self.actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        self.LOSS_CLIPPING = 0.2
        self.ENTROPY_LOSS = 5e-3
        
    def forward(self):
    # Defined in https://arxiv.org/abs/1707.06347

        prob = self.p * self.actions
        old_prob = self.actions * self.prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * self.advantages
        p2 = np.clip(r, a_min=1 - self.LOSS_CLIPPING, a_max=1 + self.LOSS_CLIPPING) * self.advantages
        loss =  -np.mean(np.minimum(p1, p2) + self.ENTROPY_LOSS * -(prob * np.log(prob + 1e-10)))
        return loss

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
        sample_weight = kwargs.get('sample_weight',None)
        self.loss, self.accuracy = [], []
        m = _X.shape[0]
        nb = m // batch_size
        a, y  = [], []
        lossFCT = eval(self.lossFCT)(a,y,m,sample_weight=sample_weight)
        loss=0
        # if len(sample_weight)!=0: _y *= sample_weight
        for idx in ProgressDisplay(range(epochs)):
            # print('idx',idx)
            loss=0
            for b in range(nb):
                k=b*batch_size
                l=k+batch_size
                a = _X[k:l]
                y = _y[k:l]
                if (sample_weight is not None):
                    sw = sample_weight[k:l]
                    if shuffle:
                        a, y , sw = Shuffle(a, y, sw)
                    lossFCT.sw = sw
                else:
                    lossFCT.sw = 1
                    if shuffle:
                        a, y = Shuffle(a, y)
                _A, _Z = [], []
                _A.append(a)
                # forward -----
                for n, layer in enumerate(self.layers):
                    # print('layer',n)
                    z = layer.forward(a)
                    a = eval(layer.a).activation(z)
                    _A.append(a)
                    _Z.append(z)
                # loss init backward
                lossFCT.normalized(a,y)
                loss_id = lossFCT.metrics()
                loss += loss_id
                # print(self.lossFCT, loss)
                dL = lossFCT.backward()
                # if len(sample_weight)!=0: dL *= sw
                # print(a.shape,dL.shape)
                da_dz = eval(self.activations[n]).prime(a)
                diff = (dL * da_dz)
                # diff = a - y
                # backward pass-----
                for i in reversed(range(0,n+1)):
                    if self.layers[i].back:
                        dw = (1 / m) * np.dot(_A[i].T,diff)
                        db = (1 / m) * np.sum(diff,axis=0,keepdims=True)
                        if i>0:
                            diff =np.dot( diff, self.layers[i].W.T) * eval(self.activations[i-1]).prime(_Z[i-1])
                        self.layers[i].update(dw,db,self.lr,self.optimiseur,idx)
            if self.metrics:
                if idx%10:
                    # loss = eval(self.lossFCT)(a,y,m)
                    # cost += lossFCT.metrics()
                    # cost = log_loss(y,a)
                    accuracy = np.mean(np.equal(np.argmax(y, axis=-1), np.argmax(a, axis=-1))) * 100
                    # accuracy = accuracy_score(y.flatten(),a.flatten())
                    self.loss.append(loss)
                    self.accuracy.append(accuracy)
        hist=Hist()
        hist.history['loss']=[loss]
        return hist
    def saveH5(self,fichier):
        h5File = h5py.File(fichier, 'w')
        h5File.create_dataset('dataset_1',data=np.array(self.layers))
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
    def save(self,fichier):
        f =  open(fichier,'wb')
        pickle.dump(self.layers,f)
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
    model.add(layer.flatten(X_train.shape[0],28*28,_activation='none',_initialisation='init.zero'))
    model.add(layer.dense(28*28,100,_activation='relu',_initialisation='init.he'))
    # model.add(layer.dense(50,100,_activation='elu',_initialisation='init.he'))
    # model.add(layer.dense(100,32,_activation='relu',_initialisation='init.he'))
    model.add(layer.dense(100,y.shape[1],_activation='softmax',_initialisation='init.he'))
    model.compile(_lr=0.01,_optimiseur='RMSprop',_loss='BinaryCrossEntropy')

    # y_pred = model.predict(X)
    # print(y,(y_pred>0.5)*1)
    # print(y,y_pred)

    X_train_i = X_train / 255.0
    model.fit(X_train_i,y,batch_size=6000,epochs=100,shuffle=True)

    y_pred = model.predict(X_test/255.)
    yp = np.argmax(y_pred,axis=-1)
    print(yp.shape,X_test.shape,yt.shape)
    y_error = [i for i in range(len(X_test)) if yp[i]!=y_test[i]]
    print(yp[:10],y_test[:10],y_error[:10],len(y_error))
    plot_mini_images(X_test,y_test,yp,y_error[:144],18)
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
    Actor.add(layer.dense(d,512,_activation='elu',_initialisation='init.he'))
    Actor.add(layer.dense(512,action_space,_activation='softmax',_initialisation='init.he'))
    Actor.compile(_lr=lr,_optimiseur='RMSprop',_loss='CategoricalCrossEntropy')

    Critic = sequentiel(metrics=True)
    Critic.add(layer.flatten(input_shape,d,_activation='none',_initialisation='init.zero'))
    Critic.add(layer.dense(d,512,_activation='elu',_initialisation='init.he'))
    Critic.add(layer.dense(512,1,_activation='linear',_initialisation='init.he'))
    Critic.compile(_lr=lr,_optimiseur='RMSprop',_loss='MSE')

    return Actor, Critic

if __name__ == '__main__':

    sys.exit(main())