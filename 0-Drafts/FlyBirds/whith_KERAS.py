from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop, gradient_descent_v2,SGD
from keras import backend as K
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles


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

    Ain = Input(X.shape[1])
    A = Dense(32,activation='ReLU')(Ain)
    Aout = Dense(1,activation='sigmoid')(A)
    model = Model(inputs=Ain, outputs=Aout)
    sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    print(model.summary())

    history = model.fit(x=X,y=y,batch_size=32,epochs=1000,verbose=0)
    print(history.history)

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
    plt.plot(history.history['loss'],c='y')
    # plt.subplot(133)
    # plt.plot(model.accuracy,c='y')
    plt.show()

    return(0)


if __name__ == '__main__':

    sys.exit(main())