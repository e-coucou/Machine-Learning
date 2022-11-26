import numpy as np
import NN as nn

import sys

#---------
def main() -> int:
    np.random.seed(127)
    X = np.random.randn(2,1)
    layers = [3,3]
    y = np.array([0,1,2]).reshape(3,1)

    print(X.shape,y.shape,layers)

    np.random.seed(127)
    NN = nn.NN_ep(X,y,layers)

    np.random.seed(127)
    NN_org = nn.NNs_ep([2,3,3,3])


    # X = np.c_[ X, np.ones((2))]
    # print(X)

    print('Nombre de couche du réseau : ', NN.c)
    for i in range(NN.c):
        print(i,NN.W[i].shape,NN.W[i])

    print(y)
    y_ep = NN.predict(X.T)
    print('y_ep',y_ep)
    print('log Loss : ', NN.log_loss(y_ep,y))
    y_org = NN_org.predict(X.T).reshape(3,1)
    print(y_org)

    print('model de base ----')
    NN_org.fit(X.T,y,epochs=1)

    print('----- Activations -')
    A_ep = NN.forward(X.T)
    for l in range(NN.c+1):
        print(A_ep[l])

    W = NN.backward(A_ep,y)

    return(0)

if __name__ == '__main__':
    sys.exit(main())