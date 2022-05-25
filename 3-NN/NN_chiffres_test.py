import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import json

# Serialize data into file:
def saveParam(file_,data_):
    np.save( file_,data_ ,allow_pickle=True)

# Read data from file:
def readParam(file_):
    return np.load( file_)

#-----------
def init(dim):
    C= len(dim)
    param={}
    for c in range(1,C):
        param['W'+str(c)] = np.random.randn(dim[c],dim[c-1])
        param['b'+str(c)] = np.random.randn(dim[c],1)
    return param
def forward_propagation(X,param):
    activation = {'A0' : X}
    C = len(param)//2
    for c in range(1,C+1):
        Z = param['W'+str(c)].dot(activation['A'+str(c-1)]) + param['b'+str(c)]
        activation['A'+str(c)] = 1 / (1 + np.exp(-Z))
    return activation
def contour(param):
    h = 100
    W1 = np.linspace(X[0].min(),X[0].max(),h)
    W2 = np.linspace(X[1].min(),X[1].max(),h)
    # W3 = np.linspace(X[2].min(),X[2].max(),h)
    W11, W22 = np.meshgrid(W1,W2)

    W_Final = np.c_[W11.ravel(),W22.ravel()].T

    Z = (predict(W_Final,param)+0).reshape(W11.shape)

    return W11,W22,Z
def back_propagation(y,activation,param ):
    m = y.shape[1]
    C = len(param)//2
    gradients = {}
    dZ = activation['A'+str(C)] - y

    for c in reversed(range(1,C+1)):
        gradients['dW'+str(c)] = 1/m * dZ.dot(activation['A'+str(c-1)].T)
        gradients['db'+str(c)] = 1/m * np.sum(dZ, axis=1,keepdims=True)
        if c>1:
            dZ = np.dot(param['W'+str(c)].T,dZ) * activation['A'+str(c-1)] * (1 - activation['A'+str(c-1)])

    return gradients
def update(gradients,param,lr):

    C = len(param)//2
    for c in range(1,C):
        param['W'+str(c)] = param['W'+str(c)] - lr* gradients['dW'+str(c)]
        param['b'+str(c)] = param['b'+str(c)] - lr* gradients['db'+str(c)]

    return param
def predict(X, param):
    activation = forward_propagation(X,param)
    C = len(param)//2
    a = activation['A'+str(C)]
    return a>0.5
    # return a
def neural_network(X,y,hidden_layer,lr=0.1,n_iter=1000): #n1 nombre d neurones de la couche 1
    #init W,b
    dimensions=list(hidden_layer)
    dimensions.insert(0,X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(0)
    param = init(dimensions)
    
    history = []
    train_Loss = []
    train_acc = []

    
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X,param)
        gradients = back_propagation( y, activations, param)
        param = update(gradients, param, lr)
        # Err.append(log_Error(a,y))
        if (i>550):
            lr = 0.05
        if i%25 ==0:
            C =len(param)//2
            train_Loss.append(log_loss(y,activations['A'+str(C)]))
            y_pred = predict(X,param)
            current_accuracy = accuracy_score(y.flatten(),y_pred.flatten())
            train_acc.append(current_accuracy)
        #     # history.append([param.copy(), train_Loss, train_acc, i])
        
    
    # W11,W22,Z = contour(param)
    # print(W11.shape,Z)
    plt.figure(figsize=(18,4))
    plt.subplot(131)
    plt.plot(train_Loss,label='Train Loss',c='y')
    plt.legend()
    plt.subplot(132)
    plt.plot(train_acc, label='Train Accuracy', c='y')
    plt.legend()
    # plt.subplot(133)
    # plt.scatter(X[0,:],X[1,:],c=y,cmap='winter')
    # plt.contour(W11,W22,Z,10,alpha=0.9,color='Blues')
    # plt.colorbar()
    plt.show()
    
    # print(a.shape,a)
    return param
#-------------------

# X_train = np.fromfile('./data/train-images-idx3-ubyte',dtype=np.uint16)
# X_train.reshape((784,-1))

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28*28).T / 255.0
X_test = X_test.reshape(-1,28*28).T /255.0
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

print('dimension de X: ', X_train.shape)
print('dimension de y: ',y_train.shape)
print('dimension de X: ', X_test.shape)
print('dimension de y: ',y_test.shape)

# calcul du reseau
W_param = neural_network(X_train,y_train,[49,16],lr=0.15,n_iter=100)

saveParam('mnist_v5.npy',W_param)
# print(W_param)
# print (len(W_param))
# print('W1:',W_param['W1'].shape)
# print('W2:',W_param['W2'].shape)
# print('W3:',W_param['W3'].shape)
# print('b1:',W_param['b1'].shape)
# print('b2:',W_param['b2'].shape)
# print('b3:',W_param['b3'].shape)

# prediction
# for i in range(9):
#     image = X_test[i].reshape((-1,1))
#     image_predict = predict(image,W_param)
#     print('prediction',image_predict.T)
#     print(y_test.T[i])
#     plt.subplot(330+i+1)
#     image = image.reshape((28, 28))*255.0
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")

# print('X_test shape:',X_test.shape,'X_train shape',X_train.shape,'y_test shape', y_test.T[0])
# image = X_test.T[0].reshape((-1,1))
# print('image shape',image.shape)
# print('image predict shape:',image_predict.shape)
# plt.show()
