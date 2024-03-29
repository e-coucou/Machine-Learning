# from keras import Sequential, Model
import numpy as np
# from keras import Input
# from keras import layers

class NN_ep():
    def __init__(self,X_,y_,layers_,learning_rate_=0.1) -> None:
        self.layers = layers_
        self.lr = learning_rate_
        self.X = X_
        self.y = y_
        self.layers.insert(0,X_.shape[0])
        self.layers.append(y_.shape[0])
        self.W = []
        self.c = len(self.layers)-1
        for i in np.arange(0,len(self.layers)-2):
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1)
            self.W.append(w / np.sqrt(self.layers[i]))
        w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        self.W.append(w / np.sqrt(self.layers[-2]))     

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    def sigmoid_prime(self, x):
        return x * (1 - x)

    def log_loss(self,A,y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1-y)*np.log(1-A))

    def predict(self,X_):
        A = np.c_[X_,np.ones(X_.shape[0])]
        for l in range(self.c):
            A = self.sigmoid(np.dot(A,self.W[l]))
        return A

    def forward(self,X_):
        A = np.c_[X_,np.ones(X_.shape[0])]
        A = [np.atleast_2d(A)] # x en colonne
        for l in range(self.c):
            a = self.sigmoid(np.dot(A[l],self.W[l]))
            A.append(a)
        return A

    def fit(self, X_, y_):
        A = self.forward(X_)

    def backward(self,A_,y_):
        error = A_[-1] - y_
        print('$$$$$$')
        print(A_[-1])
        print('y',y_)
        print('error',error)
        D = [error * self.sigmoid_prime(A_[-1])]
        print('dérivées partielles : ',D)
        for l in np.arange(len(A_) - 2, 0, -1):
            delta = D[-1].dot(self.W[l].T)
            delta = delta * self.sigmoid_prime(A_[l])
            D.append(delta)
        print('de0',D)
        D = D[::-1]
        print('de1',D)
        for l in np.arange(0, self.c):
            print(A_[l].shape,D[l].shape)
            self.W[l] += -self.lr * A_[l].T.dot(D[l])

class NNs_ep():
    def __init__(self,layers, alpha_=0.1) -> None:
        self.layers = layers
        self.alpha = alpha_
        self.W = []
        for i in np.arange(0,len(self.layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))        

    def copy(self):
        return self.W.copy()

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones((X.shape[0]))] # [Xi,1]
        for epoch in np.arange(0, epochs):
            print('zip',zip(X,y))
            for (x, target) in zip(X, y): # [ (Xi,yi) ]
                print(x,target)
                self.fit_partial(x, target)
            # if epoch == 0 or (epoch + 1) % displayUpdate == 0:
            #     loss = self.calculate_loss(X, y)
            #     self.loss_curve.append(loss)
			# 	# print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
    
    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)] # x en colonne
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
        print('base A',A)
        error = A[-1] - y
        print('$$$$$$')
        print(A[-1])
        print('y',y)
        print('error',error)
        D = [error * self.sigmoid_deriv(A[-1])]
        print('D',D)
        print('len A',len(A))
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        D = D[::-1]
        print('base D, reverse',D)
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X):
        p = np.atleast_2d(X)
        p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def mutate(self,func):
        vfunc = np.vectorize(func)
        for l in range(len(self.layers)-1):
            self.W[l] = np.array(vfunc(self.W[l]))
            
# class NeuralNetwork_EP:
#     def __init__(self, layers, alpha=0.1):
#         w=0
# 		# initialize the list of weights matrices, then store the
#     	# network architecture and learning rate
#         self.W = []
#         self.layers = layers
#         self.alpha = alpha
#         self.loss_curve = []
#         # start looping from the index of the first layer but
# 		# stop before we reach the last two layers
#         for i in np.arange(0, len(layers) - 2):
# 			# randomly initialize a weight matrix connecting the
# 			# number of nodes in each respective layer together,
# 			# adding an extra node for the bias
#             w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
#             self.W.append(w / np.sqrt(layers[i]))
# 		# the last two layers are a special case where the input
# 		# connections need a bias term but the output does not
#         w = np.random.randn(layers[-2] + 1, layers[-1])
#         self.W.append(w / np.sqrt(layers[-2]))        
#     def __repr__(self):
# 		# construct and return a string that represents the network
# 		# architecture
#         return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
#     def sigmoid(self, x):
#     	# compute and return the sigmoid activation value for a
# 		# given input value
#         return 1.0 / (1 + np.exp(-x))

#     def sigmoid_deriv(self, x):
# 		# compute the derivative of the sigmoid function ASSUMING
# 		# that x has already been passed through the 'sigmoid'
# 		# function
#         return x * (1 - x)

#     def fit(self, X, y, epochs=1000, displayUpdate=100):
# 		# insert a column of 1's as the last entry in the feature
# 		# matrix -- this little trick allows us to treat the bias
# 		# as a trainable parameter within the weight matrix
#         X = np.c_[X, np.ones((X.shape[0]))]

# 		# loop over the desired number of epochs
#         for epoch in np.arange(0, epochs):
# 			# loop over each individual data point and train
# 			# our network on it
#             for (x, target) in zip(X, y):
#                 self.fit_partial(x, target)

#             # check to see if we should display a training update
#             if epoch == 0 or (epoch + 1) % displayUpdate == 0:
#                 loss = self.calculate_loss(X, y)
#                 self.loss_curve.append(loss)
# 				# print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
    
#     def fit_partial(self, x, y):
# 		# construct our list of output activations for each layer
# 		# as our data point flows through the network; the first
# 		# activation is a special case -- it's just the input
# 		# feature vector itself
#         A = [np.atleast_2d(x)]
# 		# FEEDFORWARD:
# 		# loop over the layers in the network
#         for layer in np.arange(0, len(self.W)):
# 			# feedforward the activation at the current layer by
# 			# taking the dot product between the activation and
# 			# the weight matrix -- this is called the "net input"
# 			# to the current layer
#             net = A[layer].dot(self.W[layer])

# 			# computing the "net output" is simply applying our
# 			# nonlinear activation function to the net input
#             out = self.sigmoid(net)

# 			# once we have the net output, add it to our list of
# 			# activations
#             A.append(out)
# 		# BACKPROPAGATION
# 		# the first phase of backpropagation is to compute the
# 		# difference between our *prediction* (the final output
# 		# activation in the activations list) and the true target
# 		# value
#         error = A[-1] - y

# 		# from here, we need to apply the chain rule and build our
# 		# list of deltas 'D'; the first entry in the deltas is
# 		# simply the error of the output layer times the derivative
# 		# of our activation function for the output value
#         D = [error * self.sigmoid_deriv(A[-1])]
# 		# once you understand the chain rule it becomes super easy
# 		# to implement with a 'for' loop -- simply loop over the
# 		# layers in reverse order (ignoring the last two since we
# 		# already have taken them into account)
#         for layer in np.arange(len(A) - 2, 0, -1):
# 			# the delta for the current layer is equal to the delta
# 			# of the *previous layer* dotted with the weight matrix
# 			# of the current layer, followed by multiplying the delta
# 			# by the derivative of the nonlinear activation function
# 			# for the activations of the current layer
#             delta = D[-1].dot(self.W[layer].T)
#             delta = delta * self.sigmoid_deriv(A[layer])
#             D.append(delta)
#   		# since we looped over our layers in reverse order we need to
# 		# reverse the deltas
#         D = D[::-1]

# 		# WEIGHT UPDATE PHASE
# 		# loop over the layers
#         for layer in np.arange(0, len(self.W)):
# 			# update our weights by taking the dot product of the layer
# 			# activations with their respective deltas, then multiplying
# 			# this value by some small learning rate and adding to our
# 			# weight matrix -- this is where the actual "learning" takes
# 			# place
#             self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

#     def predict(self, X, addBias=True):
# 		# initialize the output prediction as the input features -- this
# 		# value will be (forward) propagated through the network to
# 		# obtain the final prediction
#         p = np.atleast_2d(X)

# 		# check to see if the bias column should be added
#         if addBias:
# 			# insert a column of 1's as the last entry in the feature
# 			# matrix (bias)
#             p = np.c_[p, np.ones((p.shape[0]))]

# 		# loop over our layers in the network
#         for layer in np.arange(0, len(self.W)):
# 			# computing the output prediction is as simple as taking
# 			# the dot product between the current activation value 'p'
# 			# and the weight matrix associated with the current layer,
# 			# then passing this value through a nonlinear activation
# 			# function
#             p = self.sigmoid(np.dot(p, self.W[layer]))

# 		# return the predicted value
#         return p>=0.5
    
#     def calculate_loss(self, X, targets):
# 		# make predictions for the input data points then compute
# 		# the loss
#         targets = np.atleast_2d(targets)
#         predictions = self.predict(X, addBias=False)
#         loss = 0.5 * np.sum((predictions - targets) ** 2)
#         # return the loss
#         return loss
    
#     def mutate(self,func):
#         self.W = map(func,self.W)

# class NNep():
#     def __init__(self,input_,hidden_,output_) -> None:
#         # le model Keras
#         input_data = Input(shape=(input_))
#         x = layers.Dense(hidden_, activation='relu')(input_data)
#         output_data = layers.Dense(output_, activation='relu')(x)
#         self.model = Model(input_data,output_data)
#         self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy')


#     def print(self):
#         self.model.summary()

#     def predict(self,input):
#         return self.model.predict(input,verbose=0)
#         # return output
