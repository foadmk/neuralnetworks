#plain simple neural network
#credits to http://adventuresinmachinelearning.com/neural-networks-tutorial/

import numpy as np
import numpy.random as r
import time

np.random.seed(1)

# randomly initialize our weights with mean 0
def init_syn(network):
    syn = {}
    for i in range(len(network)-1):
        syn[i] = np.empty([network[i],network[i+1]], dtype=float)
        for j in range(network[i]):
            for k in range(network[i+1]):
                syn[i][j][k] = 2*r.random() - 1
    return syn;

def feed_forward(network, syn, X):
    k = {}
    for i in range(len(network)):
        if i == 0:
            k[i] = X
        else:
            k[i] = nonlin(np.dot(k[i-1], syn[i-1]))
    return k

def back_prop(network, k, y, syn, rate, n):
    k_error = {}
    k_delta = {}  
    
    for i in range(len(network)-1,-1,-1):
        if i == len(network)-1:
            k_error[i] = y - k[i]     
        else:
            k_error[i] = k_delta[i+1].dot(syn[i].T)             
        k_delta[i] = k_error[i]*nonlin(k[i],deriv=True)       
        
    for i in range(len(network)-1):
        syn[i] += k[i].T.dot(k_delta[i+1]) * 1/n * rate
    
    return syn

def train(network, syn, X, y, cycles, lrate, batch_size):
    start_time = time.time()
    for j in range(cycles):
        Xt, yt = get_mini_batch(X, y, batch_size)
        k = feed_forward(network, syn, Xt)
        syn = back_prop(network, k, yt, syn, lrate, batch_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    return syn

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def predict(network, syn, X_arr):    
    k = feed_forward(network, syn, X_arr)
    return k[len(network)-1]

def predict_number(network, syn, X_arr): 
    vals = predict(network, syn, X_arr)
    y = {}
    for i in range(len(vals)):
        y[i] = np.argmax(vals[i])
    return y

def prediction_eval(y_pred, y_real):
    cnt = 0;
    correct = 0;
    for i in range(len(y_pred)):
        cnt += 1
        if y_pred[i] == y_real[i]:
           correct += 1
    return 100*(correct/cnt)

def get_mini_batch(X, y, batch_size):
    random_idxs = r.choice(len(y), batch_size, replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    return X_shuffled, y_shuffled

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
