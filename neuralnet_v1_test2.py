#Test with digit's (0-9) with 8x8 black and white images

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import neuralnet_v1 as my_neural
import matplotlib.pyplot as plt 
import numpy.random as r
from collections import Counter
import time

start_time = time.time()

digits = load_digits()
digits.data[0,:]
 
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_v_train = my_neural.convert_y_to_vect(y_train)
y_v_test = my_neural.convert_y_to_vect(y_test)

network = np.array([len(X[0]),100,100,50,len(y_v_test[0])])

array_network = {}
array_syn = {}
number_of_nets = 1

for i in range(number_of_nets):
    array_network[i] = network
    array_syn[i] = my_neural.init_syn(network)

acc_target = 99.9
for i in range(number_of_nets):
    acc = 0
    while acc < acc_target:
        array_syn[i] = my_neural.train(network=array_network[i], syn=array_syn[i], X=X_train, y=y_v_train, cycles=500, lrate=0.25, batch_size=25)
        predict = my_neural.predict_number(network=array_network[i], syn=array_syn[i], X_arr=X_train)
        acc = my_neural.prediction_eval(predict, y_train)
        print("Accuracy Train: {:3.5f}".format(acc))

    predict = my_neural.predict_number(network=array_network[i], syn=array_syn[i], X_arr=X_test)
    acc2 = my_neural.prediction_eval(predict, y_test)
    print("{} - Accuracy Train: {:3.5f} - Accuracy Test: {:3.15f} ".format(i, acc, acc2))

results = {}
for j in range(number_of_nets):
    results[j] = my_neural.predict_number(network=array_network[j], syn=array_syn[j], X_arr=X_test)   
    
predict_test = {}    
for i in range(len(X_test)):
    list_res = {}
    for j in range(number_of_nets):
        list_res[j] = results[j][i]
    cnt = Counter(list_res.values()).most_common(1)
    predict_test[i] = cnt[0][0]
        
test_error = my_neural.prediction_eval(predict_test, y_test)
print("Test Dataset:  {:3.4f}".format(test_error))
    

results = {}
for j in range(number_of_nets):
    results[j] = my_neural.predict_number(network=array_network[j], syn=array_syn[j], X_arr=X)
    
    
predict = {}    
for i in range(len(X)):
    list_res = {}
    for j in range(number_of_nets):
        list_res[j] = results[j][i]
    cnt = Counter(list_res.values()).most_common()
    predict[i] = cnt[0][0]




methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']



fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(hspace=0.3, wspace=0.05)

ids = r.choice(len(y), 25, replace=False)

i=0
for ax in axes.flat:
    ax.imshow(digits.images[ids[i]], interpolation='spline36', cmap='viridis')
    ax.set_title("Prediction:{}".format(predict[ids[i]]))
    i += 1
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
