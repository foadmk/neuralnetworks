import numpy as np
import neuralnet_v1 as my_neural
    
X = np.array([
            [0,0,1],
            [0,1,0],
            [1,0,0],
            [0,1,1],
            [1,0,1],
            [0,1,1],
            [1,1,1]
            ])
                
y = np.array([
          [0],
          [0],
          [0],
			[1],
			[1],
			[1],
			[0]
            ])
  
network = np.array([3,10,5,7,3,1])

syn = my_neural.init_syn(network)

syn0 = syn

for j in range(5):
    syn = my_neural.train(network, syn, X, y, 10000, lrate=1.0, batch_size=7)
    kn = my_neural.feed_forward(network, syn, X)
    print("Error:" + str(np.mean(np.abs(kn[len(network)-1] - y))))

for i in range(len(X)):
    print("Input: {},{},{} = Prediction: {} - Target: {}".format(X[i][0],X[i][1],X[i][2], kn[len(network)-1][i], y[i]))
    
