from net import net
import numpy as np

cnn = net(.001)
cnn.add_layer('conv',stride=1,num_filters=1,filter_dim=3,padding=1,activation='relu')
cnn.add_layer('max_pool',stride=1,pool_size=2)
cnn.add_layer('fc',num_neurons=10,activation='relu')
cnn.add_layer('output',output_function='softmax')

import cPickle, gzip
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

correct = 0
# training
for i in range(0,len(train_set[0])):
	data = train_set[0][i].reshape((1,28,28))
	target = np.zeros((1,10))
	target[0,train_set[1][i]] = 1

	
	# backprop through network
	predictions = cnn.forward(data) + 1e-7
	cost = -np.sum(target*np.log(predictions))
	print cost
	prediction = np.argmax(predictions)
	
	if target[0,prediction]:
		correct += 1
	print "Iteration:",i,"Accuracy", float(correct/float(i+1))
	gradient = predictions - target
	cnn.backward(gradient)
	