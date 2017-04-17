from net import net
import numpy as np

cnn = net(.001)
cnn.add_layer('conv',stride=1,num_filters=1,filter_dim=3,padding=1,activation='relu')
cnn.add_layer('max_pool',stride=1,pool_size=2)
cnn.add_layer('fc',num_neurons=10,activation='relu')

import cPickle, gzip
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

correct = 0
# training
for i in range(0,len(train_set[0])):
	data = train_set[0][i].reshape((1,28,28))
	actual = np.zeros((1,10))
	actual[0,train_set[1][i]] = 1
	
	# backprop through network
	predictions = cnn.forward(data)
	
	loss = .5*np.sum(actual - predictions)**2
	print loss
	gradient = -(actual - predictions)
	
	prediction = np.argmax(predictions)
	print prediction
	if actual[0,prediction]:
		correct += 1
	print "Accuracy", float(correct/float(i+1))
	cnn.backward(gradient)