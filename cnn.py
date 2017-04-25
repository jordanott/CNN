import sys
sys.path.insert(0, 'net/')

from net import net
import numpy as np

cnn = net(.001)
# height,width,depth
cnn.add_layer('input',shape=(28,28,1))
# cnn.add_layer('fc',num_neurons=32,activation='relu')
#cnn.add_layer('conv',stride=1,num_filters=3,filter_dim=3,padding=1,activation='relu')
# cnn.add_layer('max_pool',stride=1,pool_size=2)
# cnn.add_layer('conv',stride=1,num_filters=3,filter_dim=3,padding=1,activation='relu')
# cnn.add_layer('max_pool',stride=1,pool_size=2)
cnn.add_layer('output',num_neurons=10,activation='softmax')


import cPickle, gzip
# Load the dataset
f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

correct = 0

gradient = np.zeros((1,10))
loss = 0

for i in range(0,len(train_set[0])):
	data = train_set[0][i].reshape((28,28,1))
	target = np.zeros((1,10))
	target[0,train_set[1][i]] = 1
	
	# backprop through network
	predictions = cnn.forward(data)
	#print "PREDICTIONS:",predictions
	prediction = np.argmax(predictions)

	gradient = cnn.get_gradient(predictions,target)
	# TODO: fix loss
	# loss += cnn.get_cost(predictions,target)		
	cnn.backward(gradient)
	if target[0,prediction]:
		correct += 1
	
	#print "Loss:",loss

	if (i+1) % 128 == 0:
		print "Epcoh",i//128
		print "Accuracy",correct/128.0
		correct = 0
		
		gradient = np.zeros((1,10))

correct = 0
for i in range(0,len(test_set[0])):
	data = test_set[0][i].reshape((28,28,1))
	target = np.zeros((1,10))
	target[0,test_set[1][i]] = 1
	
	# backprop through network
	predictions = cnn.forward(data)
	#print "PREDICTIONS:",predictions
	prediction = np.argmax(predictions)

	if target[0,prediction]:
		correct += 1
	
print "Test accuracy:", correct/float(len(test_set[0]))
