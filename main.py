from net import net
import numpy as np

cnn = net(.001)
# height,width,depth
cnn.add_layer('input',shape=(28,28,1))
cnn.add_layer('conv',stride=1,num_filters=3,filter_dim=3,padding=1,activation='relu')
cnn.add_layer('max_pool',stride=1,pool_size=2)
# cnn.add_layer('conv',stride=1,num_filters=3,filter_dim=3,padding=1,activation='relu')
# cnn.add_layer('max_pool',stride=1,pool_size=2)
cnn.add_layer('output',num_neurons=10,activation='softmax')


import cPickle, gzip
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

correct = 0
# data = np.arange(16).reshape((4,4,1))

# predictions = cnn.forward(data)
# print predictions
# training
gradient = np.zeros((1,10))
cost = 0

for i in range(0,len(train_set[0])):
	data = train_set[0][i].reshape((28,28,1))
	target = np.zeros((1,10))
	target[0,train_set[1][i]] = 1
	
	# backprop through network
	predictions,reg = cnn.forward(data)
	#print "PREDICTIONS:",predictions
	prediction = np.argmax(predictions)

	corect_logprobs = -np.log(predictions[0,train_set[1][i]])
	

	data_loss = np.sum(corect_logprobs) #/num_examples
	reg_loss = 0.5*reg
	loss = data_loss + reg_loss

	if target[0,prediction]:
		correct += 1

	predictions[0,train_set[1][i]] -= 1
	gradient = predictions
	
	cnn.backward(gradient)

	#print "Loss:",loss

	if (i+1) % 100 == 0:
		print "Accuracy",correct/100.0
		correct = 0
