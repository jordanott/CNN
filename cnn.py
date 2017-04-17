from conv_layer import conv_layer
from max_pool_layer import max_pool_layer
from fully_connected_layer import fully_connected_layer
from layer_activations import *
import numpy as np

class cnn():
	def __init__(self,learning_rate=.001):
		self.learning_rate = learning_rate
		self.layers = []
		# layer types
		self.layer_types = {
		'conv':conv_layer,
		'fc':fully_connected_layer,
		'max_pool':max_pool_layer
		}

	def add_layer(self,layer_type,pool_size=2,stride=1,num_neurons=0,filter_dim=3,num_filters=1,padding=1,activation='relu'):
		# layer options
		layer_opts = {
		'stride':stride,
		'padding':padding,
		'pool_size':pool_size,
		'filter_dim':filter_dim,
		'num_neurons':num_neurons,
		'num_filters':num_filters,
		'learning_rate':self.learning_rate
		}
		# set layer activation function
		layer_opts['activation'] = activation_functions[activation][0]
		layer_opts['backtivation'] = activation_functions[activation][1]
		# add new layer
		self.layers.append(self.layer_types[layer_type](layer_opts))

	def softmax(self,z):
		# softmax function
		return np.exp(z) / np.sum(np.exp(z))

	def forward(self,data):
		for layer in self.layers:
			data = layer.forward(data)

		return data

	def backward(self,gradient):
		for layer in reversed(self.layers):
			print layer.__class__
			print gradient
			gradient = layer.backprop(gradient)

cnn = cnn(.001)
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