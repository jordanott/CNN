from conv_layer import conv_layer
from max_pool_layer import max_pool_layer
from fully_connected_layer import fully_connected_layer
from layer_activations import *
import numpy as np

class cnn():
	def __init__(self,input_dimension):
		self.input_dimension = input_dimension
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
		'num_filters':num_filters
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

		return self.softmax(data)

	def backward(self,gradient):
		for layer in reversed(self.layers):
			gradient = layer.backprop(gradient)


# # 3x3 filter, filter one, stride one
# conv1 = conv_layer(3,1,1)
# # 2x2 max pool, stride one
# pool1 = max_pool_layer(2,1)
# # 10 fully connected neurons
# fc1 = fully_connected_layer(729,10)

cnn = cnn(28)
cnn.add_layer('conv',stride=1,num_filters=1,filter_dim=3,padding=1,activation='relu')
cnn.add_layer('max_pool',stride=1,pool_size=2)
cnn.add_layer('fc',num_neurons=10,activation='relu')

data = np.arange(16).reshape((1,4,4))
soft = cnn.forward(data)
cnn.backward(soft)