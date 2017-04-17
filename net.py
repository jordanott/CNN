from conv_layer import conv_layer
from max_pool_layer import max_pool_layer
from fully_connected_layer import fully_connected_layer
from layer_activations import *
import numpy as np

class net():
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

