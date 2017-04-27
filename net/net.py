import sys
sys.path.insert(0, 'layers/')

from layer_activations import *
from conv_layer import conv_layer
from input_layer import input_layer
from output_layer import output_layer
from max_pool_layer import max_pool_layer
from fully_connected_layer import fully_connected_layer

import numpy as np

class net():
	def __init__(self,learning_rate=.001):
		self.learning_rate = learning_rate
		self.layers = []
		# layer types
		self.layer_types = {
		'conv':conv_layer,
		'fc':fully_connected_layer,
		'input':input_layer,
		'max_pool':max_pool_layer,
		'output':output_layer
		}
		# regularization loss for weights
		self.l2 = 0

	def add_layer(self,layer_type,shape=0,pool_size=2,stride=1,num_neurons=0,filter_dim=3,num_filters=1,padding=1,activation='relu',output_function='softmax'):
		# layer options
		layer_opts = {
		'stride':stride,
		'padding':padding,
		'pool_size':pool_size,
		'incoming_shape':shape,
		'filter_dim':filter_dim,
		'num_neurons':num_neurons,
		'num_filters':num_filters,
		'output_function':output_function,
		'learning_rate':self.learning_rate		
		}
		# set layer activation function
		layer_opts['activation'] = activation_functions[activation][0]
		layer_opts['backtivation'] = activation_functions[activation][1]

		if self.layers:
			# set depth of filter based off depth of incoming shape
			layer_opts['incoming_shape'] = self.layers[-1].output_shape
		#print layer_type,"********* Incoming shape:",layer_opts['incoming_shape']
		# add new layer
		self.layers.append(self.layer_types[layer_type](layer_opts))
		#print layer_type,"********* Outgoing shape:",self.layers[-1].output_shape


	def forward(self,data):
		for layer in self.layers:
			data,reg = layer.forward(data)
			
		predictions = data

		return predictions

	def backward(self,gradient):
		for layer in reversed(self.layers):
			gradient = layer.backprop(gradient)

	def cross_entropy_gradient(self,predictions,actual):
		# cross entropy gradients 
		predictions[0,np.argmax(actual)] -= 1
		return predictions

	def mean_squared_error_gradient(self,predictions,actual):
		# mean squared error gradients
		derivative = (1 - predictions) * predictions
		gradient = derivative * (predictions - actual)
		return gradient


	def get_cost(self,predictions,actual):
		#cost = -np.sum(actual*np.log(predictions))
		# compute the loss: average cross-entropy loss and regularization
		corect_logprobs = -np.log(predictions[0,np.argmax(actual)])
		data_loss = np.sum(corect_logprobs)
		reg_loss = 0.5*self.l2
		loss = data_loss + reg_loss
		return loss
