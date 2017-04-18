from layer_activations import *
from conv_layer import conv_layer
from output_layer import output_layer
from max_pool_layer import max_pool_layer
from fully_connected_layer import fully_connected_layer

import numpy as np
import cv2
class net():
	def __init__(self,learning_rate=.001):
		self.learning_rate = learning_rate
		self.layers = []
		# layer types
		self.layer_types = {
		'conv':conv_layer,
		'fc':fully_connected_layer,
		'max_pool':max_pool_layer,
		'output':output_layer
		}
		self.counter = 0

	def add_layer(self,layer_type,pool_size=2,stride=1,num_neurons=0,filter_dim=3,num_filters=1,padding=1,activation='relu',output_function='softmax'):
		# layer options
		layer_opts = {
		'stride':stride,
		'padding':padding,
		'pool_size':pool_size,
		'filter_dim':filter_dim,
		'num_neurons':num_neurons,
		'num_filters':num_filters,
		'learning_rate':self.learning_rate,
		'output_function':output_function
		}
		# set layer activation function
		layer_opts['activation'] = activation_functions[activation][0]
		layer_opts['backtivation'] = activation_functions[activation][1]
		# add new layer
		self.layers.append(self.layer_types[layer_type](layer_opts))

	def forward(self,data):
		for layer in self.layers:
			data = layer.forward(data)

			if layer.__class__ == conv_layer and self.counter %1000 == 0:
				
				
				cv2.startWindowThread()

				cv2.imshow("data",data.reshape((28,28, 1)))
				cv2.waitKey(0) 
				#cv2.imwrite("data" + str(self.counter) + ".jpg",data.reshape((28,28, 1)))

				cv2.destroyAllWindows()
				#x = raw_input("enter")
			self.counter+=1
		return data

	def backward(self,gradient):
		for layer in reversed(self.layers):
			gradient = layer.backprop(gradient)

