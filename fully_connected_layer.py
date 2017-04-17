import numpy as np
from layer_activations import *

class fully_connected_layer():

	def __init__(self,layer_opts):
		# number of neurons for fc layer
		self.num_neurons = layer_opts['num_neurons']
		# activation function
		self.activation = layer_opts['activation']
		# derivative of relu for backprop
		self.backtivation = layer_opts['backtivation']
		# learning rate for weight updates 
		self.learning_rate = layer_opts['learning_rate']
		# layer_input = activation of previous layer
		self.layer_input = None
		# dot(layer_input,weights)
		self.layer_product = None
		# layer weights 
		self.weights = None
		# derivative of loss with respect to weights
		self.dLdw = None
		# original dimension of input
		self.original_dim = None

	def init_weights(self,previous):
		# initializing weights
		return np.random.normal(size=(previous,self.num_neurons))

	def forward(self,layer_input):
		self.layer_input = layer_input
		# save original dimension of input
		self.original_dim = layer_input.shape
		# flatten layer input
		self.layer_input = layer_input.flatten()
		self.layer_input = self.layer_input.reshape((1,self.layer_input.shape[0]))
		# if weights haven't been initialized
		if self.weights == None:
			# initialize weights
			self.weights = self.init_weights(self.layer_input.shape[1])		

		self.layer_product = np.dot(self.layer_input,self.weights)
		# return output with activation on layer
		return self.activation(self.layer_product)

	def backprop(self,gradient):
		# derivative of relu(dot(layer_input,weights)) * gradient
		delta = self.backtivation(self.layer_product) * gradient
		# derivative of loss with respect to weights; prev activation * propogated gradient
		self.dLdw = np.dot(self.layer_input.T,delta)
		# return derivative of loss with respect to layer input
		delta = np.dot(delta,self.weights.T)
		# update weights: learning rate * derivative of loss with respect to weights
		self.weights -= self.learning_rate*self.dLdw

		return delta.reshape(self.original_dim)