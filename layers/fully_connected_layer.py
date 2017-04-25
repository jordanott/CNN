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
		# set shape of incoming tensor
		self.incoming_shape = layer_opts['incoming_shape']
		# set shape of outgoing tensor
		self.output_shape = self.get_output_shape()
		# layer_input = activation of previous layer
		self.layer_input = None
		# dot(layer_input,weights)
		self.layer_product = None
		# layer weights
		self.weights = None
		# layer bias
		self.bias = None


	def get_output_shape(self):
		# output shape is 
		return (1,self.num_neurons)

	def init_weights(self,previous):
		# initializing weights
		self.weights = 0.01*np.random.randn(previous,self.num_neurons) #scale=2/float(previous)
		self.bias = np.zeros((1,self.num_neurons))

	def forward(self,layer_input):
		self.layer_input = layer_input
		# flatten layer input
		self.layer_input = layer_input.flatten().reshape(1,-1)

		# if weights haven't been initialized
		if self.weights == None:
			# initialize weights
			self.init_weights(self.layer_input.shape[1])
		
		if self.weights[0][0] < -1000:
			print "PROBLEM"
			print self.weights

		self.layer_product = np.dot(self.layer_input,self.weights) + self.bias
		# return output with activation on layer
		return self.activation(self.layer_product),self.l2()

	def l2(self):
		#print self.weights
		return np.sum(self.weights*self.weights)

	def backprop(self,gradient):
		gradient = self.backtivation(self.layer_product) * gradient
		# derivative of loss with respect to weights; prev activation * propogated gradient
		dw = np.dot(self.layer_input.T,gradient)
		# return derivative of loss with respect to layer input
		delta = np.dot(gradient,self.weights.T)
		# update weights: learning rate * derivative of loss with respect to weights
		# weight regularization
		dw += self.weights * 1e-3
		self.weights += -self.learning_rate*dw 

		return delta.reshape(self.incoming_shape)
