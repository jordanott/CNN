import numpy as np
from layer_activations import *

class conv_layer():
	def __init__(self,layer_opts):
		# Ex: filter_dim = 3 => 3x3 filter
		self.filter_dim = layer_opts['filter_dim']
		# How many filters in layer
		self.num_filters = layer_opts['num_filters']
		# stride amount
		self.stride = layer_opts['stride']
		# learning rate for weight updates 
		self.learning_rate = layer_opts['learning_rate']
		# activation function
		self.activation = layer_opts['activation']
		# derivative of activation function for backprop
		self.backtivation = layer_opts['backtivation']
		# amount of padding for input
		self.padding = layer_opts['padding']
		# initialize filters 
		self.filters = self.init_filters()
		# output of sliding filter through feature map
		self.layer_product = None
		# derivative of loss with respect to weights
		self.filter_updates = None
		# input of layer
		self.layer_input_padded = None

	def init_filters(self):
		# initialize filters from normal distribution
		return np.random.normal(size=(self.num_filters,self.filter_dim,self.filter_dim))

	def add_padding(self,layer_input):
		# padd image with zeros
		return np.lib.pad(layer_input, (self.padding,self.padding), 'constant', constant_values=(0))

	def forward(self,layer_input):
		self.filter_updates = np.zeros((self.num_filters,self.filter_dim,self.filter_dim))
		# output of conv layer is same dimension as input with a depth of the number of filters
		layer_output = np.zeros((self.num_filters,layer_input.shape[1],layer_input.shape[2]))

		# pad each dimension of input with zeros
		self.layer_input_padded = np.zeros((layer_input.shape[0],layer_input.shape[1]+2,layer_input.shape[2]+2))

		for dim in range(layer_input.shape[0]):
			self.layer_input_padded[dim] = self.add_padding(layer_input[dim])

		for filter_num in range(self.num_filters): 
			for start_row in range(0,layer_output.shape[1],self.stride):
				for start_col in range(0,layer_output.shape[2],self.stride):

					layer_output[filter_num,start_row,start_col] = np.sum(self.filters[filter_num]*
						self.layer_input_padded[filter_num,start_row:start_row+self.filter_dim, start_col:start_col+self.filter_dim])
		self.layer_product = layer_output
		return self.activation(layer_output)

	def backprop(self,gradient):
		gradient = self.backtivation(self.layer_product) * gradient
		self.dLdw = np.zeros(self.layer_input_padded.shape)
		
		for dim in range(self.num_filters):
			# for each row
			for start_row in range(self.filter_dim):
				# for each column
				for start_col in range(self.filter_dim):
					self.filter_updates[dim] += gradient[dim,start_row,start_col] * self.layer_input_padded[dim,
					start_row*self.stride:start_row*self.stride + self.filter_dim,
					start_col*self.stride:start_col*self.stride + self.filter_dim]
					
					# TODO 
					self.dLdw[dim,start_row*self.stride:start_row*self.stride + self.filter_dim,
					start_col*self.stride:start_col*self.stride + self.filter_dim] += gradient[dim,start_row,start_col] * self.filters[dim,start_row,start_col]

		self.filters += -self.learning_rate*self.filter_updates
		
		return self.dLdw[:,self.padding:self.dLdw.shape[1]-2*self.padding +1,self.padding:self.dLdw.shape[2]-2*self.padding+1]