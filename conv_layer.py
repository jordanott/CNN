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
		# set shape of incoming tensor
		self.incoming_shape = layer_opts['incoming_shape']
		# set shape of outgoing tensor
		self.output_shape = self.get_output_shape()
		# initialize filters
		self.filters = self.init_filters()
		# output of sliding filter through feature map
		self.layer_product = None
		# derivative of loss with respect to weights
		self.filter_updates = None
		# input of layer
		self.layer_input_padded = None

	def get_output_shape(self):
		# (n-f)/stride + 1 = dim
		incoming_width_height = self.incoming_shape[1]
		if not ((2*self.padding + incoming_width_height - self.filter_dim)/float(self.stride)).is_integer():
			print "WARNING: Padding need..."
		out = (2*self.padding + incoming_width_height - self.filter_dim)/self.stride + 1
		# determine the dimensions of shape produced by layer
		return (out,out,self.num_filters)

	def init_filters(self):
		# initialize filters from normal distribution => filter height x filter width x depth of incoming shape
		return np.random.normal(size=(self.num_filters,self.filter_dim,self.filter_dim,self.incoming_shape[-1]))

	def add_padding(self,layer_input):
		print "printing layer input shape" , layer_input.shape
		# padd image with zeros
		padded = np.zeros((layer_input.shape[0]+ 2*self.padding,layer_input.shape[1]+2*self.padding,layer_input.shape[2]))
		padded[self.padding:self.padding+layer_input.shape[0],self.padding:self.padding+layer_input.shape[0]] = layer_input
		print "padded"
		print padded.shape
		return padded
	
	def forward(self,layer_input):
		# initialize filter update matrix
		self.filter_updates = np.zeros(self.filters.shape)
		# output of conv layer is same dimension as input with a depth of the number of filters
		layer_output = np.zeros(self.output_shape)
		print "output shape", self.output_shape
		if self.padding > 0:	
			# pad input with zeros
			self.layer_input_padded = self.add_padding(layer_input)
		else:
			# if no padding
			self.layer_input_padded = layer_input

		for filter_num in range(self.num_filters):
			for start_row in range(0,layer_output.shape[1],self.stride):
				for start_col in range(0,layer_output.shape[2],self.stride):
					# TODO fix this
					layer_output[filter_num,start_row,start_col] = np.sum(self.filters[filter_num] *
						self.layer_input_padded[start_row:start_row+self.filter_dim, start_col:start_col+self.filter_dim])

		self.layer_product = layer_output

		return self.activation(layer_output)

	def backprop(self,gradient):
		gradient = self.backtivation(self.layer_product) * gradient
		self.dLdw = np.zeros(self.layer_input_padded.shape)

		print self.layer_input_padded.shape
		for dim in range(self.num_filters):
			# for each row
			for start_row in range(self.filter_dim):
				# for each column
				for start_col in range(self.filter_dim):
					self.filter_updates[dim] += gradient[dim,start_row,start_col] * self.layer_input_padded[
					start_row*self.stride:start_row*self.stride + self.filter_dim,
					start_col*self.stride:start_col*self.stride + self.filter_dim]

					# TODO
					self.dLdw[dim,start_row*self.stride:start_row*self.stride + self.filter_dim,
					start_col*self.stride:start_col*self.stride + self.filter_dim] += gradient[start_row,start_col] * self.filters[dim,start_row,start_col]

		self.filters += -self.learning_rate*self.filter_updates

		return self.dLdw[:,self.padding:self.dLdw.shape[1]-2*self.padding +1,self.padding:self.dLdw.shape[2]-2*self.padding+1]
