import numpy as np
from layer_activations import *

class max_pool_layer():
	def __init__(self,layer_opts):
		# set dimension of max pool window
		self.pool_size = layer_opts['pool_size']
		# stride length
		self.stride = layer_opts['stride']
		# set shape of incoming tensor
		self.incoming_shape = layer_opts['incoming_shape']
		# set shape of outgoing tensor
		self.output_shape = self.get_output_shape()
		# 
		self.backpool = None

	def get_output_shape(self):
		# (n-f)/stride + 1 = dim
		incoming_width_height = self.incoming_shape[1]
		if not (( incoming_width_height - self.pool_size)/float(self.stride)).is_integer():
			print "WARNING: Padding need on pool..."
		out = (incoming_width_height - self.pool_size)/self.stride + 1
		# determine the dimensions of shape produced by layer
		return (out,out,self.incoming_shape[-1])

	def l2(self):
		return 0

	def forward(self,layer_input):
		#print "max pool input", layer_input.shape
		 
		# output of conv layer is same dimension as input with a depth of the number of filters
		layer_output = np.zeros(self.output_shape)
		self.backpool = np.zeros(layer_input.shape)
		# for each dimension in the input
		
		# for each row
		for start_row in range(0,layer_input.shape[0]-self.pool_size,self.stride):
			# for each column
			for start_col in range(0,layer_input.shape[1]-self.pool_size,self.stride):
				# 
				for dim in range(layer_input.shape[2]):
					# get index where max element occurs
					#print start_row, start_col, dim
					index = np.argmax(layer_input[start_row:start_row+self.pool_size,start_col:start_col+self.pool_size,dim])

					self.backpool[start_row+(index//self.pool_size),start_col+(index%self.pool_size),dim] = 1
					# max pool operation over pool window
					layer_output[start_row,start_col,dim] = np.max(layer_input[start_row:start_row+self.pool_size,start_col:start_col+self.pool_size,dim])
		#print "max pool output", layer_output.shape
		return layer_output, self.l2()

	def backprop(self,gradient):
		delta = np.zeros(self.backpool.shape)

		for start_row in range(gradient.shape[0]):
			# for each row
			for start_col in range(gradient.shape[1]):
				# for each column
				for dim in range(gradient.shape[2]):
					delta[start_row*self.stride:start_row*self.stride + self.pool_size,
					start_col*self.stride:start_col*self.stride + self.pool_size,dim] += gradient[start_row,start_col,dim] * self.backpool[
					start_row*self.stride:start_row*self.stride + self.pool_size,
					start_col*self.stride:start_col*self.stride + self.pool_size,dim]

		return delta
