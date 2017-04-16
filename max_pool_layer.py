import numpy as np
from layer_activations import *

class max_pool_layer():
	def __init__(self,layer_opts):
		self.pool_size = layer_opts['pool_size']
		self.stride = layer_opts['stride']
		self.backpool = None

	def forward(self,layer_input):
		# calculate dimension of layer output
		out_dim1 = (layer_input.shape[1]-self.pool_size)/self.stride + 1
		out_dim2 = (layer_input.shape[2]-self.pool_size)/self.stride + 1

		# output of conv layer is same dimension as input with a depth of the number of filters
		layer_output = np.zeros((layer_input.shape[0],out_dim1,out_dim2))
		self.backpool = np.zeros(layer_input.shape)
		# for each dimension in the input
		for dim in range(layer_input.shape[0]):
			# for each row
			for start_row in range(0,layer_input.shape[1]-self.pool_size+1,self.stride):
				# for each column
				for start_col in range(0,layer_input.shape[2]-self.pool_size+1,self.stride):
					index = np.argmax(layer_input[dim][start_row:start_row+self.pool_size,start_col:start_col+self.pool_size])
					

					self.backpool[dim,start_row+(index//self.pool_size),start_col+(index%self.pool_size)] = 1
					# max pool operation over pool window
					layer_output[dim,start_row,start_col] = np.max(layer_input[dim][start_row:start_row+self.pool_size,start_col:start_col+self.pool_size])
		
		return layer_output
	def backprop(self,gradient):
		delta = np.zeros(self.backpool.shape)

		for dim in range(gradient.shape[0]):
			# for each row
			for start_row in range(gradient.shape[1]):
				# for each column
				for start_col in range(gradient.shape[2]):
					delta[dim,start_row*self.stride:start_row*self.stride + self.pool_size,
					start_col*self.stride:start_col*self.stride + self.pool_size] += gradient[dim,start_row,start_col] * self.backpool[dim,
					start_row*self.stride:start_row*self.stride + self.pool_size,
					start_col*self.stride:start_col*self.stride + self.pool_size]
					
		return delta