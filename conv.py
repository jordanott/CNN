import numpy as np

x = np.random.rand(28,28)

class conv_layer():
	def __init__(self,filter_dim,num_filters,stride):
		self.filter_dim = filter_dim
		self.num_filters = num_filters
		self.stride = stride
		self.filters = self.init_filters()

	def init_filters(self):
		return np.random.normal(size=(self.num_filters,self.filter_dim,self.filter_dim))

	def add_padding(self,layer_input):
		# padd image with zeros
		return np.lib.pad(layer_input, (1,1), 'constant', constant_values=(0))

	def conv(self,layer_input):

		# output of conv layer is same dimension as input with a depth of the number of filters
		layer_output = np.zeros((self.num_filters,layer_input.shape[1],layer_input.shape[2]))

		# pad each dimension of input with zeros
		for dim in range(layer_input.shape[0]):
			layer_input = self.add_padding(layer_input[dim])

		for filter_num in range(self.num_filters): 
			for start_row in range(0,layer_output.shape[1],self.stride):
				for start_col in range(0,layer_output.shape[2],self.stride):
					layer_output[filter_num,start_row,start_col] = np.sum(self.filters[filter_num]*
						# layer_input[dim][...] if handeling multi dim inputs
						layer_input[start_row:start_row+self.filter_dim, start_col:start_col+self.filter_dim])
		
		return layer_output

	def backprop(self):
		# TODO
		pass


class max_pool_layer():
	def __init__(self,pool_size,stride):
		self.pool_size = pool_size
		self.stride = stride

	def max_pool(self,layer_input):
		# calculate dimension of layer output
		out_dim1 = (layer_input.shape[1]-self.pool_size)/self.stride + 1
		out_dim2 = (layer_input.shape[2]-self.pool_size)/self.stride + 1

		# output of conv layer is same dimension as input with a depth of the number of filters
		layer_output = np.zeros((layer_input.shape[0],out_dim1,out_dim2))
		
		# for each dimension in the input
		for dim in range(layer_input.shape[0]):
			# for each row
			for start_row in range(0,layer_output.shape[1],self.stride):
				# for each column
				for start_col in range(0,layer_output.shape[2],self.stride):
					# max pool operation over pool window
					layer_output[dim,start_row,start_col] = np.max(layer_input[dim][start_row:start_row+self.pool_size,start_col:start_col+self.pool_size])
		return layer_output
	def backprop():
		# TODO
		pass
x = x.reshape((1,28,28))

# 3x3 filter, filter one, stride one
conv1 = conv_layer(3,1,1)
conv2 = conv_layer(3,1,1)

out1 = conv1.conv(x)
print out1.shape
out2 = conv2.conv(out1)
print out2.shape

pool1 = max_pool_layer(2,1)
print pool1.max_pool(out2)
