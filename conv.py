import numpy as np

x = np.random.normal(size=(28,28))

learning_rate = 10
def relu(layer_input):
		return np.maximum(layer_input,0)
def d_relu(x):
    return 1. * (x > 0)

class conv_layer():

	def __init__(self,filter_dim,num_filters,stride):
		self.filter_dim = filter_dim
		self.num_filters = num_filters
		self.stride = stride
		self.filters = self.init_filters()
		self.activation = relu
		self.backtivation = d_relu

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
		return self.activation(layer_output)

	def backprop(self):
		# TODO
		pass

class fully_connected_layer():
	def __init__(self,previous_num_neurons,num_neurons):
		self.num_neurons = num_neurons
		self.previous_num_neurons = previous_num_neurons

		self.weights = self.init_weights()

		self.activation = relu
		# derivative of relu
		self.backtivation = d_relu
		# layer_input = activation of previous layer
		self.layer_input = None
		# dot(layer_input,weights)
		self.layer_product = None
		# derivative of loss with respect to weights
		self.dLdw = None

	def init_weights(self):
		return np.random.normal(size=(self.previous_num_neurons,self.num_neurons))

	def forward(self,layer_input):
		self.layer_input = layer_input
		self.layer_product = np.dot(layer_input,self.weights) 
		
		return self.activation(self.layer_product)

	def backprop(self,gradient):
		# derivative of relu(dot(layer_input,weights)) * gradient
		delta = self.backtivation(self.layer_product) * gradient
		# derivative of loss with respect to weights; prev activation * propogated gradient
		self.dLdw = np.dot(self.layer_input.T,delta)
		# return derivative of loss with respect to layer input
		return np.dot(delta,self.weights.T)

	def weight_update(self):
		# learning rate * derivative of loss with respect to weights
		self.weights -= learning_rate*self.dLdw

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


x = np.random.rand(28,28)
x = x.reshape((1,28,28))

# 3x3 filter, filter one, stride one
conv1 = conv_layer(3,1,1)
# 2x2 max pool, stride one
pool1 = max_pool_layer(2,1)
# 10 fully connected neurons
fc1 = fully_connected_layer(729,10)


out1 = conv1.conv(x)

pool_out_1 = pool1.max_pool(out1).reshape((1,-1))

fc_out_1 = fc1.forward(pool_out_1)

print fc1.backprop(np.array([[1,1,1,1,1,1,1,1,1,1]])).shape


























# x = np.array(
# [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]])