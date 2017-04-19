import numpy as np

class output_layer():
	def __init__(self,layer_opts):
		functions = {
			'softmax':[self.softmax,self.d_softmax]
		}
		self.forward = functions[layer_opts['output_function']][0]
		self.backprop = functions[layer_opts['output_function']][1]

		# set shape of incoming data
		self.incoming_shape = layer_opts['incoming_shape']
		# set output shape of layer
		self.output_shape = self.incoming_shape

	def softmax(self,z):
		# clamp gradient to avoid explosion
		z = np.minimum(500,z)
		# softmax function
		return np.exp(z) / np.sum(np.exp(z))

	def d_softmax(self,y):
		# derivative of softmax function
		return y*(1-y)
