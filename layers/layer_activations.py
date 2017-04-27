import numpy as np

def relu(x):
	return np.maximum(x,0)
def d_relu(x):
    return 1. * (x > 0)

def tanh(x):
	return np.tanh(x)

def d_tanh(x):
	return 1.0 - np.tanh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
	# softmax function
	return np.exp(x) / np.sum(np.exp(x))

def d_softmax(x):
	# derivative of softmax function
	return x

activation_functions = {
	'relu':[relu,d_relu],
	'tanh':[tanh,d_tanh],
	'sigmoid':[sigmoid,d_sigmoid],
	'softmax':[softmax,d_softmax]
	}