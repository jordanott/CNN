import numpy as np

def relu(layer_input):
	return np.maximum(layer_input,0)
def d_relu(x):
    return 1. * (x > 0)

activation_functions = {
	'relu':[relu,d_relu]
	}