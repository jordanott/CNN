
class input_layer():
    def __init__(self,layer_opts):
        # set shape of incoming data
        self.incoming_shape = layer_opts['incoming_shape']
        # set output shape of layer
        self.output_shape = self.incoming_shape
        
    def forward(self,layer_input):
    	return layer_input, self.l2()

    def backprop(self,gradient):
    	pass

    def l2(self):
		return 0