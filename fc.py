import sys
sys.path.insert(0, 'net/')

from net import net
import numpy as np


fc = net(.1)
# height,width,depth
fc.add_layer('input',shape=(3,1))
fc.add_layer('fc',num_neurons=2,activation='sigmoid')
fc.add_layer('output',num_neurons=2,activation='softmax')

x = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
y = [[0,1],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[1,0]]
x = np.array(x)
counter = 0
accuracy = 0
while True:
	for i in range(0,len(x)):
		predictions = fc.forward(x[i])
		#print "predictions", predictions
		#print predictions
		index = np.argmax(predictions)
		#print index
		if y[i][index] == 1:
			accuracy += 1

		
		gradient = fc.get_gradient(predictions,y[i])
		loss = fc.get_cost(predictions,y[i])
		# print "gradient",gradient
		#print "cost", loss
		fc.backward(gradient)
		if counter % 200 == 0:
			# print predictions
			print accuracy/200.0
			accuracy = 0

		counter += 1

	