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
		predictions = fc.forward(x[i])[0]
		#print "predictions", predictions
		#print predictions
		index = np.argmax(predictions)
		#print index
		if y[i][index] == 1:
			accuracy += 1

		cost = -np.sum(y[i]*np.log(predictions))
		predictions[0,np.argmax(y[i])] -= 1

		gradient = predictions
		# print "gradient",gradient
		print "cost", cost
		fc.backward(gradient)
		if counter % 200 == 0:
			# print predictions
			print accuracy/200.0
			accuracy = 0

		counter += 1

	