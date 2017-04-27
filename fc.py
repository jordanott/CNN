import sys
sys.path.insert(0, 'net/')

from net import net
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import math

fc = net(.1)
# height,width,depth
fc.add_layer('input',shape=(2,1))
fc.add_layer('fc',num_neurons=3,activation='relu')
fc.add_layer('output',num_neurons=2,activation='softmax')

# x = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
# y = [[0,1],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[1,0]]
# x = np.array(x)

steps = 1000
h = 0
k = 0
r = 1
labels = np.zeros((steps))
Y = np.zeros((steps,2))
X = np.zeros((steps,2))
counter = 0
theta = np.arange(0,2*3.14,2*3.14/500)

for i in range(2): 
    for t in theta:
        X_ = h + r*math.cos(t) + np.random.randn()*.8
        Y_ = k - r*math.sin(t) + np.random.randn()*.8
        X[counter] = np.array([X_,Y_])
        labels[counter] = i
        Y[counter,i] = 1
        counter +=1
    r += 5

# plt.scatter(X[:,0],X[:,1],c=labels,s=40,cmap=plt.cm.Spectral)
# plt.show()


counter = 0
accuracy = 0
for _ in range(1):
	for i in range(0,len(X)):
		predictions = fc.forward(X[i])
		#print "predictions", predictions
		#print predictions
		index = np.argmax(predictions)
		#print index
		if Y[i][index] == 1:
			accuracy += 1

		
		gradient = fc.cross_entropy_gradient(predictions,Y[i])
		#loss = fc.get_cost(predictions,y[i])
		
		fc.backward(gradient)
		if counter % 200 == 0:
			# print predictions
			print accuracy/200.0
			accuracy = 0

		counter += 1


# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
new_x = xx.ravel()
new_y = yy.ravel()
print new_x.shape[0]
Z = np.zeros((new_x.shape[0],2))
for i in range(new_x.shape[0]):
	Z[i] = fc.forward(np.array([new_x[i],new_y[i]]))

Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
	