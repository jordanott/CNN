import sys
sys.path.insert(0, 'net/')

from net import net
import numpy as np
import time

x = np.random.normal(size=(64,64,1))

cnn = net(.001)
# height,width,depth
cnn.add_layer('input',shape=(64,64,1))
# cnn.add_layer('fc',num_neurons=32,activation='relu')
cnn.add_layer('conv',stride=1,num_filters=1,filter_dim=3,padding=1,activation='relu')
x = cnn.forward(x)
t0 = time.time()
cnn.backward(x)
t1 = time.time()

total = t1-t0
print total
