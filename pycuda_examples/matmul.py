import numpy as np
import time
# import pycuda stuff
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

BLOCK_SIZE = 16

n = 4
ni = np.int32(n)

# matrix A 
a = np.random.randn(2, 3)
a = np.arange(6).reshape((2,3))
a = a.astype(np.float32)

# matrix B
b = np.random.randn(3, 4)*100
b = np.arange(12).reshape((3,4))
b = b.astype(np.float32)

# matrix B
c = np.empty([2, 4])
c = c.astype(np.float32)

# allocate memory on device
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# copy matrix to memory
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# compile kernel
mod = SourceModule(open("kernels.cu", "r").read())

# get function
matmul = mod.get_function("matmul")


# set grid size
if n%BLOCK_SIZE != 0:
    grid=(n/BLOCK_SIZE+1,n/BLOCK_SIZE+1,1)
else:
    grid=(n/BLOCK_SIZE,n/BLOCK_SIZE,1)
ni = np.int32(n)
# call gpu function
start = time.time()
matmul(a_gpu, b_gpu,c_gpu,np.int32(2),np.int32(3),np.int32(3),np.int32(4),np.int32(2),np.int32(4),block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(1,1))
end = time.time()
print "Time: %.5f s"%(end-start)

# copy back the result
cuda.memcpy_dtoh(c, c_gpu)

print np.linalg.norm(c - np.dot(a,b))
print c
print np.dot(a,b)
print c - np.dot(a,b)