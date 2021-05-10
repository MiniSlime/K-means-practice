import numpy as np
import matplotlib.pyplot as plt
import time

t = int(time.time())
print(t)
np.random.seed(t)

x = np.r_[np.random.normal(size=1000,loc=0,scale=1),
          np.random.normal(size=1000,loc=4,scale=1)]

y = np.r_[np.random.normal(size=1000,loc=10,scale=1),
          np.random.normal(size=1000,loc=10,scale=1)]
data = np.c_[x, y]

p = plt.subplot()
p.scatter(data[:,0], data[:,1], c = "black", alpha = 0.5)
p.set_aspect('equal')
plt.show()