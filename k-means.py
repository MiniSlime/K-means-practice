import numpy as np
import matplotlib.pyplot as plt
import time

t = int(time.time())
print(t)
np.random.seed(t)

x = np.r_[np.random.normal(size=1000,loc=0,scale=1),
          np.random.normal(size=1000,loc=4,scale=1),
          np.random.normal(size=1000,loc=2,scale=1)]

y = np.r_[np.random.normal(size=1000,loc=10,scale=1),
          np.random.normal(size=1000,loc=15,scale=1),
          np.random.normal(size=1000,loc=20,scale=1)]
data = np.c_[x, y]

p = plt.subplot()
p.scatter(data[:,0], data[:,1], c = "black", alpha = 0.5)
p.set_aspect('equal')
plt.show()

n_clusters = 3
max_iter = 300
clusters = np.random.randint(0,n_clusters,data.shape[0])

for _ in range(max_iter):
    centroids = np.array([data[clusters == n, :].mean(axis = 0) for n in range(n_clusters)])
    new_clusters = np.array([np.linalg.norm(data - c, axis = 1) for c in centroids]).argmin(axis = 0)

    for n in range(n_clusters):
        if not np.any(new_clusters == n):
            centroids[n] = data[np.random.choice(data.shape[0], 1), :]

    if np.allclose(clusters, new_clusters):
        break

    clusters = new_clusters

p = plt.subplot()

p.scatter(data[clusters==0, 0], data[clusters==0, 1], c = 'red')
p.scatter(data[clusters==1, 0], data[clusters==1, 1], c = 'white', edgecolors='black')
p.scatter(data[clusters==2, 0], data[clusters==2, 1], c = 'blue')
# 中心点
p.scatter(centroids[:, 0], centroids[:, 1], color='orange', marker='s')

p.set_aspect('equal')

plt.show()