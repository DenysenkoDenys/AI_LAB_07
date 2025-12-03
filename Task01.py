import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_path = 'data_clustering.txt'
points = np.loadtxt(file_path, delimiter=',')

print("Перші 5 записів із набору:")
print(points[:5])

plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=50, color='royalblue', edgecolor='k')
plt.title("Початкові дані")
plt.xlabel("Ознака X")
plt.ylabel("Ознака Y")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans.fit(points)

cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7, 7))
plt.imshow(
    Z,
    interpolation='nearest',
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Pastel2,
    aspect='auto',
    origin='lower'
)

plt.scatter(points[:, 0], points[:, 1], c=cluster_labels, s=50, cmap='tab10', edgecolor='k')

plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c='black',
    s=200,
    marker='X',
    label='Центри кластерів'
)

plt.title("Кластеризація методом K-середніх")
plt.xlabel("Ознака X")
plt.ylabel("Ознака Y")
plt.legend()
plt.show()

print("Координати знайдених центрів кластерів:")
print(cluster_centers)
