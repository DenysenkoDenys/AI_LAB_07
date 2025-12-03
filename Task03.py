import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

data = np.loadtxt('data_clustering.txt', delimiter=',')
print("Перші 5 записів у наборі даних:\n", data[:5])

bandwidth_value = estimate_bandwidth(data, quantile=0.2, n_samples=200)
print(f"\nРозрахована ширина ядра: {bandwidth_value:.3f}")

meanshift_model = MeanShift(bandwidth=bandwidth_value, bin_seeding=True)
meanshift_model.fit(data)

labels = meanshift_model.labels_
centroids = meanshift_model.cluster_centers_

num_clusters = len(np.unique(labels))
print(f"\nКількість знайдених кластерів: {num_clusters}")
print("\nКоординати центрів кластерів:")
print(centroids)

plt.figure(figsize=(7, 6))
palette = plt.cm.tab10(np.linspace(0, 1, num_clusters))

for idx, color in zip(range(num_clusters), palette):
    cluster_points = (labels == idx)
    plt.scatter(data[cluster_points, 0], data[cluster_points, 1],
                color=color, edgecolor='k', s=50, label=f'Кластер {idx + 1}')
    plt.scatter(centroids[idx, 0], centroids[idx, 1],
                color='black', marker='X', s=200)

plt.title("Кластеризація методом Mean Shift")
plt.xlabel("X координата")
plt.ylabel("Y координата")
plt.legend()
plt.grid(True)
plt.show()
