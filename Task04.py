import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

data = np.loadtxt('data_clustering.txt', delimiter=',')
print("Перші п’ять рядків набору даних:\n", data[:5])

affinity_model = AffinityPropagation(damping=0.8, random_state=42)
affinity_model.fit(data)

labels = affinity_model.labels_
centers = affinity_model.cluster_centers_
num_clusters = len(np.unique(labels))

print(f"\nКількість знайдених кластерів: {num_clusters}")
print("\nКоординати центрів кластерів:")
print(centers)

plt.figure(figsize=(7, 6))
palette = plt.cm.tab10(np.linspace(0, 1, num_clusters))

for i, color in zip(range(num_clusters), palette):
    cluster_points = (labels == i)
    plt.scatter(data[cluster_points, 0], data[cluster_points, 1],
                s=50, color=color, edgecolor='k', label=f'Кластер {i + 1}')
    plt.scatter(centers[i, 0], centers[i, 1],
                color='black', marker='X', s=200, edgecolor='white')

plt.title("Кластеризація за допомогою Affinity Propagation")
plt.xlabel("X координата")
plt.ylabel("Y координата")
plt.legend()
plt.grid(True)
plt.show()
