import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

# 生成随机数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 进行 K-means 聚类
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)

# 获取聚类中心的位置
centers = kmeans.cluster_centers_

# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis')

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering with 10 Centers')
plt.savefig('scatter_plot_with_new_clustering.png')
plt.show()
