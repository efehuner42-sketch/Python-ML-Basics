# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ==========================================
# 2. GENERATE SYNTHETIC DATA
# ==========================================
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

# ==========================================
# 3. INITIALIZE AND TRAIN MODEL
# ==========================================
model = KMeans(n_clusters=3, init="k-means++", max_iter=300, random_state=42)
model.fit(X)

# ==========================================
# 4. PREDICT CLUSTERS
# ==========================================
cluster_predictions = model.predict(X)

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=cluster_predictions, cmap="viridis")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c="red")
plt.title("Customer Segmentation")
plt.show()