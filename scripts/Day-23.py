# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ==========================================
# 2. GENERATE SYNTHETIC DATA
# ==========================================
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# ==========================================
# 3. CALCULATE INERTIA FOR K VALUES (1-10)
# ==========================================
error_scores = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, random_state=42
    )
    model.fit(X)
    error_scores.append(model.inertia_)

# ==========================================
# 4. PLOT ELBOW METHOD GRAPH
# ==========================================
plt.plot(range(1, 11), error_scores, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.show()