# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 2. LOAD DATASET AND SCALE FEATURES
# ==========================================
data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. DIMENSIONALITY REDUCTION WITH PCA
# ==========================================
# Reduce 30 dimensions down to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Data Shape: ", X_scaled.shape)
print("After PCA Shape: ", X_pca.shape)

# ==========================================
# 4. 2D DATA VISUALIZATION
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="plasma")
plt.xlabel("First Principal Component (PCA 1)")
plt.ylabel("Second Principal Component (PCA 2)")
plt.title("Displaying 30-Dimensional Data in 2 Dimensions")
plt.show()

# ==========================================
# 5. MODEL BENCHMARK: ORIGINAL (SCALED) DATA
# ==========================================
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_scaled = LogisticRegression(random_state=42)
model_scaled.fit(X_train_orig, y_train_orig)
predictions_scaled = model_scaled.predict(X_test_orig)

print("Success of Scaled (All Features): ", accuracy_score(y_test_orig, predictions_scaled))

# ==========================================
# 6. MODEL BENCHMARK: REDUCED (PCA) DATA
# ==========================================
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

model_pca = LogisticRegression(random_state=42)
model_pca.fit(X_train_pca, y_train_pca)
predictions_pca = model_pca.predict(X_test_pca)

print("Success of PCA (2 Features): ", accuracy_score(y_test_pca, predictions_pca))