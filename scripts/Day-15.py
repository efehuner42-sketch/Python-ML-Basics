import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
# Load the Pima Indians Diabetes dataset from CSV file
df = pd.read_csv("diabetes.csv")

# Extract features (X) and target variable (y) from the DataFrame
X = df.drop(["Outcome"], axis=1).values
y = df["Outcome"].values


# ==========================================
# 2. SPLIT AND SCALE THE DATASET
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for models sensitive to scale (Logistic Regression & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 3. TRAIN MODELS AND MAKE PREDICTIONS
# ==========================================
# Model 1: Logistic Regression
model1 = LogisticRegression()
model1.fit(X_train_scaled, y_train)
tahmin1 = model1.predict(X_test_scaled)

# Model 2: K-Nearest Neighbors (KNN)
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train_scaled, y_train)
tahmin2 = model2.predict(X_test_scaled)

# Model 3: Decision Tree (uses unscaled training data)
model3 = DecisionTreeClassifier(random_state=42)
model3.fit(X_train, y_train)
tahmin3 = model3.predict(X_test)


# ==========================================
# 4. CONFUSION MATRIX CALCULATIONS
# ==========================================
from sklearn.metrics import confusion_matrix, classification_report

matris1 = confusion_matrix(y_test, tahmin1)
matris2 = confusion_matrix(y_test, tahmin2)
matris3 = confusion_matrix(y_test, tahmin3)


# ==========================================
# 5. VISUALIZE CONFUSION MATRICES (HEATMAPS)
# ==========================================
# Logistic Regression Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(matris1, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 11, "fontweight": "bold"})
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Modelin Tahminleri')
plt.show()

# KNN Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(matris2, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 11, "fontweight": "bold"})
plt.title('Confusion Matrix - KNN')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Modelin Tahminleri')
plt.show()

# Decision Tree Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(matris3, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 11, "fontweight": "bold"})
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Modelin Tahminleri')
plt.show()


# ==========================================
# 6. PRINT CLASSIFICATION REPORTS (KARNELER)
# ==========================================
print("=== LOGISTIC REGRESSION REPORT ===")
print(classification_report(y_test, tahmin1))

print("=== KNN REPORT ===")
print(classification_report(y_test, tahmin2))

print("=== DECISION TREE REPORT ===")
print(classification_report(y_test, tahmin3))