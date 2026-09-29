import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
# Load the Pima Indians Diabetes dataset from CSV file
df = pd.read_csv("diabetes.csv")

# Exploratory Data Analysis (EDA) checks (optional/commented)
# print(df.info())
# print(df.isna().sum())
# print(df.duplicated())
# print(df.describe())

# Extract features (X) and target variable (y) from the DataFrame
X = df.drop(["Outcome"], axis=1).values
y = df["Outcome"].values


# ==========================================
# 2. CORRELATION HEATMAP VISUALIZATION
# ==========================================
# Calculate correlation matrix and visualize it using a heatmap
corr = df.corr()
sns.heatmap(data=corr, square=True, cmap="RdBu", annot=True, annot_kws={"fontsize":11, "fontweight": "bold"})
plt.show()


# ==========================================
# 3. SPLIT AND SCALE THE DATASET
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features for models sensitive to scale (Logistic Regression & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# ==========================================
# 4. TRAIN MODELS AND MAKE PREDICTIONS
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
# 5. EVALUATE MODEL PERFORMANCES
# ==========================================
# Calculate accuracy scores for each model
skor1 = accuracy_score(y_test, tahmin1)
skor2 = accuracy_score(y_test, tahmin2)
skor3 = accuracy_score(y_test, tahmin3)

# print("Logistic basari orani: " , skor1)
# print("KNN basari orani: " , skor2)
# print("Tree basari orani: " , skor3)


# ==========================================
# 6. VISUALIZE MODEL ACCURACY COMPARISON
# ==========================================
# Prepare data for bar chart comparison
model_isimleri = ["Logistic", "KNN", "Tree"]
basari_oranlari = [skor1, skor2, skor3]

# Create and display the accuracy comparison bar chart
plt.figure(figsize=(12,6))
plt.bar(x=model_isimleri, height=basari_oranlari, color=["blue", "red", "green"])

plt.title("Modellerin Basari Oranlarinin Karsilastirilmasi")
plt.xlabel("Modeller")
plt.ylabel("Basari Oranlari")
plt.ylim(0, 1.00)

plt.show()