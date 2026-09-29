import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. IMPORT NECESSARY LIBRARIES
# ==========================================
# Import pandas, KNeighborsClassifier, and StandardScaler


# ==========================================
# 2. CREATE THE DUMMY DATASET (X and y)
# ==========================================
# Create a DataFrame containing age, salary, and purchase status
df = pd.DataFrame({
    'Yas': [25, 35, 45, 20, 55, 60],
    'Maas': [40000, 60000, 80000, 20000, 120000, 100000],
    'Satin_Alma': [0, 0, 1, 0, 1, 1]
})

# Extract features (X) and target variable (y) from the DataFrame
X = df[['Yas', 'Maas']].values
y = df['Satin_Alma'].values


# ==========================================
# 3. FEATURE SCALING (Very Important for KNN!)
# ==========================================
# Standardize features so that salary (e.g., 120,000) does not overpower age (e.g., 55)
scaler = StandardScaler()
X_olcekli = scaler.fit_transform(X)


# ==========================================
# 4. TRAIN THE K-NN MODEL (.fit)
# ==========================================
# Set the number of neighbors (K) to 3 and fit the scaled data
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_olcekli, y)


# ==========================================
# 5. MAKE A NEW PREDICTION (.predict)
# ==========================================
# Predict whether a new customer who is 40 years old and has a salary of 70,000 will buy the car
# Note: You must scale the new customer's data using the same scaler before predicting!
yeni_musteri = scaler.transform([[40, 70000]])
print(model.predict(yeni_musteri))