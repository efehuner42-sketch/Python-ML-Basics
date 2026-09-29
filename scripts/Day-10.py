import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. IMPORT METRICS & LIBRARIES
# ==========================================
# Include error calculation functions in your code


# ==========================================
# 2. CREATE DATASET & TRAIN/TEST SPLIT
# ==========================================
df = pd.DataFrame({
    'Metrekare': [80, 90, 100, 120, 150, 180],
    'Oda_Sayisi': [2, 3, 3, 4, 4, 5],
    'Bina_Yasi': [10, 8, 5, 2, 1, 0],
    'Fiyat': [200000, 230000, 280000, 350000, 420000, 500000]
})

X = df.drop(["Fiyat"], axis=1).values
y = df["Fiyat"].values

# Split the dataset with properties into 80% Train and 20% Test[cite: 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ==========================================
# 3. TRAIN THE MODEL & PREDICT TEST SET
# ==========================================
# Train the model with training data, then pass X_test features to get predictions[cite: 4]
model = LinearRegression()
model.fit(X_train, y_train)

y_tahmin = model.predict(X_test)


# ==========================================
# 4. CALCULATE ERRORS
# ==========================================
# Compare actual prices (y_test) and model's predicted prices (y_tahmin)[cite: 4]
# MAE (Mean Absolute Error)[cite: 4]
mae = mean_absolute_error(y_test, y_tahmin)

# R-Squared (Success Rate / Coefficient of Determination)[cite: 4]
r2 = r2_score(y_test, y_tahmin)


# ==========================================
# 5. PRINT & INTERPRET RESULTS
# ==========================================
print("MAE (Mean Absolute Error):", mae)
print("R2 Score:", r2)


