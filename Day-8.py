import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. IMPORT NECESSARY LIBRARIES
# ==========================================
# (Libraries imported above)[cite: 4]


# ==========================================
# 2. CREATE A DUMMY DATASET (X and y)
# ==========================================
# Create a sample DataFrame with square meters (X) and prices (y)[cite: 4]
df = pd.DataFrame({
    'Metrekare': [50, 60, 80, 100, 120, 150],
    'Fiyat': [150000, 180000, 240000, 300000, 360000, 450000]
})

# Define features (X) and target (y)[cite: 4]
X = df[["Metrekare"]]
y = df["Fiyat"]


# ==========================================
# 3. SPLIT THE DATA (Train & Test Split)
# ==========================================
# Split 80% of the data for training and 20% for testing with a random state[cite: 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================================
# 4. TRAIN THE MODEL (.fit)
# ==========================================
# Initialize the Linear Regression model and fit it using the training data[cite: 4]
model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 5. MAKE PREDICTIONS (.predict)
# ==========================================
# Predict the price for a house size that the model has never seen before, e.g., 110 square meters[cite: 4]
predicted_price = model.predict([[110]])

print("Predicted Price for 110 sqm:", predicted_price)