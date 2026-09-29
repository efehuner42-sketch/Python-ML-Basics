import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==========================================
# 1. IMPORT NECESSARY LIBRARIES
# ==========================================
# Import numpy, pandas, train_test_split, and LogisticRegression


# ==========================================
# 2. CREATE CLASSIFICATION DATASET (X and y)
# ==========================================
# Create a DataFrame with study hours and exam status (0 or 1)[cite: 7]
df = pd.DataFrame({
  "Calisma_Saati": [1,2,3,4,5,6,7,8],
  "Sinav_Durumu":[0,0,0,0,1,1,1,1]
})

X = df[["Calisma_Saati"]].values
y = df["Sinav_Durumu"].values

# Split the dataset into training and test sets[cite: 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# ==========================================
# 3. TRAIN THE LOGISTIC REGRESSION MODEL (.fit)
# ==========================================
# Initialize the Logistic Regression model and fit it using the training data[cite: 7]
model = LogisticRegression()
model.fit(X_train, y_train)


# ==========================================
# 4. MAKE CLASSIFICATION PREDICTION (.predict)
# ==========================================
# Predict whether a student who studies 4.5 hours will pass (1) or fail (0)[cite: 7]
print(model.predict([[4.5]]))


# ==========================================
# 5. BONUS CHALLENGE (Probability Prediction)
# ==========================================
# Use predict_proba to check the exact probabilities for 0 and 1 classes[cite: 7]
print(model.predict_proba([[4.5]]))