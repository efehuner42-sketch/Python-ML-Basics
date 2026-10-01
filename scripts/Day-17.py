import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the built-in breast cancer dataset from scikit-learn
data = load_breast_cancer()

# Extract features (X) and target labels (y)
X = data.data
y = data.target

# Initialize the Logistic Regression model with a high max_iter to ensure convergence
model = LogisticRegression(max_iter=10000)

# Apply 5-fold cross-validation to evaluate the model reliably across different data splits
skores = cross_val_score(model, X, y, cv=5)

# Print the individual test scores and the average performance across all folds
print("Results of 5 different tests: ", skores)
print("Average success: ", np.mean(skores))