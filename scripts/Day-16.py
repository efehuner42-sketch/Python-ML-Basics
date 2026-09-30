import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. GENERATE AN IMBALANCED DATASET (SIMULATION)
# ==========================================
# Generate 100,000 synthetic samples with 95% majority (0) and 5% minority (1) classes[cite: 3, 4]
X, y = make_classification(
    n_samples=100000, 
    n_features=2, 
    n_redundant=0, 
    weights=[0.95, 0.05], 
    random_state=42
)

# Convert to a DataFrame to inspect the data and class distribution (Assignment Step)
df = pd.DataFrame(X, columns=["Ozellik1", "Ozellik2"])
df["Hedef"] = y

# ==========================================
# 2. CHECK THE CURRENT STATUS
# ==========================================
print("--- Original Data Class Distribution ---")
print(df["Hedef"].value_counts()) #[cite: 3]

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict with the original (imbalanced) model
model_orijinal = LogisticRegression()
model_orijinal.fit(X_train, y_train)
tahmin_original = model_orijinal.predict(X_test)

print("\n=== ORIGINAL (IMBALANCED) DATA REPORT ===")
print(classification_report(y_test, tahmin_original))


# ==========================================
# 3. BALANCE THE SCALE WITH SMOTE
# ==========================================
# Apply SMOTE only to the training data to oversample the minority class (prevents data leakage)[cite: 3]
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


# ==========================================
# 4. CHECK NEW STATUS (BALANCED DATA)
# ==========================================
print("\n--- SMOTE Balanced Training Data Class Distribution ---")
print(pd.Series(y_train_balanced).value_counts()) #[cite: 3]

# Train the model using the balanced data and the advanced class_weight parameter
model_balanced = LogisticRegression(class_weight="balanced", random_state=42)
model_balanced.fit(X_train_balanced, y_train_balanced)
tahmin_balanced = model_balanced.predict(X_test)

print("\n=== SMOTE BALANCED DATA REPORT ===")
print(classification_report(y_test, tahmin_balanced))