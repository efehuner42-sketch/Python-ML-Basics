from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset and define features (X) and target labels (y)
data = load_wine()
X = data.data
y = data.target

# Split the data into 70% training and 30% testing sets for proper evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# STEP 1: OVERFITTING (MEMORIZATION)
# ==========================================
# Build a Decision Tree with no restrictions (max_depth=None) to let it memorize the data
model_memorization = DecisionTreeClassifier(max_depth=None, random_state=42)
model_memorization.fit(X_train, y_train)

print("--- MODEL_MEMORIZATION (Unrestricted Tree) ---")
# Training success will likely be 1.0 (100%) because the model memorized every detail
print("Train's success: ",  accuracy_score(y_train, model_memorization.predict(X_train)))
print("Test's success: ",  accuracy_score(y_test, model_memorization.predict(X_test)))

# ==========================================
# STEP 2: REGULARIZATION (PRUNING)
# ==========================================
# Restrict the tree depth (max_depth=3) to prevent overfitting and force generalization
model_regular = DecisionTreeClassifier(max_depth=3, random_state=42)
model_regular.fit(X_train, y_train)

print("\n--- MODEL_REGULAR (Pruned Tree) ---")
# Training success drops slightly, but the gap between Train and Test closes, meaning it generalizes better
print("Train's success: ",  accuracy_score(y_train, model_regular.predict(X_train)))
print("Test's success: ",  accuracy_score(y_test, model_regular.predict(X_test)))