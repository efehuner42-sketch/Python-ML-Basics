# 1. IMPORT REQUIRED LIBRARIES
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 2. LOAD AND SPLIT DATA (USING return_X_y=True)
X, y = load_breast_cancer(return_X_y=True)

# FIXED: CHANGED x_test TO X_test (UPPERCASE X FOR MATRICES)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# 3. TRAIN AND EVALUATE RANDOM FOREST
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

# USING F-STRING FOR CLEANER OUTPUT
print(f"Success of Random Forest: {accuracy_score(y_test, rf_prediction):.4f}")

# 4. TRAIN AND EVALUATE GRADIENT BOOSTING
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_prediction = gb_model.predict(X_test)

# USING F-STRING FOR CLEANER OUTPUT
print(f"Success of Gradient Boosting: {accuracy_score(y_test, gb_prediction):.4f}")