# 1. IMPORT REQUIRED LIBRARIES
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 2. LOAD AND SPLIT DATA
# USING return_X_y=True TO MAKE IT SHORTER AND CLEANER
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# 3. BUILD THE PIPELINE OBJECT
model = Pipeline([
    ("normalizer", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

# 4. FIT THE PIPELINE WITH A SINGLE COMMAND
model.fit(X_train, y_train)

# 5. MAKE PREDICTIONS AND EVALUATE SUCCESS
predictions = model.predict(X_test)
print(f"Success of Pipeline(K-Neighbors): {accuracy_score(y_test, predictions):.4f}")