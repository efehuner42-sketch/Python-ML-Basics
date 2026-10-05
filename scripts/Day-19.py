# 1. Import Necessary Libraries[cite: 1]
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2. Load and Split the Wine Dataset[cite: 1]
# We will predict 3 different wine types based on their chemical properties[cite: 1].
df = load_wine()

X = df.data
y = df.target

# Splitting the data into 70% training and 30% testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

# 3. First, Train and Test a Single Decision Tree[cite: 1]
dtc = DecisionTreeClassifier(random_state=22)
dtc_model = dtc.fit(X_train, y_train)
dtc_prediction = dtc_model.predict(X_test)

# 4. Now, Train and Test the Random Forest[cite: 1]
# With the n_estimators=200 parameter, we instruct it to build 200 trees[cite: 1].
rfc = RandomForestClassifier(n_estimators=200, random_state=22)
rfc_model = rfc.fit(X_train, y_train)
rfc_prediction = rfc_model.predict(X_test)

# 5. Compare the Results[cite: 1]
# By looking at the two print outputs, you can see the performance of the "ensemble" concept[cite: 1].
print(f"Success rate of DTC: {accuracy_score(y_test, dtc_prediction):.4f}")
print(f"Success rate of RFC: {accuracy_score(y_test, rfc_prediction):.4f}")