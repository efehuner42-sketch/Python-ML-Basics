from sklearn.svm import SVC 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset with 2 distinct clusters using make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model with a linear kernel
model1 = SVC(kernel="linear")
model1.fit(X_train, y_train)

# Make predictions and calculate accuracy for the linear model
predictions1 = model1.predict(X_test)
score1 = accuracy_score(y_test, predictions1)

# Initialize and train the SVM model with an RBF (Radial Basis Function) kernel
model2 = SVC(kernel="rbf")
model2.fit(X_train, y_train)

# Make predictions and calculate accuracy for the RBF model
predictions2 = model2.predict(X_test)
score2 = accuracy_score(y_test, predictions2)

# Print the success rates of both models
print("SVM(linear) success rate: ", score1)
print("SVM(rbf) success rate: ", score2)