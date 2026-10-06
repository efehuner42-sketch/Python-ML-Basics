from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load the built-in wine dataset
wine_data = load_wine()

# Extract features (X) and target labels (y)
X = wine_data.data
y = wine_data.target

# Split the dataset into 70% training and 30% testing sets (added random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the hyperparameter grid to search through
param_grid = {
  "n_estimators": [50, 100, 200],
  "max_depth": [None, 5, 10]
}

# Initialize the Random Forest model (added random_state) and GridSearchCV with 5-fold cross-validation
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# Fit the grid search to find the optimal hyperparameter combination
grid_search.fit(X_train, y_train)

# Output the best parameters and the highest cross-validation score
print("The best parameters: ", grid_search.best_params_)
print("The best score: ", grid_search.best_score_)