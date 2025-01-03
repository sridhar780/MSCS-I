import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import randint

# Step 1: Load Dataset
file_path = r"E:\Downloads\credit.csv"  # Replace with your file name
data = pd.read_csv(file_path)

# Step 2: Preprocess the Data
# Drop unnecessary columns (if any)
data = data.drop(columns=[], errors='ignore')  # Add columns to drop if needed

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop(columns=["default"])  # Replace "default" with target column name
y = data["default"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train an Overfitted Decision Tree
dt_overfit = DecisionTreeClassifier(random_state=42)
dt_overfit.fit(X_train, y_train)

# Evaluate Overfitted Model
print("Overfitted Decision Tree Results:")
y_pred_train = dt_overfit.predict(X_train)
y_pred_test = dt_overfit.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))

# Visualize Overfitted Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_overfit, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True)
plt.title("Overfitted Decision Tree")
plt.show()

# Step 4: Pruning Techniques (Control Overfitting)
dt_pruned = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
dt_pruned.fit(X_train, y_train)

# Evaluate Pruned Model
y_pred_pruned = dt_pruned.predict(X_test)
print("\nPruned Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_pruned))
print("Classification Report:\n", classification_report(y_test, y_pred_pruned))

# Step 5: Grid Search for Hyperparameter Optimization
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluate Best Model from Grid Search
print("\nBest Parameters (Grid Search):", grid_search.best_params_)
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
print("Accuracy (Grid Search):", accuracy_score(y_test, y_pred_grid))
print("Classification Report:\n", classification_report(y_test, y_pred_grid))

# Step 6: Random Search for Hyperparameter Optimization
param_dist = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_dist,
                                   n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Evaluate Best Model from Random Search
print("\nBest Parameters (Random Search):", random_search.best_params_)
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
print("Accuracy (Random Search):", accuracy_score(y_test, y_pred_random))
print("Classification Report:\n", classification_report(y_test, y_pred_random))

# Step 7: Cross-Validation for Final Model
final_model = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
cv_scores = cross_val_score(final_model, X, y, cv=10, scoring='accuracy')

print("\nCross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))
