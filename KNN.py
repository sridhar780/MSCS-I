import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
file_path = r"E:\Downloads\Chronic_Kidney_Dsease_data.csv" # Replace with the correct file path
data = pd.read_csv(file_path)

# Step 2: Drop non-predictive columns
data = data.drop(columns=["PatientID", "DoctorInCharge"], axis=1)

# Step 3: Handle categorical columns
# Convert categorical columns (Gender, Ethnicity, etc.) to numeric using Label Encoding
categorical_columns = data.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Step 4: Separate features (X) and target (y)
X = data.drop(columns=["Diagnosis"], axis=1)  # Input features
y = data["Diagnosis"]  # Target variable

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)  # k=5 (default)
knn.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = knn.predict(X_test)

# Step 9: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
