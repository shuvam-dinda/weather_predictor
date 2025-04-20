# weather_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
data = pd.read_csv("weather_data.csv")  # Make sure this matches your filename
print("Data Loaded Successfully!\n")
print(data.head())

# Step 2: Basic info
print("\nData Info:")
print(data.info())

# Step 3: Drop rows with missing values
data.dropna(inplace=True)

# Step 4: Encode the target variable (weather)
le = LabelEncoder()
data['weather'] = le.fit_transform(data['weather'])  # Convert strings to numeric labels

# Optional: Print encoded weather mapping
print("\nWeather Label Mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")

# Step 5: Prepare features and target
X = data.drop(['date', 'weather'], axis=1)  # Drop non-numeric and target column
y = data['weather']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)

# Step 9: Evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

