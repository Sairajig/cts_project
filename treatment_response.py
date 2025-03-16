import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
data = pd.read_csv('disease_patient_data_extended_v2.csv')  # Replace with the correct file path

# Print the column names to check
print("Column Names:", data.columns)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Preprocess Data
# Handle missing values (for both numeric and categorical features)
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])  # For categorical features
    else:
        data[column] = data[column].fillna(data[column].mean())  # For numeric features

# Ensure that 'response' column exists in the dataset
if 'response' not in data.columns:
    raise ValueError("The 'response' column is missing from the dataset.")

# Convert categorical features to one-hot encoding
data = pd.get_dummies(data, columns=['gender', 'disease', 'treatment_type', 'alternative_treatment'], drop_first=True)

# Split features and target variable
X = data.drop(columns=['response'])  # Features
y = data['response']  # Target (treatment response)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (fit only on training data, then transform both)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict with Random Forest
y_pred_rf = rf_model.predict(X_val)

# Evaluate Random Forest
accuracy_rf = accuracy_score(y_val, y_pred_rf)
print(f'Random Forest Treatment Response Prediction Accuracy: {accuracy_rf * 100:.2f}%')
print(classification_report(y_val, y_pred_rf))

# Feature Importance for Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# Build and train a Neural Network
nn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the Neural Network
loss, accuracy_nn = nn_model.evaluate(X_val, y_val)
print(f'Neural Network Treatment Response Prediction Accuracy: {accuracy_nn * 100:.2f}%')
