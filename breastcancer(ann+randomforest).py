# -*- coding: utf-8 -*-
"""breastcancer(ann+randomforest).ipynb


"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

file_name = "Breastcancer dataset.csv"
data = pd.read_csv(file_name)

print(data.head())
print(data.info())



# Data Preprocessing
# Remove unnecessary columns
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Encode diagnosis column: M = 0 (Malignant), B = 1 (Benign)
data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})

# Separate features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest for Feature Selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Select important features
importances = rf.feature_importances_
important_features_indices = np.where(importances > 0.01)[0]  # Threshold for feature selection
X_train_selected = X_train_scaled[:, important_features_indices]
X_test_selected = X_test_scaled[:, important_features_indices]

# Build and Train ANN
model = Sequential([
    Dense(64, input_dim=X_train_selected.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the ANN
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN
history = model.fit(X_train_selected, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate ANN
ann_predictions = model.predict(X_test_selected).flatten()
ann_binary_predictions = (ann_predictions > 0.5).astype(int)

ann_accuracy = accuracy_score(y_test, ann_binary_predictions)
ann_roc_auc = roc_auc_score(y_test, ann_predictions)
print(f"ANN Accuracy: {ann_accuracy:.2f}")
print(f"ANN ROC AUC: {ann_roc_auc:.2f}")

# Combine ANN and Random Forest Predictions (Ensemble)
rf_predictions = rf.predict_proba(X_test_scaled)[:, 1]
ensemble_predictions = (rf_predictions + ann_predictions) / 2  # Average predictions
ensemble_binary_predictions = (ensemble_predictions > 0.5).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_binary_predictions)
ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}")
print(f"Ensemble ROC AUC: {ensemble_roc_auc:.2f}")

# Plot ANN Training Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('ANN Training Accuracy')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# ANN Evaluation
print("=== ANN Evaluation ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, ann_binary_predictions))
print("\nClassification Report:")
print(classification_report(y_test, ann_binary_predictions))
print(f"Precision: {precision_score(y_test, ann_binary_predictions):.2f}")
print(f"Recall: {recall_score(y_test, ann_binary_predictions):.2f}")
print(f"F1-Score: {f1_score(y_test, ann_binary_predictions):.2f}")



# Ensemble Evaluation
print("\n=== Ensemble Evaluation ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, ensemble_binary_predictions))
print("\nClassification Report:")
print(classification_report(y_test, ensemble_binary_predictions))
print(f"Precision: {precision_score(y_test, ensemble_binary_predictions):.2f}")
print(f"Recall: {recall_score(y_test, ensemble_binary_predictions):.2f}")
print(f"F1-Score: {f1_score(y_test, ensemble_binary_predictions):.2f}")

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation on Random Forest
rf_cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("\n=== Random Forest Cross-Validation ===")
print(f"Cross-Validation Accuracy Scores: {rf_cv_scores}")
print(f"Mean Accuracy: {rf_cv_scores.mean():.2f}")
print(f"Standard Deviation: {rf_cv_scores.std():.2f}")

from sklearn.metrics import roc_curve, auc, log_loss, matthews_corrcoef

# Compute and Plot ROC Curve for ANN
fpr, tpr, thresholds = roc_curve(y_test, ann_predictions)
roc_auc_ann = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_ann:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - ANN')
plt.legend(loc='lower right')
plt.show()

# Compute Matthews Correlation Coefficient (MCC) for ANN
mcc_ann = matthews_corrcoef(y_test, ann_binary_predictions)

# Log Loss for ANN
log_loss_ann = log_loss(y_test, ann_predictions)

# Specificity for ANN
tn, fp, fn, tp = confusion_matrix(y_test, ann_binary_predictions).ravel()
specificity_ann = tn / (tn + fp)

# Print Additional Metrics for ANN
print("\n=== ANN Additional Metrics ===")
print(f"ROC AUC: {roc_auc_ann:.2f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_ann:.2f}")
print(f"Log Loss: {log_loss_ann:.2f}")
print(f"Specificity: {specificity_ann:.2f}")

# Repeat for Ensemble Model
fpr_ensemble, tpr_ensemble, thresholds_ensemble = roc_curve(y_test, ensemble_predictions)
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

mcc_ensemble = matthews_corrcoef(y_test, ensemble_binary_predictions)
log_loss_ensemble = log_loss(y_test, ensemble_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, ensemble_binary_predictions).ravel()
specificity_ensemble = tn / (tn + fp)

print("\n=== Ensemble Additional Metrics ===")
print(f"ROC AUC: {roc_auc_ensemble:.2f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_ensemble:.2f}")
print(f"Log Loss: {log_loss_ensemble:.2f}")
print(f"Specificity: {specificity_ensemble:.2f}")