import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

import joblib
# Load the dataset from the CSV file
# data = pd.read_csv('/Users/kothavamsi/Desktop/jaundice_vs/jaundice_synthetic_data_cleaned.csv')
# data = pd.read_csv('/content/jaundice_challenging_dataset_40000.csv')
# data = pd.read_csv('/Users/kothavamsi/Desktop/jaundice_vs/jaundice_dataset.csv')
data = pd.read_csv('/Users/kothavamsi/Desktop/jaundice_vs/Liver_Patient_Dataset_Balanced.csv')
# Step 1: Split the data into features and target
# X = data.drop(columns=['Target'])
# y = data['Target']

# X = data.drop(columns=['Jaundice'])
# y = data['Jaundice']

# X = data.drop(columns=['jaundice_risk'])
# y = data['jaundice_risk']

X = data.drop(columns=['Result'])
y = data['Result']

imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))


xgb_model.save_model('xgb_model.json')  # Corrected to save the trained model

# joblib.dump(xgb_model, 'xgb_model.pkl')
# # joblib.dump(scaler, 'scaler.pkl')
# # Train the scaler on X_train
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)


joblib.dump(xgb_model, "xgb_model.pkl")

# Assuming you have X_train already prepared
scaler = StandardScaler()
scaler.fit(X_train)  # Fit it to the training data

# Save it again
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved successfully!")

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.show()

from imblearn.over_sampling import SMOTE
import pandas as pd

# Load your dataset
df = pd.read_csv("Liver_Patient_Dataset_Balanced.csv")  # Update with your file path

# Separate features (X) and target (y)
X = df.drop(columns=["Result"])
y = df["Result"]

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Fix target variable labels (convert 2 → 1)
y_resampled = y_resampled.map({1: 0, 2: 1})

# Convert back to DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced["Result"] = y_resampled

# Save the balanced dataset
df_balanced.to_csv("Liver_Patient_Dataset_Balanced.csv", index=False)

print("SMOTE applied successfully. Balanced dataset saved as 'Liver_Patient_Dataset_Balanced.csv'.")

import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Get model predictions (probabilities for ROC curve)
y_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Print AUC score
print(f"AUC Score: {roc_auc:.4f}")
