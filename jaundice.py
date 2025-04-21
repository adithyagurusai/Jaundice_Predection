import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Liver Patient Dataset (LPD)_train.csv", encoding="ISO-8859-1")

# Identify categorical columns
categorical_cols = ['Sex']  # Update if there are more categorical columns

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Now apply imputation
X = data.drop(columns=['Target'])  # Replace 'Target' with actual label column name
y = data['Target']  

imputer = SimpleImputer(strategy='mean')  # Use 'most_frequent' if necessary
X_imputed = imputer.fit_transform(X)

print("Preprocessing done successfully!")
