import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("datasets/diabetes.csv")

print("Dataset Loaded Successfully\n")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


# ==============================
# DATA PREPROCESSING
# ==============================

# Convert categorical columns to numeric
df = pd.get_dummies(df)

print("\nAfter Encoding:")
print(df.head())


# ==============================
# FEATURE / TARGET SPLIT
# ==============================

X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# MODEL TRAINING
# ==============================

model = RandomForestClassifier()

model.fit(X_train, y_train)


# ==============================
# MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, "models/diabetes_model.pkl")

print("\nModel saved successfully in models/diabetes_model.pkl")