import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("datasets/breast_cancer.csv")

print("Dataset Loaded")
print(df.head())


# ==============================
# DATA CLEANING
# ==============================

# Drop unnecessary columns if they exist
if "id" in df.columns:
    df = df.drop("id", axis=1)

if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)


# Convert diagnosis column (M/B) to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})


# ==============================
# FEATURE / TARGET SPLIT
# ==============================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


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

print("Model Accuracy:", accuracy)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, "models/cancer_model.pkl")

print("Breast cancer model saved successfully!")