import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("datasets/heart.csv")

print("Dataset Loaded Successfully\n")
print("Columns:")
print(df.columns)


# Convert categorical columns to numeric
df = pd.get_dummies(df)


# Features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# Save model
joblib.dump(model, "models/heart_model.pkl")

print("\nHeart disease model saved successfully!")