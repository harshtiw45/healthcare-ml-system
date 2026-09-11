import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("datasets/parkinsons.csv")

# Remove identifier column
df = df.drop("name", axis=1)

# Split features and target
X = df.drop("status", axis=1)
y = df["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Parkinson's Model Accuracy:", accuracy)

# Save model
joblib.dump(pipeline, "models/parkinsons_model.pkl")

print("Improved Parkinson's model saved successfully!")