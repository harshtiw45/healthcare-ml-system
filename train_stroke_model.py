import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("datasets/stroke.csv")

# Fill missing BMI
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Drop ID
df = df.drop("id", axis=1)

# Encode categorical features
df = pd.get_dummies(df)

# Features / Target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train better model
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=10,   # helps imbalance
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/stroke_model.pkl")

print("Improved stroke model saved.")