import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# 2. Features / Target
features = [
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
X = pd.get_dummies(df[features], drop_first=True)
y = df["diabetes"]

print("=== Dataset shape ===")
print(f"X: {X.shape}, y: {y.shape}")
print("\n=== Class distribution ===")
print(y.value_counts(normalize=True).rename("proportion"))

# 3. Train / Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Train Random Forest (improved configuration)
model = RandomForestClassifier(
    n_estimators=400,          # more trees for stability
    max_depth=None,           # let trees grow, but regularize with min_samples_*
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",  # handle class imbalance
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

#  Save model and metadata
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

#  Predict & Report
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC: {roc_auc:.3f}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual: 0 (Not diabetic)", "Actual: 1 (Diabetic)"],
    columns=["Pred: 0", "Pred: 1"],
)
print("\n=== Confusion Matrix ===")
print(cm_df)

# Feature Importance (for static heatmap)
feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\n=== Top 15 Feature Importances ===")
print(feature_importance.head(15))

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)

# Save test data for Gradio metrics display
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Plot and save static feature importance barplot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(15),
    x="importance",
    y="feature"
)
plt.title("Feature Importance for Diabetes Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\n Model trained and saved successfully!")
print(" Feature importance saved to feature_importance.csv and feature_importance.png")
print(" Test data saved to X_test.csv and y_test.csv")
