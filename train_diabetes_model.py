# =====================================
# 1. IMPORT LIBRARIES
# =====================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =====================================
# 2. LOAD DATASET
# =====================================

df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:\n", df["diabetes"].value_counts())


# =====================================
# 3. NUMERIC SUMMARY
# =====================================

print("\n=== Numeric Summary ===\n")
print(df.describe())


# =====================================
# 4. OUTLIER DETECTION (Only Continuous Columns)
# =====================================

print("\n=== Outlier Detection (Continuous Features Only) ===\n")

continuous_cols = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# Optional: Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=df[continuous_cols])
plt.xticks(rotation=45)
plt.title("Outlier Detection (Continuous Features)")
plt.show()


# =====================================
# 5. ENCODE CATEGORICAL VARIABLES
# =====================================

df = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)


# =====================================
# 6. SPLIT FEATURES & TARGET
# =====================================

X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# =====================================
# 7. TRAIN-TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # important for imbalanced dataset
)


# =====================================
# 8. TRAIN MODEL
# =====================================

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=200
)

model.fit(X_train, y_train)


# =====================================
# 9. MODEL EVALUATION
# =====================================

y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =====================================
# 10. SAVE MODEL
# =====================================

joblib.dump(model, "diabetes_model.pkl")

print("\n✅ Model saved successfully as diabetes_model.pkl")
