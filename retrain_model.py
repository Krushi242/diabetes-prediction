import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load original dataset
df_original = pd.read_csv("diabetes_prediction_dataset.csv")

# Encode
df_original = pd.get_dummies(df_original, columns=["gender", "smoking_history"], drop_first=True)

# Load user data if exists
try:
    df_user = pd.read_csv("user_data.csv")
    df_combined = pd.concat([df_original, df_user], ignore_index=True)
except:
    df_combined = df_original

X = df_combined.drop("diabetes", axis=1, errors='ignore')
y = df_combined["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=300
)

model.fit(X_train, y_train)

joblib.dump(model, "diabetes_model.pkl")

print("Model retrained successfully!")
