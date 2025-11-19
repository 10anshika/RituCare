import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load PCOS dataset
df = pd.read_csv("dataset/menstrual_health_and_pcodrisk_detection.csv")

# Auto-detect target column
target_col = "PCOS (Y/N)" if "PCOS (Y/N)" in df.columns else df.columns[-1]

# Use only features available in the app
app_features = ['LengthofCycle', 'BMI', 'Age', 'Weight', 'Height', 'UnusualBleeding', 'ReproductiveCategory', 'Maristatus', 'Numberpreg']
available_features = [col for col in app_features if col in df.columns]

X = df[available_features]
y = df[target_col]

# Handle missing values and convert to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        # Try to convert to numeric, if fails, factorize
        try:
            X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')
        except:
            X.loc[:, col] = pd.factorize(X[col])[0]
    X.loc[:, col] = X[col].fillna(X[col].mean() if X[col].dtype in ['float64', 'int64'] else X[col].mode()[0] if not X[col].mode().empty else 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ PCOS model trained successfully. Accuracy: {acc:.2f}")

# Save
os.makedirs("models", exist_ok=True)
with open("models/pcos_model.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps["model"], f)
with open("models/preprocessor.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps["scaler"], f)

print("✅ Models saved successfully at /models/pcos_model.pkl and /models/preprocessor.pkl")
