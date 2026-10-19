import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

CSV_FILE = "fsl_dataset.csv"
MODEL_FILE = "fsl_model.joblib"

if not os.path.exists(CSV_FILE):
    print(f"Error: '{CSV_FILE}' not found. Run 'python data.py' first to collect data.")
    exit(1)

df = pd.read_csv(CSV_FILE)

if df.empty or len(df["label"].unique()) < 2:
    print("ValueError: Collect more examples from at least TWO different labels using data.py!")
    exit(1)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_FILE)
print(f"Success: Model saved as '{MODEL_FILE}'.")
