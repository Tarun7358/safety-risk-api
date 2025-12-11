import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load CSV
df = pd.read_csv("training_data.csv")
print("\nLoaded Data:\n", df.head())

# Features (X)
X = df[["age", "timeOfDay", "crowdDensity", "areaSafetyScore", "weather"]]

# Label (y)
y = df["risk_level"]   # <-- Correct column name

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "model.pkl")

print("\n🎉 Model trained and saved as model.pkl successfully!")
