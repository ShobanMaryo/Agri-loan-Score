import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("agriloanscore_dataset_tn.csv")

# Drop ID column
df = df.drop(columns=["Farmer_ID"])
# Features & Target
X = df.drop("Eligible", axis=1)
y = df["Eligible"]

# Categorical & numeric features
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# ML model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Full pipeline
clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "agriloanscore_model.pkl")

print("✅ Model trained and saved as agriloanscore_model.pkl")
