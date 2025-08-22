import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
print("📂 Loading dataset...")
df = pd.read_csv("emails.csv")

print("✅ Loaded", len(df), "emails")

# Combine subject, body, and sender into one text field
df["text"] = (
    df["subject"].fillna("") + " " +
    df["body"].fillna("") +
    " sender:" + df["sender"].fillna("")
)

X = df["text"]
y = df["label"]

# -------------------------------
# 2. Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🧪 Training on {len(X_train)} samples, testing on {len(X_test)} samples")

# -------------------------------
# 3. Build Pipeline (TF-IDF + SVM)
# -------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
    ("clf", LinearSVC())
])

# -------------------------------
# 4. Train
# -------------------------------
print("🔧 Training model...")
pipeline.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate
# -------------------------------
y_pred = pipeline.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 6. Save model
# -------------------------------
joblib.dump(pipeline, "email_importance_model.pkl")
print("💾 Model saved as email_importance_model.pkl")
