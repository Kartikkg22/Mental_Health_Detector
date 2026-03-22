import os
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess import clean_text


# 📌 Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 📁 Paths
data_path = os.path.join(BASE_DIR, "data", "dataset.csv")
model_dir = os.path.join(BASE_DIR, "model")

# Create model folder if not exists
os.makedirs(model_dir, exist_ok=True)


# 📊 Load dataset
df = pd.read_csv(data_path)

# ✅ Drop unwanted column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ✅ Rename columns based on your dataset
df = df.rename(columns={
    "statement": "text",
    "status": "label"
})

# ✅ Clean labels
df['label'] = df['label'].astype(str).str.strip().str.lower()

# ✅ Clean text
df['text'] = df['text'].astype(str).apply(clean_text)

# 🔥 OPTIONAL: limit dataset size (remove if you want full dataset)
df = df.sample(n=min(5000, len(df)), random_state=42)

print("Dataset shape:", df.shape)
print(df['label'].value_counts())


X = df['text']
y = df['label']


# 🔠 Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)


# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)


# 🤖 Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# 📈 Evaluation
y_pred = model.predict(X_test)
print("\n📊 Model Performance:\n")
print(classification_report(y_test, y_pred))


# 💾 Save model
model_path = os.path.join(model_dir, "model.pkl")
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved!")