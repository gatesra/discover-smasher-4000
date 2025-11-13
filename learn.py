# learn.py — trains a local scorer from data/training.csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

DATA = Path("data/training.csv")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
OUT = MODEL_DIR / "title_scorer.joblib"

if not DATA.exists():
    raise SystemExit("data/training.csv not found. Create it with header: title,franchise,label,timestamp")

df = pd.read_csv(DATA)
df = df.dropna(subset=["title","label"])
df["y"] = (df["label"].str.lower() == "win").astype(int)

# basic engineered features
df["title_len"] = df["title"].str.len()
df["word_count"] = df["title"].str.split().str.len()

feature_cols = ["title", "franchise", "title_len", "word_count"]

pre = ColumnTransformer(
    transformers=[
        ("txt", TfidfVectorizer(ngram_range=(1,2), max_features=6000), "title"),
        ("fr", OneHotEncoder(handle_unknown="ignore"), ["franchise"]),
        ("num", "passthrough", ["title_len","word_count"]),
    ],
    remainder="drop"
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
])

# train/test split only if we have enough rows
if len(df) >= 10 and df["y"].nunique() == 2:
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
    )
    pipe.fit(X_train, y_train)
    try:
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
        print("AUC:", round(auc,3))
    except Exception:
        print("AUC: n/a")
else:
    # not enough data to split; train on all rows
    print("Not enough labeled rows for a test split — training on all data.")
    pipe.fit(df[feature_cols], df["y"])

joblib.dump(pipe, OUT)
print("Saved model →", OUT)