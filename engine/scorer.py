# engine/scorer.py
# ---------------------------------------
# Global-only headline scorer for Discover titles.
# - Trains from training/training_balanced.csv (if present),
#   else training/training.csv, else ./training.csv.
# - Robust text + lightweight numeric features.
# - LogisticRegression (saga) inside a Pipeline, calibrated via sigmoid.
# - Exposes: train_local_model(), score_candidates(), score_titles()

from __future__ import annotations

import os
import re
import joblib
import numpy as np
import pandas as pd
import joblib
import traceback
import streamlit as st
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import _column_transformer as _ct  # compat shim target

# ----------------------------
# Paths & constants
# ----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "title_scorer_global.joblib")

# Try in this order:
DATA_CANDIDATES = [
    "training/training_balanced.csv",  # preferred when present
    "training/training.csv",
    "training.csv",
]

TEXT_COL = "title"
NUM_COLS = [
    "title_len_words",
    "title_len_chars",
    "has_number",
    "has_colon",
    "has_question",
    "has_quote",
    "has_identity",
    "has_controversy",
    "payoff_front",
]

REQUIRED_TRAIN_COLS = {TEXT_COL, "win"}  # keep it global/generic


# ----------------------------
# Utilities
# ----------------------------
def _pick_data_path() -> str:
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No training data found. Expected one of: "
        + ", ".join(DATA_CANDIDATES)
    )


def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("").str.strip()
    # normalize whitespace
    s = s.replace(r"\s+", " ", regex=True)
    return s


def _first_payoff_position(title: str) -> int:
    """
    Heuristic: position (0-based) of first payoff/impact token in the title.
    Returns large index if none found.
    """
    payoff_terms = re.compile(
        r"\b("
        r"fix|nerf|buff|update|patch|hotfix|delay|delayed|"
        r"release|launch|reveals?|leaks?|confirms?|confirmed|"
        r"wins?|loss|breaks?|ban|unban|free|exclusive|"
        r"explained|official|canceled|cancelled"
        r")\b",
        flags=re.IGNORECASE,
    )
    words = re.findall(r"\b\w+'\w+|\w+\b", title.lower())
    for idx, w in enumerate(words):
        if payoff_terms.search(w):
            return idx
    return len(words) + 999  # none found -> very large


def _add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple, robust numeric features expected by the pipeline.
    """
    t = _clean_text(df[TEXT_COL])

    df = df.copy()
    df[TEXT_COL] = t

    df["title_len_words"] = t.str.split().map(lambda x: len(x) if isinstance(x, list) else 0).astype(int)
    df["title_len_chars"] = t.str.len().fillna(0).astype(int)

    df["has_number"] = t.str.contains(r"\d", regex=True, na=False).astype(int)
    df["has_colon"] = t.str.contains(r":", regex=False, na=False).astype(int)
    df["has_question"] = t.str.contains(r"\?", regex=False, na=False).astype(int)
    df["has_quote"] = t.str.contains(r"[\"'“”‘’]", regex=True, na=False).astype(int)

    # Non-capturing groups to avoid pandas warnings
    df["has_identity"] = t.str.contains(
        r"\b(?:I|You|We|They|Fans|Players|Community|Devs?|Developers)\b",
        case=False, regex=True, na=False
    ).astype(int)

    df["has_controversy"] = t.str.contains(
        r"\b(?:backlash|angry|outrage|controvers\w*|drama|leaks?)\b",
        case=False, regex=True, na=False
    ).astype(int)

    # Payoff front: whether the first payoff token occurs in first 5 words
    df["payoff_front"] = t.map(lambda s: 1 if _first_payoff_position(s) <= 4 else 0).astype(int)

    return df


def _build_estimator() -> CalibratedClassifierCV:
    """
    Pipeline:
      - ColumnTransformer:
          * Tfidf on `title`
          * StandardScaler on numeric features (with_mean=False for sparse safety)
      - LogisticRegression (saga) with class_weight balanced
      - Calibrated with sigmoid (stable)
    """
    txt_vect = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=200_000,
        strip_accents="unicode",
        lowercase=True,
    )

    pre = ColumnTransformer(
        transformers=[
            ("txt", txt_vect, TEXT_COL),
            ("num", StandardScaler(with_mean=False), NUM_COLS),
        ],
        sparse_threshold=0.3,  # keep as sparse when dominated by tfidf
        remainder="drop",
        verbose_feature_names_out=False,
    )

    base = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        penalty="l2",
        C=0.5,
        max_iter=5000,
        tol=1e-3,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", base),
        ]
    )

    # Calibrate probabilities; sigmoid is robust on smaller folds
    calibrated = CalibratedClassifierCV(
        estimator=pipe,
        method="sigmoid",
        cv=3,
    )
    return calibrated


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------------
# Sklearn compatibility shim
# ----------------------------
def _ensure_sklearn_compat():
    """
    Compatibility shim for older pickled sklearn ColumnTransformer objects that
    expect a private helper class `_RemainderColsList` which may not exist in
    the current sklearn build.

    We define a minimal stand-in so joblib/pickle can resolve the reference.
    """
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Minimal stand-in used only for unpickling legacy models."""
            pass
        _ct._RemainderColsList = _RemainderColsList


# ----------------------------
# Public API
# ----------------------------
def train_local_model() -> Dict[str, Any]:
    """
    Train the global model from CSV and save to models/title_scorer_global.joblib.
    Returns a small metadata dict.
    """
    data_path = _pick_data_path()
    print(f"✅ Using dataset: {data_path}")

    df = pd.read_csv(data_path)
    missing = REQUIRED_TRAIN_COLS - set(df.columns)
    if missing:
        raise ValueError(f"training CSV missing required columns: {missing}")

    # Clean + features
    df = df.dropna(subset=[TEXT_COL, "win"]).copy()
    df["win"] = df["win"].astype(int).clip(0, 1)
    df = _add_numeric_features(df)

    X = df[[TEXT_COL] + NUM_COLS]
    y = df["win"].astype(int)

    est = _build_estimator()

    # CV AUC on the calibrated estimator (safe, but a bit slower). If desired,
    # you can swap to scoring the uncalibrated pipe for speed.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(est, X, y, scoring="roc_auc", cv=cv, n_jobs=1)
    cv_auc = float(np.mean(scores)) if len(scores) else None

    # Fit on all data
    est.fit(X, y)

    # Save
    _ensure_model_dir()
    joblib.dump(est, MODEL_PATH)

    return {
        "global_model_path": MODEL_PATH,
        "rows": int(len(df)),
        "cv_auc": cv_auc,
        "calibration": "sigmoid",
    }


def _load_model():
    # Ensure sklearn has the compatibility shim before unpickling
    _ensure_sklearn_compat()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        # Show the REAL error in the Streamlit UI
        st.error(f"Model load failed: {repr(e)}")
        st.code(traceback.format_exc())
        raise


def score_candidates(candidates: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Input: [{ "title": "..." }, ...]
    Output: DataFrame with columns: title, pwin (0..1)
    """
    if not candidates:
        return pd.DataFrame(columns=["title", "pwin"])

    df = pd.DataFrame(candidates)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing '{TEXT_COL}' in candidates.")

    df = _add_numeric_features(df)
    X = df[[TEXT_COL] + NUM_COLS]

    clf = _load_model()
    p = clf.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "title": df[TEXT_COL].values,
        "pwin": p,
    })
    return out


def score_titles(titles: List[str]) -> pd.DataFrame:
    """
    Convenience wrapper for a plain list of strings.
    Returns DataFrame with: title, pwin
    """
    if not titles:
        return pd.DataFrame(columns=["title", "pwin"])
    cands = [{"title": t} for t in titles]
    return score_candidates(cands)


# ------------- Optional local smoke test -------------
if __name__ == "__main__":
    # Minimal demo to verify import/execution if you run:
    #   python engine/scorer.py
    demo = [
        "Blizzard Confirms No WoW For Consoles — Fans React",
        "Players Celebrate Huge Buff After Patch Notes Drop",
        "Nintendo Switch 2 Delay Explained: What It Means For Launch",
    ]
    try:
        # Only runs if a model already exists.
        df = score_titles(demo)
        print(df)
    except FileNotFoundError as e:
        print(str(e))