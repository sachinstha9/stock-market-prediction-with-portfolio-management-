from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def model_candidates(random_state=42):
    return {
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=random_state))]),
        "Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.07, l2_regularization=1.0, random_state=random_state))]),
    }

def walk_forward_splits(df, horizon, n_splits=3):
    dates = pd.Index(sorted(pd.to_datetime(df.index).unique()))
    if len(dates) < (n_splits + 2) * max(10, horizon):
        raise ValueError(f"Not enough unique dates ({len(dates)}) for {n_splits} walk-forward folds")
    blocks = np.array_split(dates, n_splits + 1)
    for i in range(1, len(blocks)):
        test_dates = blocks[i]
        if not len(test_dates):
            continue
        test_start = pd.Timestamp(test_dates[0])
        # Purge labels whose future outcome overlaps the test window.
        prior_dates = dates[dates < test_start]
        train_dates = prior_dates[:-horizon] if len(prior_dates) > horizon else prior_dates[:0]
        train_idx = np.flatnonzero(pd.to_datetime(df.index).isin(train_dates))
        test_idx = np.flatnonzero(pd.to_datetime(df.index).isin(test_dates))
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx
