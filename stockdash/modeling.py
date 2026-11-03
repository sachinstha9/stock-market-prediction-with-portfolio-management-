
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, MODEL_PATH
from .data import add_features


def model_candidates(random_state=42):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.07,
                l2_regularization=1.0,
                random_state=random_state,
            )),
        ]),
    }

def walk_forward_splits(df, horizon, n_splits=3):
    dates = pd.Index(sorted(pd.to_datetime(df.index).unique()))

    if len(dates) < (n_splits + 2) * max(10, horizon):
        raise ValueError(
            f"Not enough unique dates ({len(dates)}) "
            f"for {n_splits} walk-forward folds"
        )

    blocks = np.array_split(dates, n_splits + 1)

    for i in range(1, len(blocks)):
        test_dates = blocks[i]

        if not len(test_dates):
            continue

        test_start = pd.Timestamp(test_dates[0])
        prior_dates = dates[dates < test_start]

        train_dates = (
            prior_dates[:-horizon]
            if len(prior_dates) > horizon
            else prior_dates[:0]
        )

        index_dates = pd.to_datetime(df.index)
        train_idx = np.flatnonzero(index_dates.isin(train_dates))
        test_idx = np.flatnonzero(index_dates.isin(test_dates))

        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx

def evaluate_models(df, horizon, n_splits=3):
    X, y = df[FEATURES], df["target"]
    records, predictions = [], []

    for name, estimator in model_candidates().items():
        fold_scores = []

        for fold, (tr, te) in enumerate(
            walk_forward_splits(df, horizon, n_splits), 1
        ):
            if y.iloc[tr].nunique() < 2:
                continue

            fitted = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
            pred = fitted.predict(X.iloc[te])

            fold_scores.append({
                "fold": fold,
                "accuracy": accuracy_score(y.iloc[te], pred),
                "f1_macro": f1_score(
                    y.iloc[te], pred, average="macro", zero_division=0
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y.iloc[te], pred
                ),
            })

            for idx, label in zip(te, pred):
                predictions.append({
                    "model": name,
                    "index": df.index[idx],
                    "ticker": df.iloc[idx]["ticker"],
                    "actual": int(y.iloc[idx]),
                    "predicted": int(label),
                    "target_return": float(df.iloc[idx]["target_return"]),
                })

        if fold_scores:
            records.append({
                "model": name,
                "accuracy": np.mean([s["accuracy"] for s in fold_scores]),
                "f1_macro": np.mean([s["f1_macro"] for s in fold_scores]),
                "balanced_accuracy": np.mean(
                    [s["balanced_accuracy"] for s in fold_scores]
                ),
                "folds": len(fold_scores),
                "fold_details": fold_scores,
            })

    if not records:
        raise RuntimeError("No valid walk-forward folds were produced.")

    scores = (
        pd.DataFrame(records)
        .sort_values(
            ["balanced_accuracy", "f1_macro"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return scores, pd.DataFrame(predictions)

def train_and_save(
    df,
    horizon,
    mode="UP/DOWN",
    minimum_balanced_accuracy=0.48,
    save_path=MODEL_PATH,
):
    scores, oos_predictions = evaluate_models(df, horizon)

    winner_name = str(scores.iloc[0]["model"])
    winner_score = float(scores.iloc[0]["balanced_accuracy"])

    if winner_score < minimum_balanced_accuracy:
        return {
            "saved": False,
            "reason": (
                f"Best walk-forward balanced accuracy {winner_score:.3f} "
                f"is below required {minimum_balanced_accuracy:.3f}."
            ),
            "scores": scores,
            "predictions": oos_predictions,
            "model_name": winner_name,
        }

    final_model = clone(model_candidates()[winner_name]).fit(
        df[FEATURES], df["target"]
    )

    metadata = {
        "model": final_model,
        "model_name": winner_name,
        "features": FEATURES,
        "horizon": int(horizon),
        "mode": mode,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(len(df)),
        "stocks": sorted(df["ticker"].unique().tolist()),
        "walk_forward_balanced_accuracy": winner_score,
        "validation_scores": scores.drop(
            columns="fold_details"
        ).to_dict("records"),
        "target_classes": sorted(int(v) for v in df["target"].unique()),
        "oos_predictions": [
            {
                "date": pd.Timestamp(r["index"]).isoformat(),
                "ticker": str(r["ticker"]),
                "predicted": int(r["predicted"]),
                "actual": int(r["actual"]),
                "target_return": float(r["target_return"]),
            }
            for r in oos_predictions[
                oos_predictions["model"] == winner_name
            ].to_dict("records")
        ],
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata, save_path)

    return {
        "saved": True,
        "reason": "Candidate passed the configured validation gate and was saved.",
        "scores": scores,
        "predictions": oos_predictions,
        "model_name": winner_name,
        "metadata": metadata,
    }

def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        return None

    return joblib.load(path)

def predict_latest(metadata, prices):

    frame = add_features(prices).dropna(
        subset=metadata["features"]
    )

    if frame.empty:
        raise ValueError(
            "Not enough history to calculate prediction features"
        )

    row = frame.iloc[[-1]]
    model = metadata["model"]

    predicted = int(model.predict(row[metadata["features"]])[0])

    probs = (
        model.predict_proba(row[metadata["features"]])[0]
        if hasattr(model, "predict_proba")
        else None
    )

    classes = list(model.classes_) if hasattr(model, "classes_") else []

    return {
        "date": frame.index[-1],
        "close": float(row["Close"].iloc[0]),
        "predicted_class": predicted,
        "probabilities": (
            {
                str(int(c)): float(p)
                for c, p in zip(classes, probs)
            }
            if probs is not None
            else {}
        ),
    }