"""Train a cutoff-safe baseline and save its full preprocessing pipeline."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(features_csv, model_path, metrics_path, seed=42):
    df = pd.read_csv(features_csv)
    required = {"At_Risk", "id_student"}
    if not required.issubset(df.columns):
        raise ValueError("Feature CSV must contain At_Risk and id_student.")
    df = df.dropna(subset=["At_Risk", "id_student"]).copy()
    y = df["At_Risk"].astype(int)
    groups = df["id_student"]
    X = df.drop(columns=["At_Risk", "id_student"])
    if X.empty:
        raise ValueError("No predictor columns found.")
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        )),
    ])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = {
        "model": "RandomForestClassifier",
        "split": "GroupShuffleSplit by id_student",
        "rows": len(df),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "accuracy": float(accuracy_score(y.iloc[test_idx], pred)),
        "precision_at_risk": float(precision_score(y.iloc[test_idx], pred, zero_division=0)),
        "recall_at_risk": float(recall_score(y.iloc[test_idx], pred, zero_division=0)),
        "f1_at_risk": float(f1_score(y.iloc[test_idx], pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y.iloc[test_idx], prob)),
        "pr_auc": float(average_precision_score(y.iloc[test_idx], prob)),
        "confusion_matrix": confusion_matrix(y.iloc[test_idx], pred).tolist(),
        "classification_report": classification_report(
            y.iloc[test_idx], pred, zero_division=0, output_dict=True
        ),
        "feature_columns": X.columns.tolist(),
        "note": "Random group holdout; validate with a future-presentation holdout before deployment.",
    }
    # Refit final artifact on all available prepared rows after holdout evaluation.
    model.fit(X, y)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "feature_columns": X.columns.tolist()}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="CSV exported by cutoff-safe feature generation")
    p.add_argument("--model", default="models/student_risk_pipeline.joblib")
    p.add_argument("--metrics", default="reports/metrics.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(a.features, a.model, a.metrics, a.seed)
