"""Train a cutoff-aware baseline and save its preprocessing pipeline."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(features_csv, model_path, metrics_path, seed=42):
    """Evaluate on a student-grouped holdout, then save a pipeline fit to all rows."""
    df = pd.read_csv(features_csv)
    required = {"At_Risk", "id_student"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature CSV must contain {sorted(required)}.")
    df = df.dropna(subset=["At_Risk", "id_student"]).copy()
    df["At_Risk"] = pd.to_numeric(df["At_Risk"], errors="coerce")
    df = df.dropna(subset=["At_Risk"])
    if not set(df["At_Risk"].unique()).issubset({0, 1}):
        raise ValueError("At_Risk must contain only binary values 0 and 1.")
    df["At_Risk"] = df["At_Risk"].astype(int)
    if df.empty or df["At_Risk"].nunique() < 2:
        raise ValueError("Training requires rows from both At_Risk classes.")
    if df["id_student"].nunique() < 2:
        raise ValueError("Grouped evaluation requires at least two distinct students.")

    y = df["At_Risk"]
    groups = df["id_student"]
    X = df.drop(columns=["At_Risk", "id_student"])
    if X.empty:
        raise ValueError("No predictor columns found.")

    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    transformers = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat,
        ))
    prep = ColumnTransformer(transformers, remainder="drop")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    y_test = y.iloc[test_idx]
    y_train = y.iloc[train_idx]
    if y_train.nunique() < 2:
        raise ValueError(
            "Grouped training split contains only one class. Add more students "
            "or use a different evaluation split/seed."
        )

    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        )),
    ])
    model.fit(X.iloc[train_idx], y_train)
    pred = model.predict(X.iloc[test_idx])
    prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = {
        "model": "RandomForestClassifier",
        "split": "GroupShuffleSplit by id_student",
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_students": int(groups.iloc[train_idx].nunique()),
        "test_students": int(groups.iloc[test_idx].nunique()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_at_risk": float(precision_score(y_test, pred, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, pred, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() == 2 else None,
        "pr_auc": float(average_precision_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["actual_not_at_risk", "actual_at_risk"],
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["not_at_risk", "at_risk"],
            zero_division=0, output_dict=True
        ),
        "feature_columns": X.columns.tolist(),
        "note": (
            "Random student-grouped holdout; this does not establish future-cohort "
            "performance. Evaluate on a later presentation before deployment."
        ),
    }

    # Refit after evaluation for a deployable artifact. Reported metrics remain from holdout.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV exported by cutoff-safe feature generation")
    parser.add_argument("--model", default="models/student_risk_pipeline.joblib")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.features, args.model, args.metrics, args.seed)
