"""Evaluate a later presentation while preventing student overlap across splits.

Requires prepared cutoff-safe features retaining code_presentation. Presentation
ordering and label timing must be verified against the intended prediction point.
This script reports evidence; it does not certify prospective validity.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate(features_path, output_path="reports/temporal_metrics.json"):
    df = pd.read_csv(features_path)
    required = {"At_Risk", "code_presentation", "id_student"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Temporal evaluation requires columns: {sorted(missing)}")

    df["At_Risk"] = pd.to_numeric(df["At_Risk"], errors="coerce")
    df = df.dropna(subset=["At_Risk", "code_presentation", "id_student"]).copy()
    if df.empty or not set(df["At_Risk"].unique()).issubset({0, 1}) or df["At_Risk"].nunique() != 2:
        raise ValueError("At_Risk must contain both binary classes 0 and 1.")
    if df["id_student"].duplicated().all() and df["id_student"].nunique() < 2:
        raise ValueError("Temporal evaluation requires multiple distinct students.")

    presentations = sorted(df["code_presentation"].astype(str).unique())
    if len(presentations) < 2:
        raise ValueError("At least two presentations are required.")
    test_presentation = presentations[-1]
    test = df[df["code_presentation"].astype(str) == test_presentation].copy()
    test_students = set(test["id_student"])
    earlier = df[df["code_presentation"].astype(str) < test_presentation]
    # Purge earlier rows for students in the test presentation. Otherwise the
    # same learner can occur in both sets and make generalization look easier.
    train = earlier[~earlier["id_student"].isin(test_students)].copy()
    purged_rows = int(len(earlier) - len(train))
    if train.empty or test.empty:
        raise ValueError("Temporal split is empty after excluding overlapping students.")

    drop = ["At_Risk", "id_student", "code_presentation"]
    X_train, y_train = train.drop(columns=drop), train["At_Risk"].astype(int)
    X_test, y_test = test.drop(columns=drop), test["At_Risk"].astype(int)
    if X_train.empty or list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test predictor schemas do not match.")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Train and latest-presentation test split must each contain both classes.")

    categorical = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in X_train.columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))

    model = Pipeline([
        ("prep", ColumnTransformer(transformers, remainder="drop")),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    result = {
        "evaluation": "train on earlier presentations; test on latest presentation; purge overlapping student IDs",
        "train_presentations": sorted(train["code_presentation"].astype(str).unique().tolist()),
        "test_presentation": test_presentation,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "purged_overlapping_student_rows": purged_rows,
        "train_students": int(train["id_student"].nunique()),
        "test_students": int(test["id_student"].nunique()),
        "student_overlap": int(len(set(train["id_student"]) & set(test["id_student"]))),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_at_risk": float(precision_score(y_test, predicted, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, predicted, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "warning": (
            "Verify presentation chronology, feature availability, and prospective label timing. "
            "Purging overlapping students may reduce sample size; this report is not proof of deployment readiness."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="reports/temporal_metrics.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.features, args.output), indent=2))
