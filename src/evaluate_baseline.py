"""Leakage-aware baseline evaluation scaffold.

This module deliberately does NOT train on assessment aggregates as-is.
The project must decide and enforce the prediction cutoff before calling it.
It expects a prepared feature CSV with one row per student-course-presentation,
a binary target column named At_Risk, and a grouping column id_student.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


TARGET = "At_Risk"
GROUP = "id_student"
DROP_FROM_FEATURES = {TARGET, GROUP}


def load_prepared_data(path: Path) -> pd.DataFrame:
    """Load a prepared feature table and validate its required columns."""
    df = pd.read_csv(path)
    required = {TARGET, GROUP}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=[TARGET, GROUP]).copy()
    values = set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"{TARGET} must be binary 0/1; observed values: {values}")
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


def evaluate(path: Path, test_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    df = load_prepared_data(path)
    X = df.drop(columns=list(DROP_FROM_FEATURES))
    y = df[TARGET]
    groups = df[GROUP]

    # Group split prevents the same student's records crossing train/test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    rows = []
    for name, estimator in candidates.items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision_at_risk": precision_score(y_test, pred, zero_division=0),
                "Recall_at_risk": recall_score(y_test, pred, zero_division=0),
                "F1_at_risk": f1_score(y_test, pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, prob),
                "PR_AUC": average_precision_score(y_test, prob),
            }
        )
        print(f"\n=== {name} ===")
        print("Confusion matrix [rows=true, columns=predicted]:")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, zero_division=0))

    results = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    print("\nGrouped-split results:")
    print(results.to_string(index=False))
    print(
        "\nIMPORTANT: A grouped split does not prevent target leakage. "
        "Only use this script after constructing predictors strictly from data "
        "available at the intended prediction cutoff."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="CSV with cutoff-safe engineered features, At_Risk, and id_student",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.features.exists():
        raise FileNotFoundError(args.features)
    evaluate(args.features, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
