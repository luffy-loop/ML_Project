"""Train a cutoff-aware baseline and save its preprocessing pipeline."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    )
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def threshold_analysis(y_true, probabilities, thresholds=(0.25, 0.50, 0.75)):
    """Compare outreach workload and positive-class metrics at candidate thresholds."""
    y_values = pd.Series(y_true).astype(int).to_numpy()
    probability_values = pd.Series(probabilities).to_numpy()
    rows = []
    for threshold in thresholds:
        predicted = (probability_values >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "flagged_count": int(predicted.sum()),
            "flagged_rate": float(predicted.mean()),
            "precision_at_risk": float(precision_score(y_values, predicted, zero_division=0)),
            "recall_at_risk": float(recall_score(y_values, predicted, zero_division=0)),
            "f1_at_risk": float(f1_score(y_values, predicted, zero_division=0)),
        })
    return rows


def subgroup_audit(X_test, y_test, probabilities, min_group_size=5):
    """Report descriptive error rates for available demographic columns.

    This is a screening aid, not a fairness certification. Small groups are
    omitted to reduce unstable estimates; group labels are not altered here.
    """
    audit_columns = ("gender", "age_band", "region", "highest_education")
    y_values = pd.Series(y_test).astype(int).to_numpy()
    probability_values = pd.Series(probabilities).to_numpy()
    results = {}
    for column in audit_columns:
        if column not in X_test.columns:
            continue
        frame = pd.DataFrame({
            "group": X_test[column].astype("string").fillna("Missing").to_numpy(),
            "actual": y_values,
            "probability": probability_values,
        })
        group_rows = []
        for group_name, group in frame.groupby("group", dropna=False, sort=True):
            if len(group) < min_group_size:
                continue
            actual = group["actual"].to_numpy()
            predicted = (group["probability"].to_numpy() >= 0.5).astype(int)
            positives = int((actual == 1).sum())
            negatives = int((actual == 0).sum())
            true_positives = int(((actual == 1) & (predicted == 1)).sum())
            false_positives = int(((actual == 0) & (predicted == 1)).sum())
            group_rows.append({
                "group": str(group_name),
                "n": int(len(group)),
                "actual_at_risk_rate": float(actual.mean()),
                "flagged_rate": float(predicted.mean()),
                "recall_at_0_5": float(true_positives / positives) if positives else None,
                "false_positive_rate_at_0_5": float(false_positives / negatives) if negatives else None,
            })
        results[column] = group_rows
    return results


def train(features_csv, model_path, metrics_path, seed=42):
    """Evaluate on a student-grouped holdout, then save a pipeline fit to all rows."""
    df = pd.read_csv(features_csv)
    required = {"At_Risk", "id_student"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature CSV must contain {sorted(required)}.")
    if "final_result" in df.columns:
        raise ValueError(
            "Leakage-prone outcome column 'final_result' must not be used as a predictor; "
            "regenerate cutoff-safe features without final-result labels."
        )
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

    # Post-hoc audit uses held-out rows, before the deployable all-row refit.
    # Permutation importance measures predictive association, not causation.
    from sklearn.inspection import permutation_importance
    importance = permutation_importance(
        model, X.iloc[test_idx], y_test, scoring="average_precision",
        n_repeats=3, random_state=seed, n_jobs=-1
    )
    importance_rows = sorted(
        [
            {"feature": str(feature), "importance_mean": float(mean), "importance_std": float(std)}
            for feature, mean, std in zip(X.columns, importance.importances_mean, importance.importances_std)
        ],
        key=lambda row: row["importance_mean"],
        reverse=True,
    )[:10]
    roc_points, pr_points, calibration_points = [], [], []
    if y_test.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_points = [{"false_positive_rate": float(x), "true_positive_rate": float(y)} for x, y in zip(fpr, tpr)]
        precision_values, recall_values, _ = precision_recall_curve(y_test, prob)
        pr_points = [{"recall": float(r), "precision": float(p)} for p, r in zip(precision_values, recall_values)]
        fraction_positive, mean_predicted = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
        calibration_points = [{"mean_predicted": float(x), "fraction_positive": float(y)} for y, x in zip(fraction_positive, mean_predicted)]

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
        "brier_score": float(brier_score_loss(y_test, prob)),
        "roc_curve": roc_points,
        "precision_recall_curve": pr_points,
        "calibration_curve": calibration_points,
        "threshold_analysis": threshold_analysis(y_test, prob),
        "subgroup_audit": subgroup_audit(X.iloc[test_idx], y_test, prob),
        "permutation_importance_top10": importance_rows,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["actual_not_at_risk", "actual_at_risk"],
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["not_at_risk", "at_risk"],
            zero_division=0, output_dict=True
        ),
        "feature_columns": X.columns.tolist(),
        "audit_notes": [
            "Random student-grouped holdout; this does not establish future-cohort performance.",
            "Brier score summarizes probability error; lower is better, but calibration curves should also be inspected.",
            "Threshold comparisons show trade-offs and do not prescribe an operational threshold.",
            "Subgroup metrics are descriptive and can be unstable; they are not a fairness certification.",
            "Permutation importance measures held-out predictive association, not causation.",
            "Evaluate on a genuinely later presentation before any deployment decision."
        ],
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
