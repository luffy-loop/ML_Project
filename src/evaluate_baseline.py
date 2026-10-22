"""Leakage-aware grouped model comparison for prepared cutoff-safe features."""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET, GROUP = "At_Risk", "id_student"

def build_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    steps = []
    if num: steps.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat: steps.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(steps, remainder="drop")

def evaluate(path, test_size=0.2, seed=42, output="reports/model_comparison.json"):
    df = pd.read_csv(path)
    if missing := ({TARGET, GROUP} - set(df.columns)):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "final_result" in df.columns:
        raise ValueError(
            "Leakage-prone outcome column 'final_result' must not be used as a predictor; "
            "regenerate cutoff-safe features without final-result labels."
        )
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, GROUP]).copy()
    if not set(df[TARGET].unique()).issubset({0, 1}) or df[TARGET].nunique() != 2:
        raise ValueError("At_Risk must contain both binary classes 0 and 1.")
    X, y, groups = df.drop(columns=[TARGET, GROUP]), df[TARGET].astype(int), df[GROUP]
    if X.empty or groups.nunique() < 2: raise ValueError("Need predictors and at least two distinct students.")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Grouped split must contain both classes in train and test; try more data or another --seed.")
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    rows = []
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        model.fit(X_train, y_train)
        pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "accuracy": float(accuracy_score(y_test, pred)),
            "precision_at_risk": float(precision_score(y_test, pred, zero_division=0)),
            "recall_at_risk": float(recall_score(y_test, pred, zero_division=0)),
            "f1_at_risk": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)), "pr_auc": float(average_precision_score(y_test, prob)),
            "brier_score": float(brier_score_loss(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
            "classification_report": classification_report(y_test, pred, labels=[0,1], target_names=["not_at_risk","at_risk"], output_dict=True, zero_division=0)})
    payload = {"evaluation": "student-grouped holdout; not a future-cohort test", "seed": seed,
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_students": int(groups.iloc[train_idx].nunique()), "test_students": int(groups.iloc[test_idx].nunique()),
        "results": rows, "warning": "Only valid if predictors use information available at the stated cutoff. Validate on a genuinely later presentation before deployment."}
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True); parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", default="reports/model_comparison.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.features, args.test_size, args.seed, args.output), indent=2))
