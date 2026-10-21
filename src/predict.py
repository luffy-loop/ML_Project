"""Validated prediction helpers shared by the API, CLI, and Streamlit app."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_artifact(path="models/student_risk_pipeline.joblib"):
    """Load a model artifact and validate its minimum interface."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Train it first; see README.md."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"pipeline", "feature_columns"} <= set(artifact):
        raise ValueError("Invalid model artifact: expected pipeline and feature_columns.")
    columns = artifact["feature_columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column.strip() for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("Invalid model artifact: feature_columns must be a non-empty list of unique names.")
    if not callable(getattr(artifact["pipeline"], "predict_proba", None)):
        raise ValueError("Invalid model artifact: pipeline must implement predict_proba.")
    classes = getattr(artifact["pipeline"], "classes_", None)
    if classes is not None and 1 not in list(classes):
        raise ValueError("Invalid model artifact: classifier has no positive class labelled 1.")
    return artifact


def _positive_class_index(pipeline, probabilities):
    """Find the probability column corresponding to At_Risk=1."""
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        # Some compatible estimators/wrappers don't expose classes_. Their
        # two-column predict_proba contract follows the conventional [0, 1].
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return 1
        raise ValueError("Model must expose classes_ or return two probability columns ordered [0, 1].")
    classes = list(classes)
    if len(classes) != probabilities.shape[1] or 1 not in classes:
        raise ValueError("Model must return a probability column for the positive class labelled 1.")
    return classes.index(1)


def predict_frame(frame, artifact):
    """Score a prepared feature frame after validating its schema and values.

    Missing values are allowed because the training pipeline owns imputation.
    Infinite numeric values, duplicate columns, and malformed artifacts are not.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate input columns are not allowed: {', '.join(duplicates)}")

    if not isinstance(artifact, dict) or not {"pipeline", "feature_columns"} <= set(artifact):
        raise ValueError("Invalid model artifact: expected pipeline and feature_columns.")
    columns = artifact["feature_columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column.strip() for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("Invalid model artifact: feature_columns must be a non-empty list of unique names.")

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required predictor columns: {', '.join(missing)}")

    features = frame.loc[:, columns]
    numeric = features.select_dtypes(include=[np.number])
    if numeric.size and np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Numeric predictor columns must not contain positive or negative infinity.")

    pipeline = artifact["pipeline"]
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("Invalid model artifact: pipeline must implement predict_proba.")
    try:
        probabilities = np.asarray(pipeline.predict_proba(features))
    except (IndexError, TypeError) as exc:
        raise ValueError("Model could not return prediction probabilities.") from exc
    if probabilities.ndim != 2 or probabilities.shape[0] != len(frame):
        raise ValueError("Model returned an unexpected prediction probability shape.")
    positive_index = _positive_class_index(pipeline, probabilities)
    scores = probabilities[:, positive_index]
    if scores.shape != (len(frame),):
        raise ValueError("Model returned an unexpected number of prediction scores.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model returned scores that are not finite values in [0, 1].")

    result = frame.copy()
    result["At_Risk_Score"] = scores
    result["Score_Band"] = pd.cut(
        scores,
        bins=[-0.001, .25, .5, .75, 1.001],
        labels=["LOW_SCORE", "ELEVATED_SCORE", "HIGH_SCORE", "VERY_HIGH_SCORE"],
        include_lowest=True,
    ).astype(str)
    return result
