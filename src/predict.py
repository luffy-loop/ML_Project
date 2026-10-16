"""Validated prediction helper shared by the CLI and Streamlit app."""
from pathlib import Path
import joblib
import pandas as pd

def load_artifact(path="models/student_risk_pipeline.joblib"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}. Train it first; see README.md.")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "feature_columns" not in artifact:
        raise ValueError("Invalid model artifact: expected pipeline and feature_columns.")
    return artifact

def predict_frame(frame, artifact):
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    columns = artifact["feature_columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required predictor columns: {', '.join(missing)}")
    features = frame.loc[:, columns]
    probabilities = artifact["pipeline"].predict_proba(features)[:, 1]
    if not pd.Series(probabilities).between(0, 1).all():
        raise ValueError("Model returned scores outside [0, 1].")
    result = frame.copy()
    result["At_Risk_Score"] = probabilities
    result["Score_Band"] = pd.cut(probabilities, bins=[-0.001, .25, .5, .75, 1.001],
        labels=["LOW_SCORE", "ELEVATED_SCORE", "HIGH_SCORE", "VERY_HIGH_SCORE"], include_lowest=True).astype(str)
    return result
