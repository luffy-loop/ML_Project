"""Streamlit demo for the trained student-risk pipeline.

Run: streamlit run app.py
Train first with src/train.py. Use a cutoff-safe, anonymized feature CSV.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Success Analytics", page_icon="🎓", layout="wide")
st.title("🎓 AI-Powered Student Performance & Success Prediction")
st.write(
    "Upload a prepared feature CSV to score students with a trained model. "
    "This prototype supports early academic outreach; it must not be used to punish students."
)
model_path = st.text_input("Model artifact path", "models/student_risk_pipeline.joblib")
upload = st.file_uploader("Upload cutoff-safe feature CSV", type=["csv"])

if not Path(model_path).is_file():
    st.warning("Train a model first. See README.md for setup and training commands.")
    st.stop()
if upload is None:
    st.info("Upload a CSV containing the same predictor columns used during training.")
    st.stop()

try:
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    df = pd.read_csv(upload)
    if df.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        st.error(f"Missing required model columns: {', '.join(missing)}")
        st.stop()
    # Extra columns such as labels or student IDs are ignored intentionally.
    features = df.loc[:, feature_columns]
    probabilities = pipeline.predict_proba(features)[:, 1]
    output = df.copy()
    output["At_Risk_Probability"] = probabilities
    output["Risk_Band"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["LOW", "GUARDED", "HIGH", "VERY HIGH"],
        include_lowest=True,
    ).astype(str)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records scored", f"{len(output):,}")
    c2.metric("High / very-high risk", int(output["At_Risk_Probability"].ge(0.5).sum()))
    c3.metric("Mean model score", f"{output['At_Risk_Probability'].mean():.1%}")
    st.caption(
        "Risk bands are provisional and model scores are not guaranteed to be calibrated. "
        "Use only for supportive outreach, and validate thresholds before operational use."
    )
    st.dataframe(output.head(100), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name="student_risk_predictions.csv",
        mime="text/csv",
    )
except (KeyError, ValueError, OSError) as exc:
    st.error(f"Could not score this file: {exc}")
except Exception as exc:
    st.error(f"Unexpected scoring error: {exc}")
