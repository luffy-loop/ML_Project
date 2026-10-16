"""Streamlit demo for a trained student-support research baseline."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.predict import load_artifact, predict_frame

st.set_page_config(page_title="Student Success Analytics", page_icon="🎓", layout="wide")
st.title("Student Success Analytics")
st.caption("Research prototype · Supportive outreach only · Not a decision-making system")
st.warning("Scores and score bands are provisional unless calibration and thresholds have been validated on a future cohort. Never use these results to penalize or restrict students.")
model_path = st.text_input("Model artifact path", "models/student_risk_pipeline.joblib")
upload = st.file_uploader("Upload prepared cutoff-safe feature CSV", type=["csv"])
if not Path(model_path).is_file():
    st.info("Train a model first using the commands in README.md.")
    st.stop()
if upload is None:
    st.info("Upload a CSV containing the predictor columns used during training. Use anonymized records for shared demos.")
    st.stop()
try:
    artifact = load_artifact(model_path)
    source = pd.read_csv(upload)
    output = predict_frame(source, artifact)
    scores = output["At_Risk_Score"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Records scored", f"{len(output):,}")
    c2.metric("Scores ≥ 0.50", int(scores.ge(.5).sum()))
    c3.metric("Median model score", f"{scores.median():.1%}")
    st.subheader("Score distribution")
    st.bar_chart(scores.value_counts(bins=10, sort=False).sort_index())
    st.subheader("Scored records")
    st.dataframe(output.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"),
        file_name="student_support_scores.csv", mime="text/csv")
    st.caption("A model score is not a diagnosis, a cause, or a validated probability of failure. Confirm context with an educator and offer voluntary support.")
except (KeyError, ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"Could not score this file: {exc}")
