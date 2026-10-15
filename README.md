# AI-Powered Student Performance & Success Prediction

A machine-learning project to study student outcomes and identify learners who may benefit from timely academic support.

> **Status:** This repository contains exploratory and publication-oriented notebooks, plus a reusable cutoff-aware baseline pipeline and Streamlit demo. The production-style pipeline has not yet been run end-to-end against the full OULAD data in a clean environment. Treat saved notebook metrics as preliminary until the corrected temporal evaluation has completed successfully.

## Repository contents

- `MLProject.ipynb` — original exploratory notebook.
- `OULAD_Final_2026_Q1_Publication_Notebook.ipynb` — extended temporal evaluation, calibration, explainability, fairness and reporting workflow. Run all cells; saved outputs are not proof of a complete run.
- `src/data.py` — loads OULAD CSVs and creates features filtered to a documented day cutoff.
- `src/build_features.py` — writes a prepared feature CSV and data-quality report.
- `src/train.py` — student-grouped baseline evaluation and final pipeline artifact.
- `app.py` — Streamlit CSV-scoring demo.
- `tests/` — lightweight tests using synthetic data; these do not replace validation on the full dataset.
- `cleaned_AI_Powered_Student_Performance_Prediction_and_Success_Analytics.pptx` — presentation.
- `AI_POWERED_STUDENT_PERFORMANCE_AND_SUCCESS_PREDICTION_SYSTEM.docx` — project document.

## Dataset setup

The feature pipeline expects these OULAD CSV files under `data/raw/`:

- `studentInfo.csv`
- `studentAssessment.csv`
- `assessments.csv`
- `studentVle.csv`

Obtain the dataset from its official distribution and review its usage conditions. Never commit raw student-level data to this repository unless permission and necessity have been verified.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducible baseline pipeline

From the repository root:

```bash
python src/build_features.py --data-dir data/raw --cutoff-day 28
python src/train.py --features data/processed/features_day28.csv
pytest -q
streamlit run app.py
```

The build command also writes `reports/data_quality.json`; training writes `reports/metrics.json` and `models/student_risk_pipeline.joblib`. These outputs are generated locally and ignored by Git.

The model uses only input columns present in the feature CSV, excluding `At_Risk` and `id_student`. Keep the student ID for grouped evaluation and identification only; do not use it as a predictive feature. The app accepts a CSV with the same predictors used in training. Remove direct identifiers before using it in any shared demo.

## Important research caveats

- The label is based on final result (`Fail`/`Withdrawn` = at-risk; `Pass`/`Distinction` = not-at-risk). Predictors must be available at the actual prediction cutoff. The baseline filters assessment definitions and VLE events by date, but the intended cutoff and source dataset date semantics must be verified against OULAD documentation before making operational claims.
- The grouped holdout prevents the same student ID from being split across train and test, but it is **not** a future-presentation holdout. Evaluate on a genuinely later presentation before claiming future-cohort generalization.
- Risk bands in `app.py` are illustrative bins, not calibrated operational thresholds. Scores should not be treated as calibrated probabilities until calibration is evaluated.
- Check data quality, group-level error rates, missingness and subgroup fairness. Use predictions to offer voluntary support, never as an automatic basis for penalties or access restrictions.
- The second notebook includes research outputs (SHAP/LIME, timing, fairness, ablation, confidence intervals). Its saved execution metadata shows many cells not completed; run from top to bottom and fix failures before citing its results.

## Current completion status

The repository structure and baseline code are in place. **The full data pipeline, publication notebook and interactive demo are not yet verified end-to-end on the full dataset.** Do not claim the project is fully complete until the generated metrics have been reviewed and the project document/presentation have been reconciled with those results.
