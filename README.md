# AI-Powered Student Performance & Success Prediction

A machine-learning project to study student outcomes and identify learners who may benefit from timely academic support.

> **Status:** The repository currently contains an exploratory notebook. The saved notebook results are preliminary; validate features at the intended prediction cutoff and re-evaluate with student-grouped/future-cohort splits before presenting the metrics as real-world early-warning performance.

## Repository contents

- `MLProject.ipynb` — current data processing, model training, metrics and plots.
- `cleaned_AI_Powered_Student_Performance_Prediction_and_Success_Analytics.pptx` — project presentation.
- `AI_POWERED_STUDENT_PERFORMANCE_AND_SUCCESS_PREDICTION_SYSTEM.docx` — project document.
- `PROJECT_REVIEW_AND_ACTION_PLAN.md` — code review, priority fixes, implementation phases and definition of done.

## Current workflow

1. Load student information, assessment, and virtual learning environment (VLE) activity data.
2. Define a binary outcome: `Withdrawn`/`Fail` → at-risk; `Pass`/`Distinction` → not-at-risk.
3. Aggregate student/course features.
4. Train Logistic Regression, Random Forest, and CatBoost.
5. Compare precision, recall, F1, accuracy and ROC-AUC; inspect feature importance and confusion matrix.

## Dataset setup

The notebook currently expects these CSV files in its working directory:

- `studentInfo.csv`
- `studentAssessment.csv`
- `assessments.csv`
- `studentVle.csv`

The notebook appears designed for the Open University Learning Analytics Dataset (OULAD). Obtain the dataset through its official source and check its licence and usage conditions. Do not commit raw data containing student-level information to this repository unless you have verified permission and necessity. Put downloaded data in a local `data/` directory and adjust notebook paths if desired.

## Run in Google Colab

1. Open [MLProject.ipynb](./MLProject.ipynb) in Google Colab.
2. Upload the four required CSV files when prompted, or change the notebook to read from a mounted Drive/data directory.
3. Run cells from top to bottom.
4. Save the generated metric tables and plots after you have corrected the prediction cutoff and validation split described below.

## Run locally

Create and activate a virtual environment, then install the main notebook dependencies:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn catboost jupyter
jupyter notebook MLProject.ipynb
```

Use a Python version supported by the installed package releases. For reproducibility, pin tested versions in a `requirements.txt` after the notebook has been run successfully in a clean environment.

## Important validation before submission

The target uses the final course result. The current notebook's `mean_score`, `max_score`, and `assessment_count` features are aggregated across assessment rows and may include information unavailable at the time an early-warning prediction would be made. This can cause target leakage. If the project claims to predict risk by day 28, enforce a day-28 cutoff for every predictor and exclude later data.

A random row split can also allow records for the same student to appear in both training and test sets. Compare performance using a student-grouped split and, if predicting future intakes, a chronological/presentation holdout. Recompute all metrics, risk thresholds and feature importance after making these changes.

See [PROJECT_REVIEW_AND_ACTION_PLAN.md](./PROJECT_REVIEW_AND_ACTION_PLAN.md) for the complete checklist.

## Responsible use

Use risk estimates to prioritize supportive outreach, not to penalize students or make high-impact decisions automatically. Explain the limits of historical data, evaluate errors and group-level performance, protect student privacy, and ensure that probabilities and risk labels are validated before showing them to users.
