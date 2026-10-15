# AI-Powered Student Performance & Success Prediction — Review and Action Plan

This review is based on the current `MLProject.ipynb` notebook and the file inventory in the repository. The PPTX and DOCX are present in the repository but their binary contents could not be extracted through this inspection path, so this document does not claim to be a slide-by-slide requirements audit.

## What is already implemented

- Loads the Open University Learning Analytics Dataset (OULAD)-style files: `studentInfo.csv`, `studentAssessment.csv`, `assessments.csv`, and `studentVle.csv`.
- Defines a binary target: `Withdrawn` and `Fail` become at-risk; `Pass` and `Distinction` become not-at-risk.
- Aggregates assessment score summaries and first-28-day VLE click/activity features.
- Trains and compares Logistic Regression, Random Forest, and CatBoost.
- Reports accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, feature importance, and risk bands.

The saved notebook output reports a random stratified test split with CatBoost accuracy about 0.927, F1 about 0.928, and ROC-AUC about 0.978. Treat those numbers as preliminary, not validated evidence of early-warning performance.

## Highest-priority issues to fix

### 1. Prevent target leakage and define the prediction time clearly

The target is derived from the final outcome (`final_result`). Yet the current features include `mean_score`, `max_score`, and `assessment_count` aggregated across the student’s assessment record, potentially including information recorded after the intended intervention time. The high importance of `assessment_count` is a signal to investigate, not proof by itself.

Decide what the system predicts and when:
- **Early warning (recommended):** predict final failure/withdrawal using only information available by a documented cutoff, e.g. day 28. Exclude all later assessment and VLE activity.
- **End-of-course analytics:** may use final-course activity, but must not be described as early prediction.

For a day-28 prediction, construct all features strictly from data available on or before day 28; join outcomes only as labels. Confirm date conventions in the dataset documentation.

### 2. Validate the split strategy

Rows represent student-module-presentation records. A random row split can put records from the same student or related course/presentation patterns into both train and test sets. Add a stronger evaluation:
- Group split by student ID, so a student's rows do not cross the train/test boundary.
- If the intended use is future cohorts, do a chronological or presentation-based holdout.
- Keep a final untouched test set and tune thresholds only on training/validation data.

### 3. Make feature aggregation robust

- Keep a data-quality report for malformed VLE rows and count how many are skipped by `on_bad_lines="skip"`; silent skipping can bias engagement features.
- Check missingness after joins and report how many students have no assessment/VLE features.
- Validate merge keys and duplicated student-module-presentation rows.
- Keep `id_student` as a grouping/splitting key, not a predictive feature.
- Check target balance and any differences by course/presentation before modelling.

### 4. Report decision-relevant metrics

Keep ROC-AUC, but also report:
- Precision, recall, F1 and PR-AUC for the at-risk class.
- Confusion matrix with clear actual/predicted axis labels.
- Calibration or reliability curve if probabilities are displayed as risk probabilities.
- Recall at a chosen intervention capacity (e.g. top 10% or 20% highest-risk students).
- A threshold chosen on validation data, not arbitrary labels alone.

The current SAFE/WARNING/AT-RISK/CRITICAL cutoffs (0.25, 0.50, 0.75) are hand-set and should be validated against calibration and intervention needs.

### 5. Fairness, privacy, and responsible use

- Avoid framing predictions as deterministic outcomes. Use them to offer support, not to punish or restrict students.
- Evaluate false-positive/false-negative rates across relevant demographic groups where lawful and appropriate.
- Minimize identifying data in outputs and do not display student IDs in any public demo.
- Document limitations, dataset provenance, missing/incorrect records, and that results on historical data may not generalize to a live institution.

## Recommended project structure

Move from one large notebook to a reproducible structure:

```
ML_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md              # dataset download/setup; do not commit raw personal data
├── notebooks/
│   └── MLProject.ipynb        # exploration and figures
├── src/
│   ├── data.py                # load, validate, merge
│   ├── features.py            # time-cutoff feature generation
│   ├── train.py               # model training and validation
│   ├── evaluate.py            # metrics, confusion matrix, calibration
│   └── predict.py             # load model and predict
├── models/                    # generated model files; generally ignored by git
├── reports/
│   └── figures/
└── app.py                     # optional demo dashboard (Streamlit)
```

## Implementation plan

### Phase 1 — Make results reliable
1. Write down the exact prediction question and cutoff date.
2. Rebuild features to respect that cutoff; do not use later assessment scores.
3. Add a student-grouped split and a chronological/cohort holdout.
4. Compare Logistic Regression, Random Forest, and CatBoost on the same splits.
5. Generate a single metrics table, confusion matrix, PR curve, calibration plot, and error analysis.
6. Re-check the baseline and update all reported values in the notebook and presentation.

### Phase 2 — Make the project reproducible
1. Add `requirements.txt` with tested package versions.
2. Add data setup instructions; keep large/raw data outside Git where appropriate.
3. Extract reusable data loading, feature creation, training, and evaluation into `src/`.
4. Save the final pipeline/model with version and training metadata.
5. Add a small smoke-test dataset or test code for target creation, cutoff enforcement, and prediction schema.

### Phase 3 — Build the user-facing demo (if required by the PPT)
1. Create a Streamlit interface for either a CSV upload or a single student/course record.
2. Show risk probability, a validated risk category, and the top contributing factors.
3. Add a plain-language note that this is a support aid, not a definitive judgment.
4. Use anonymized/sample records only and document local run instructions.

### Phase 4 — Align final submission and presentation
1. Replace preliminary random-split scores with leakage-resistant evaluation results.
2. Include the problem statement, objectives, dataset/schema, architecture, pipeline, model comparison, results, limitations, ethics, and future work.
3. Ensure every result shown in the PPT can be reproduced by a notebook/script in the repository.
4. Add screenshots of the actual working demo only after it runs.
5. Add a clear README with setup commands, dataset acquisition instructions, and reproducible run steps.

## Definition of done

- [ ] Prediction cutoff is explicit and enforced in feature engineering.
- [ ] No post-cutoff features are used in an early-warning model.
- [ ] Student-grouped and/or future-cohort evaluation is reported.
- [ ] Metrics include at-risk recall, precision, F1, ROC-AUC, PR-AUC and a confusion matrix.
- [ ] Probability calibration and risk thresholds are justified or risk scores are not presented as calibrated probabilities.
- [ ] Data quality and missingness are quantified.
- [ ] Model, preprocessing and feature-generation logic can be rerun from a clean environment.
- [ ] README documents data setup, installation, training, evaluation and demo launch.
- [ ] Ethical use, privacy, dataset limits and fairness checks are documented.
- [ ] Presentation claims match the final, reproducible outputs.

## Current evidence and caution

The notebook's saved random-split results are:
- Logistic Regression: accuracy 0.808, F1 0.813, ROC-AUC 0.894.
- Random Forest: accuracy 0.922, F1 0.924, ROC-AUC 0.973.
- CatBoost: accuracy 0.927, F1 0.928, ROC-AUC 0.978.

These are copied from existing notebook outputs, not results from the recommended corrected evaluation. Do not put them in the final PPT as validated early-warning metrics until the leakage and split issues have been addressed.
