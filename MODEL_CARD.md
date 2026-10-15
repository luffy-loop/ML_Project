# Model Card: Student Success Early-Warning Baseline

## Summary

This repository provides a research prototype that uses early-course OULAD activity and assessment features to estimate whether a student's eventual course result is Fail or Withdrawn rather than Pass or Distinction. The target is a retrospective proxy; it is not a definitive measure of a student's ability or future.

## Intended use

- Educational research and classroom demonstrations.
- Exploring whether information available by a stated course-day cutoff may help prioritize voluntary academic support.
- Comparing model performance and error patterns before considering any real-world pilot.

## Out-of-scope use

- Automatic grading, discipline, exclusion, admissions, funding, or access decisions.
- Treating scores as calibrated probabilities or labels as definitive judgments about a student.
- Deployment without review by educators, affected students, and the institution's data-protection process.

## Data and prediction timing

The feature builder uses OULAD student information, assessment submissions, assessment definitions, and VLE interactions. The default feature cutoff is day 28. Assessment features are included only when both the scheduled assessment date and recorded submission date are at or before the cutoff; VLE interactions are also cutoff-filtered. The project must still validate dataset date semantics and withdrawal timing before claiming that every feature is available before the outcome is determined.

The current label is based on final course result. In particular, the Withdrawn label may reflect a withdrawal that occurred before or near the prediction cutoff. The current baseline does not establish a valid prospective withdrawal forecast. A future version should incorporate registration/unregistration timing and define a prospective target explicitly.

## Evaluation

The training script reports metrics from a student-grouped holdout so a student's rows do not appear in both the training and test partitions. It includes accuracy, precision, recall, F1, ROC-AUC (when defined), PR-AUC, a Brier score, candidate-threshold workload trade-offs, subgroup error summaries, and held-out permutation importance. A separate later-presentation holdout is still required to evaluate future-cohort generalization.

Threshold values (0.25, 0.50, and 0.75) are illustrative comparisons, not validated operating points. Subgroup summaries omit groups smaller than five in the held-out data and may be statistically unstable. They are not a fairness certification. Permutation importance is predictive association, not causal explanation.

## Known limitations and risks

- The complete pipeline and publication notebook must be run on the full official dataset before citing project metrics as final.
- A random grouped split does not establish temporal generalization.
- Final-result labels can encode processes that have already occurred by the prediction cutoff.
- Missingness and participation patterns may differ across groups and may affect predictions.
- Risk bands in the Streamlit app are provisional; the Brier score alone does not prove calibration.
- Predictions can create stigma or divert resources if interpreted as facts rather than uncertain signals.

## Responsible use

Use predictions only to offer supportive, voluntary outreach with a human educator in the loop. Explain uncertainty, protect student privacy, minimize identifiers in demo files, review subgroup errors, and allow students to contest or contextualize automated signals. Do not use this prototype to impose penalties or restrict opportunities.

## Reproducibility

Follow the setup and pipeline commands in README.md. The automated tests use synthetic data to check code contracts; they do not validate model quality on the real dataset. Record the dataset version, cutoff, target definition, split, metrics, and environment when publishing results.
